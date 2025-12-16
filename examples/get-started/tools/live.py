import sys
import csv
import json
import argparse
import numpy as np
import pickle
import serial
from io import StringIO
from threading import Lock
from collections import deque, Counter
from scipy import signal
from scipy.stats import skew, kurtosis

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
import pyqtgraph as pg

# Config
PACKET_RATE = 20
WINDOW_SEC = 2.0
STEP_SEC = 0.5

WINDOW_SIZE = int(WINDOW_SEC * PACKET_RATE)
PLOT_HISTORY = 100  # Number of packets to show in plot

# Display labels
DISPLAY_LABELS = {0: "EMPTY", 1: "ACTIVE"}
# DISPLAY_LABELS = {0: "EMPTY", 1: "ACTIVE", 2: "STILL"}  # 3-class version

# Colors for each receiver
RX_COLORS = [
    (255, 100, 100),  # Red
    (100, 255, 100),  # Green
    (100, 100, 255),  # Blue
]

DEFAULT_FEATURE_ORDER = ['mean', 'std', 'var', 'range', 'skew', 'kurtosis', 
                         'diff_mean', 'diff_std', 'spatial_var', 
                         'spectral_energy', 'motion_energy']

DATA_COLUMNS_NAMES_C5C6 = ["type", "id", "mac", "rssi", "rate", "noise_floor", 
                           "fft_gain", "agc_gain", "channel", "local_timestamp",
                           "sig_len", "rx_state", "len", "first_word", "data"]


def bandpass(data, fs, low=0.1, high=5.0):
    if len(data) < 20:
        return data
    try:
        b, a = signal.butter(4, [low/(fs/2), high/(fs/2)], btype='band')
        return signal.filtfilt(b, a, data)
    except:
        return data


def preprocess_window(csi_buffer, agc_buffer, fft_buffer):
    n_pkt = len(csi_buffer)
    if n_pkt == 0:
        return None, None
    
    n_sc = len(csi_buffer[0])
    amp = np.zeros((n_pkt, n_sc))
    phase = np.zeros((n_pkt, n_sc))
    
    for i in range(n_pkt):
        csi = csi_buffer[i]
        gain_db = agc_buffer[i] + fft_buffer[i]
        
        # Amplitude with gain compensation
        a = np.abs(csi)
        if gain_db != 0:
            a = a / (10 ** (gain_db / 20))
        amp[i, :len(a)] = a
        
        # Phase extraction (no gain compensation needed)
        phase[i, :len(csi)] = np.angle(csi)
    
    # Apply bandpass filter to amplitude
    if n_pkt > 20:
        for j in range(n_sc):
            amp[:, j] = bandpass(amp[:, j], PACKET_RATE)
    
    return amp, phase


def extract_features(amp, phase):
    if amp is None or len(amp) == 0:
        return None
    
    f = {}
    
    # amplitude features
    f['mean'] = np.mean(amp)
    f['std'] = np.std(amp)
    f['var'] = np.var(amp)
    f['range'] = np.max(amp) - np.min(amp)
    f['skew'] = skew(amp.flatten())
    f['kurtosis'] = kurtosis(amp.flatten())
    
    diff = np.diff(amp, axis=0)
    f['diff_mean'] = np.mean(np.abs(diff))
    f['diff_std'] = np.std(diff)
    f['spatial_var'] = np.mean(np.var(amp, axis=1))
    
    sig = np.mean(amp, axis=1)
    if len(sig) > 8:
        fft_mag = np.abs(np.fft.rfft(sig - np.mean(sig)))
        freqs = np.fft.rfftfreq(len(sig), 1/PACKET_RATE)
        f['spectral_energy'] = np.sum(fft_mag**2)
        mask = (freqs >= 0.1) & (freqs <= 2.0)
        f['motion_energy'] = np.sum(fft_mag[mask]**2) if np.any(mask) else 0
    else:
        f['spectral_energy'] = 0
        f['motion_energy'] = 0
    
    # phase features
    # Unwrap phase to handle 2π discontinuities
    phase_unwrapped = np.unwrap(phase, axis=1)
    
    # Phase variance (indicates signal instability/multipath changes)
    f['phase_var'] = np.var(phase_unwrapped)
    f['phase_std'] = np.std(phase_unwrapped)
    
    # Temporal phase changes (movement causes phase shifts)
    phase_diff = np.diff(phase_unwrapped, axis=0)
    f['phase_diff_mean'] = np.mean(np.abs(phase_diff))
    f['phase_diff_std'] = np.std(phase_diff)
    
    # Spatial phase consistency across subcarriers
    f['phase_spatial_var'] = np.mean(np.var(phase_unwrapped, axis=1))
    
    # Phase range (total phase variation)
    f['phase_range'] = np.max(phase_unwrapped) - np.min(phase_unwrapped)
    
    return f


class CSIBuffer:
    def __init__(self):
        self.lock = Lock()
        self.csi_buffer = deque(maxlen=WINDOW_SIZE * 2)
        self.agc_buffer = deque(maxlen=WINDOW_SIZE * 2)
        self.fft_buffer = deque(maxlen=WINDOW_SIZE * 2)
        # Separate buffer for plotting (stores raw amplitudes)
        self.plot_buffer = deque(maxlen=PLOT_HISTORY)
        self.n_subcarriers = 0
    
    def add_packet(self, csi_complex, agc_gain, fft_gain):
        with self.lock:
            self.csi_buffer.append(csi_complex.copy())
            self.agc_buffer.append(agc_gain)
            self.fft_buffer.append(fft_gain)
            # Store amplitude for plotting
            self.plot_buffer.append(np.abs(csi_complex))
            self.n_subcarriers = len(csi_complex)
    
    def get_window(self):
        with self.lock:
            if len(self.csi_buffer) < WINDOW_SIZE:
                return None, None, None
            return (list(self.csi_buffer)[-WINDOW_SIZE:],
                    list(self.agc_buffer)[-WINDOW_SIZE:],
                    list(self.fft_buffer)[-WINDOW_SIZE:])
    
    def get_plot_data(self):
        """Get amplitude data for plotting"""
        with self.lock:
            if len(self.plot_buffer) == 0:
                return None
            return np.array(list(self.plot_buffer))
    
    def ready(self):
        with self.lock:
            return len(self.csi_buffer) >= WINDOW_SIZE
    
    def size(self):
        with self.lock:
            return len(self.csi_buffer)


class SerialReader(QThread):
    packet_received = pyqtSignal()
    
    def __init__(self, port, buffer):
        super().__init__()
        self.port = port
        self.buffer = buffer
        self.running = True
        self.packet_count = 0
    
    def run(self):
        try:
            ser = serial.Serial(port=self.port, baudrate=921600, timeout=1.0)
            print(f"Connected: {self.port}")
        except Exception as e:
            print(f"Failed: {self.port} - {e}")
            return
        
        while self.running:
            try:
                line = ser.readline()
                if not line:
                    continue
                
                strings = line.decode('utf-8', errors='ignore').strip()
                if 'CSI_DATA' not in strings:
                    continue
                
                csi_data = next(csv.reader(StringIO(strings)))
                if len(csi_data) != len(DATA_COLUMNS_NAMES_C5C6):
                    continue
                
                csi_raw = json.loads(csi_data[-1])
                csi_len = len(csi_raw) // 2
                csi_complex = np.array([complex(csi_raw[i*2+1], csi_raw[i*2]) for i in range(csi_len)], dtype=np.complex64)
                
                agc = float(csi_data[7])
                fft = float(csi_data[6])
                
                self.buffer.add_packet(csi_complex, agc, fft)
                self.packet_count += 1
                self.packet_received.emit()
            except:
                continue
        ser.close()
    
    def stop(self):
        self.running = False


class InferenceEngine:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            saved = pickle.load(f)
        
        if isinstance(saved, dict):
            self.model = saved.get('model')
            self.scaler = saved.get('scaler')
            self.feature_names = saved.get('feature_names', DEFAULT_FEATURE_ORDER)
            self.labels = saved.get('labels', DISPLAY_LABELS)
            
            config = saved.get('config', {})
            trained_rate = config.get('PACKET_RATE', 'unknown')
            print(f"\n MODEL INFO")
            print(f"Model trained with PACKET_RATE: {trained_rate}")
            print(f"Live.py using PACKET_RATE: {PACKET_RATE}")
            if trained_rate != 'unknown' and trained_rate != PACKET_RATE:
                print(f"WARNING: PACKET_RATE mismatch!")
        else:
            self.model = saved
            self.scaler = None
            self.feature_names = DEFAULT_FEATURE_ORDER
            self.labels = DISPLAY_LABELS
    
    def predict(self, csi, agc, fft):
        amp, phase = preprocess_window(csi, agc, fft)
        if amp is None:
            return None, 0.0, None
        
        features = extract_features(amp, phase)
        if features is None:
            return None, 0.0, None
        
        X = np.array([[features.get(name, 0) for name in self.feature_names]])
        if self.scaler:
            X = self.scaler.transform(X)
        
        pred = self.model.predict(X)[0]
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            conf = np.max(proba)
        else:
            proba = None
            conf = 1.0
        
        return pred, conf, features


class LiveWindow(QWidget):
    def __init__(self, model_path, ports):
        super().__init__()
        self.ports = ports
        self.setWindowTitle(f"CSI Live - {len(ports)} Receivers")
        
        self.engine = InferenceEngine(model_path)
        self.pred_history = deque(maxlen=5)
        
        # Multiple buffers and readers for each receiver
        self.buffers = [CSIBuffer() for _ in ports]
        self.readers = []
        
        # Main layout: plots on left, prediction panel on right
        main_layout = QHBoxLayout()
        
        # === LEFT SIDE: CSI Plots ===
        plots_widget = QWidget()
        plots_layout = QVBoxLayout(plots_widget)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setSpacing(2)
        
        self.plot_widgets = []
        self.plot_curves = []
        
        # Subcarriers to plot (every 16th for performance)
        self.subcarrier_indices = list(range(0, 256, 16))
        
        for i, port in enumerate(ports):
            color = RX_COLORS[i % len(RX_COLORS)]
            
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground('#1e1e1e')
            plot_widget.setTitle(f"RX{i+1}: {port}", color='w', size='10pt')
            plot_widget.setLabel('left', 'Amplitude', color='#888')
            plot_widget.setLabel('bottom', 'Time (packets)', color='#888')
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            plot_widget.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)
            plot_widget.setMinimumHeight(150)
            
            # Create curves for selected subcarriers
            curves = []
            for sc_idx in self.subcarrier_indices:
                curve = plot_widget.plot([], pen=pg.mkPen(color=color, width=1))
                curves.append(curve)
            
            self.plot_widgets.append(plot_widget)
            self.plot_curves.append(curves)
            plots_layout.addWidget(plot_widget)
        
        main_layout.addWidget(plots_widget, stretch=3)
        
        # prediction panel
        pred_panel = QFrame()
        pred_panel.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border-left: 2px solid #3c3c3c;
            }
        """)
        pred_panel.setFixedWidth(280)
        
        pred_layout = QVBoxLayout(pred_panel)
        pred_layout.setContentsMargins(15, 20, 15, 20)
        
        # Title
        title_label = QLabel("PREDICTION")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 12px; color: #888; font-weight: bold; letter-spacing: 2px;")
        pred_layout.addWidget(title_label)
        
        pred_layout.addSpacing(20)
        
        # Main prediction label
        self.pred_label = QLabel("...")
        self.pred_label.setAlignment(Qt.AlignCenter)
        self.pred_label.setStyleSheet("""
            font-size: 42px; 
            font-weight: bold; 
            color: #ffffff;
            padding: 20px;
            background-color: #333333;
            border-radius: 10px;
        """)
        pred_layout.addWidget(self.pred_label)
        
        pred_layout.addSpacing(15)
        
        # Confidence label
        self.conf_label = QLabel("")
        self.conf_label.setAlignment(Qt.AlignCenter)
        self.conf_label.setStyleSheet("font-size: 14px; color: #aaa;")
        pred_layout.addWidget(self.conf_label)
        
        # Vote label
        self.vote_label = QLabel("")
        self.vote_label.setAlignment(Qt.AlignCenter)
        self.vote_label.setStyleSheet("font-size: 12px; color: #666;")
        pred_layout.addWidget(self.vote_label)
        
        pred_layout.addStretch()
        
        # Status label
        self.status_label = QLabel("Connecting...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 11px; color: #666;")
        self.status_label.setWordWrap(True)
        pred_layout.addWidget(self.status_label)
        
        main_layout.addWidget(pred_panel)
        
        self.setLayout(main_layout)
        
        # Window size based on number of receivers
        plot_height = 150 * len(ports)
        self.resize(1000, max(400, plot_height))
        
        # Dark theme for window
        self.setStyleSheet("background-color: #1e1e1e;")
        
        # Start readers for all receivers
        for i, port in enumerate(ports):
            reader = SerialReader(port, self.buffers[i])
            reader.packet_received.connect(self.on_packet)
            reader.start()
            self.readers.append(reader)
        
        # Timer for predictions
        self.pred_timer = QTimer()
        self.pred_timer.timeout.connect(self.run_prediction)
        self.pred_timer.start(int(STEP_SEC * 1000))
        
        # Timer for plot updates (faster than predictions)
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plots)
        self.plot_timer.start(100)  # Update plots every 100ms
    
    def on_packet(self):
        # Show buffer status for all receivers
        statuses = []
        total_packets = 0
        for i, buf in enumerate(self.buffers):
            pct = min(100, int(buf.size() / WINDOW_SIZE * 100))
            statuses.append(f"R{i+1}:{pct}%")
            total_packets += self.readers[i].packet_count
        self.status_label.setText(f"Buffers: {' | '.join(statuses)}\nTotal packets: {total_packets}")
    
    def update_plots(self):
        """Update the CSI amplitude plots"""
        for i, buf in enumerate(self.buffers):
            plot_data = buf.get_plot_data()
            if plot_data is None or len(plot_data) == 0:
                continue
            
            n_packets, n_sc = plot_data.shape
            
            for j, sc_idx in enumerate(self.subcarrier_indices):
                if j < len(self.plot_curves[i]) and sc_idx < n_sc:
                    self.plot_curves[i][j].setData(plot_data[:, sc_idx])
    
    def run_prediction(self):
        # Collect predictions from all ready receivers
        predictions = []
        confidences = []
        all_features = []
        
        for i, buf in enumerate(self.buffers):
            if not buf.ready():
                continue
            
            csi, agc, fft = buf.get_window()
            if csi is None:
                continue
            
            pred, conf, features = self.engine.predict(csi, agc, fft)
            
            if pred is not None and features is not None:
                predictions.append(pred)
                confidences.append(conf)
                all_features.append((i+1, pred, conf, features))
                print(f"R{i+1}: var={features['var']:.6f}, std={features['std']:.6f} -> pred={pred}, conf={conf:.2f}")
        
        if not predictions:
            self.vote_label.setText("Waiting for predictions...")
            return
        
        # Majority vote across receivers
        vote_counts = Counter(predictions)
        voted_pred = vote_counts.most_common(1)[0][0]
        vote_count = vote_counts[voted_pred]
        total_votes = len(predictions)
        avg_conf = np.mean(confidences)
        
        # Add to history for temporal smoothing
        self.pred_history.append(voted_pred)
        
        # Smooth over recent predictions
        if len(self.pred_history) >= 3:
            smoothed = Counter(self.pred_history).most_common(1)[0][0]
        else:
            smoothed = voted_pred
        
        label = DISPLAY_LABELS.get(smoothed, f"Class {smoothed}")
        self.pred_label.setText(label)
        
        # Color-code the prediction
        if smoothed == 0:  # NO person
            self.pred_label.setStyleSheet("""
                font-size: 42px; font-weight: bold; color: #4CAF50;
                padding: 20px; background-color: #1b3d1b; border-radius: 10px;
            """)
        else:  # YES person (covers class 1, or 1/2 for 3-class)
            self.pred_label.setStyleSheet("""
                font-size: 42px; font-weight: bold; color: #FF5722;
                padding: 20px; background-color: #3d1b1b; border-radius: 10px;
            """)
        
        self.conf_label.setText(f"{int(avg_conf * 100)}% confidence")
        self.vote_label.setText(f"Vote: {vote_count}/{total_votes} receivers")
    
    def closeEvent(self, event):
        for reader in self.readers:
            reader.stop()
        for reader in self.readers:
            reader.wait()
        event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help="Model file (.pkl)")
    parser.add_argument('-p', '--ports', required=True, nargs='+', 
                        help="Serial ports (one or more receivers, e.g., -p COM3 COM4 COM5)")
    args = parser.parse_args()
    
    print(f"\n Multi-Receiver CSI Live ")
    print(f"Using {len(args.ports)} receivers: {', '.join(args.ports)}")
    print(f"Majority vote across receivers determines final prediction\n")
    
    app = QApplication(sys.argv)
    window = LiveWindow(args.model, args.ports)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
