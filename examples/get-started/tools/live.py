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

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

# Config
PACKET_RATE = 20
WINDOW_SEC = 2.0
STEP_SEC = 0.5

WINDOW_SIZE = int(WINDOW_SEC * PACKET_RATE)

# Display labels
DISPLAY_LABELS = {0: "NO", 1: "YES"}

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
        return None
    
    n_sc = len(csi_buffer[0])
    amp = np.zeros((n_pkt, n_sc))
    
    for i in range(n_pkt):
        a = np.abs(csi_buffer[i])
        gain_db = agc_buffer[i] + fft_buffer[i]
        if gain_db != 0:
            a = a / (10 ** (gain_db / 20))
        amp[i, :len(a)] = a
    
    if n_pkt > 20:
        for j in range(n_sc):
            amp[:, j] = bandpass(amp[:, j], PACKET_RATE)
    
    return amp


def extract_features(amp):
    if amp is None or len(amp) == 0:
        return None
    
    f = {}
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
    
    return f


class CSIBuffer:
    def __init__(self):
        self.lock = Lock()
        self.csi_buffer = deque(maxlen=WINDOW_SIZE * 2)
        self.agc_buffer = deque(maxlen=WINDOW_SIZE * 2)
        self.fft_buffer = deque(maxlen=WINDOW_SIZE * 2)
    
    def add_packet(self, csi_complex, agc_gain, fft_gain):
        with self.lock:
            self.csi_buffer.append(csi_complex.copy())
            self.agc_buffer.append(agc_gain)
            self.fft_buffer.append(fft_gain)
    
    def get_window(self):
        with self.lock:
            if len(self.csi_buffer) < WINDOW_SIZE:
                return None, None, None
            return (list(self.csi_buffer)[-WINDOW_SIZE:],
                    list(self.agc_buffer)[-WINDOW_SIZE:],
                    list(self.fft_buffer)[-WINDOW_SIZE:])
    
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
            print(f"\n=== MODEL INFO ===")
            print(f"Model trained with PACKET_RATE: {trained_rate}")
            print(f"Live.py using PACKET_RATE: {PACKET_RATE}")
            if trained_rate != 'unknown' and trained_rate != PACKET_RATE:
                print(f"WARNING: PACKET_RATE mismatch!")
        else:
            self.model = saved
            self.scaler = None
            self.feature_names = DEFAULT_FEATURE_ORDER
            self.labels = DISPLAY_LABELS
        
        print(f"Model: {type(self.model).__name__}")
        print(f"==================\n")
    
    def predict(self, csi, agc, fft):
        amp = preprocess_window(csi, agc, fft)
        if amp is None:
            return None, 0.0, None
        
        features = extract_features(amp)
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
    def __init__(self, model_path, port):
        super().__init__()
        self.setWindowTitle(f"CSI Live - {port}")
        self.setFixedSize(300, 150)
        
        self.engine = InferenceEngine(model_path)
        self.pred_history = deque(maxlen=5)
        self.buffer = CSIBuffer()
        
        layout = QVBoxLayout()
        
        self.pred_label = QLabel("Loading...")
        self.pred_label.setAlignment(Qt.AlignCenter)
        self.pred_label.setStyleSheet("font-size: 48px; font-weight: bold;")
        layout.addWidget(self.pred_label)
        
        self.conf_label = QLabel("")
        self.conf_label.setAlignment(Qt.AlignCenter)
        self.conf_label.setStyleSheet("font-size: 14px; color: gray;")
        layout.addWidget(self.conf_label)
        
        self.status_label = QLabel("Buffer: 0%")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 12px; color: gray;")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        
        # Single receiver only
        self.reader = SerialReader(port, self.buffer)
        self.reader.packet_received.connect(self.on_packet)
        self.reader.start()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.run_prediction)
        self.timer.start(int(STEP_SEC * 1000))
    
    def on_packet(self):
        pct = min(100, int(self.buffer.size() / WINDOW_SIZE * 100))
        self.status_label.setText(f"Buffer: {pct}% | Packets: {self.reader.packet_count}")
    
    def run_prediction(self):
        if not self.buffer.ready():
            return
        
        csi, agc, fft = self.buffer.get_window()
        if csi is None:
            return
        
        pred, conf, features = self.engine.predict(csi, agc, fft)
        
        if pred is not None:
            # Debug print
            print(f"var={features['var']:.6f}, std={features['std']:.6f}, range={features['range']:.4f} -> pred={pred}, conf={conf:.2f}")
            
            self.pred_history.append(pred)
            
            if len(self.pred_history) >= 3:
                smoothed = Counter(self.pred_history).most_common(1)[0][0]
            else:
                smoothed = pred
            
            label = DISPLAY_LABELS.get(smoothed, f"Class {smoothed}")
            self.pred_label.setText(label)
            self.conf_label.setText(f"{int(conf * 100)}% confidence")
    
    def closeEvent(self, event):
        self.reader.stop()
        self.reader.wait()
        event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help="Model file (.pkl)")
    parser.add_argument('-p', '--port', required=True, help="Serial port (single receiver)")
    args = parser.parse_args()
    
    print(f"Using single receiver: {args.port}")
    print("(Training was done per-receiver, so live should use one receiver too)")
    
    app = QApplication(sys.argv)
    window = LiveWindow(args.model, args.port)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
