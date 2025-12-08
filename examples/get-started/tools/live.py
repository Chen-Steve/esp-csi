import sys
import csv
import json
import argparse
import numpy as np
import pickle
import serial
from io import StringIO
from threading import Lock
from collections import deque
from datetime import datetime
from scipy import signal
from scipy.stats import skew, kurtosis
import sklearn

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
    QFrame, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont

# Config
PACKET_RATE = 100
WINDOW_SEC = 2.0
STEP_SEC = 0.5

WINDOW_SIZE = int(WINDOW_SEC * PACKET_RATE)
STEP_SIZE = int(STEP_SEC * PACKET_RATE)

LABELS = {0: "empty", 1: "person"}

DATA_COLUMNS_NAMES_C5C6 = ["type", "id", "mac", "rssi", "rate", "noise_floor", 
                           "fft_gain", "agc_gain", "channel", "local_timestamp",
                           "sig_len", "rx_state", "len", "first_word", "data"]

# Colors
DARK_BG = "#1e1e2e"
CARD_BG = "#2a2a3e"
TEXT_PRIMARY = "#ffffff"
TEXT_DIM = "#888888"
GREEN = "#4ade80"
BLUE = "#60a5fa"
YELLOW = "#fbbf24"
RED = "#f87171"

PREDICTION_COLORS = {0: GREEN, 1: BLUE, 2: YELLOW, 3: RED}


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


FEATURE_ORDER = ['mean', 'std', 'var', 'range', 'skew', 'kurtosis', 
                 'diff_mean', 'diff_std', 'spatial_var', 
                 'spectral_energy', 'motion_energy']


def features_to_array(features):
    if features is None:
        return None
    return np.array([[features.get(name, 0) for name in FEATURE_ORDER]])


class CSIBuffer:
    def __init__(self, max_size=WINDOW_SIZE * 2):
        self.lock = Lock()
        self.csi_buffer = deque(maxlen=max_size)
        self.agc_buffer = deque(maxlen=max_size)
        self.fft_buffer = deque(maxlen=max_size)
        self.packet_count = 0
    
    def add_packet(self, csi_complex, agc_gain, fft_gain):
        with self.lock:
            self.csi_buffer.append(csi_complex.copy())
            self.agc_buffer.append(agc_gain)
            self.fft_buffer.append(fft_gain)
            self.packet_count += 1
    
    def get_window(self, size=WINDOW_SIZE):
        with self.lock:
            if len(self.csi_buffer) < size:
                return None, None, None
            return (list(self.csi_buffer)[-size:],
                    list(self.agc_buffer)[-size:],
                    list(self.fft_buffer)[-size:])
    
    def ready(self):
        with self.lock:
            return len(self.csi_buffer) >= WINDOW_SIZE


csi_buffer = CSIBuffer()


class SerialReaderThread(QThread):
    packet_received = pyqtSignal(int)
    
    def __init__(self, port):
        super().__init__()
        self.port = port
        self.running = True
    
    def run(self):
        global csi_buffer
        
        try:
            ser = serial.Serial(port=self.port, baudrate=921600, 
                               bytesize=8, parity='N', stopbits=1, timeout=1.0)
            print(f"Connected to {self.port}")
        except Exception as e:
            print(f"Failed to connect to {self.port}: {e}")
            return
        
        while self.running:
            try:
                line = ser.readline()
                if not line:
                    continue
                
                strings = line.decode('utf-8', errors='ignore').strip()
                if not strings or 'CSI_DATA' not in strings:
                    continue
                
                csv_reader = csv.reader(StringIO(strings))
                csi_data = next(csv_reader)
                
                if len(csi_data) != len(DATA_COLUMNS_NAMES_C5C6):
                    continue
                
                csi_raw = json.loads(csi_data[-1])
                csi_len = len(csi_raw) // 2
                
                csi_complex = np.array([
                    complex(csi_raw[i*2+1], csi_raw[i*2]) 
                    for i in range(csi_len)
                ], dtype=np.complex64)
                
                fft_gain = float(csi_data[6])
                agc_gain = float(csi_data[7])
                
                csi_buffer.add_packet(csi_complex, agc_gain, fft_gain)
                self.packet_received.emit(csi_buffer.packet_count)
                
            except:
                continue
        
        ser.close()
    
    def stop(self):
        self.running = False


class InferenceEngine:
    def __init__(self, model_path):
        self.model = None
        self.scaler = None
        self.load_model(model_path)
    
    def load_model(self, path):
        with open(path, 'rb') as f:
            saved = pickle.load(f)
        
        if isinstance(saved, dict):
            self.model = saved.get('model')
            self.scaler = saved.get('scaler')
        else:
            self.model = saved
            self.scaler = None
        
        print(f"Loaded: {type(self.model).__name__}")
    
    def predict(self, csi_window, agc_window, fft_window):
        if self.model is None:
            return None, 0.0
        
        amp = preprocess_window(csi_window, agc_window, fft_window)
        if amp is None:
            return None, 0.0
        
        features = extract_features(amp)
        if features is None:
            return None, 0.0
        
        X = features_to_array(features)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        pred = self.model.predict(X)[0]
        
        if hasattr(self.model, 'predict_proba'):
            confidence = np.max(self.model.predict_proba(X)[0])
        else:
            confidence = 1.0
        
        return pred, confidence


class LiveWindow(QWidget):
    def __init__(self, model_path, ports):
        super().__init__()
        self.setWindowTitle("CSI Live")
        self.setFixedSize(500, 400)
        self.setStyleSheet(f"background-color: {DARK_BG};")
        
        self.engine = InferenceEngine(model_path)
        self.pred_history = deque(maxlen=5)
        
        self.setup_ui()
        
        self.readers = []
        for port in ports:
            reader = SerialReaderThread(port)
            reader.packet_received.connect(self.on_packet)
            reader.start()
            self.readers.append(reader)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.run_prediction)
        self.timer.start(int(STEP_SEC * 1000))
        
        self.last_count = 0
        self.last_time = datetime.now()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Rate indicator
        rate_layout = QHBoxLayout()
        rate_layout.addStretch()
        
        self.rate_label = QLabel("0 pkt/s")
        self.rate_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 14px;")
        rate_layout.addWidget(self.rate_label)
        
        layout.addLayout(rate_layout)
        
        # Main prediction
        self.pred_label = QLabel("—")
        self.pred_label.setAlignment(Qt.AlignCenter)
        self.pred_label.setStyleSheet(f"""
            color: {TEXT_PRIMARY};
            font-size: 72px;
            font-weight: bold;
            background-color: {CARD_BG};
            border-radius: 16px;
            padding: 40px;
        """)
        layout.addWidget(self.pred_label)
        
        # Confidence
        conf_layout = QVBoxLayout()
        conf_layout.setSpacing(8)
        
        conf_header = QHBoxLayout()
        conf_title = QLabel("Confidence")
        conf_title.setStyleSheet(f"color: {TEXT_DIM}; font-size: 14px;")
        conf_header.addWidget(conf_title)
        
        self.conf_value = QLabel("—")
        self.conf_value.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: bold;")
        conf_header.addStretch()
        conf_header.addWidget(self.conf_value)
        conf_layout.addLayout(conf_header)
        
        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setTextVisible(False)
        self.conf_bar.setFixedHeight(8)
        self.conf_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {CARD_BG};
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background-color: {BLUE};
                border-radius: 4px;
            }}
        """)
        conf_layout.addWidget(self.conf_bar)
        
        layout.addLayout(conf_layout)
        
        # Buffer indicator
        self.buffer_label = QLabel("Buffer: 0%")
        self.buffer_label.setAlignment(Qt.AlignCenter)
        self.buffer_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px;")
        layout.addWidget(self.buffer_label)
        
        self.setLayout(layout)
    
    def on_packet(self, count):
        buffer_pct = min(100, int(len(csi_buffer.csi_buffer) / WINDOW_SIZE * 100))
        self.buffer_label.setText(f"Buffer: {buffer_pct}%")
        
        now = datetime.now()
        elapsed = (now - self.last_time).total_seconds()
        if elapsed >= 1.0:
            rate = (count - self.last_count) / elapsed
            self.rate_label.setText(f"{rate:.0f} pkt/s")
            self.last_count = count
            self.last_time = now
    
    def run_prediction(self):
        if not csi_buffer.ready():
            return
        
        csi, agc, fft = csi_buffer.get_window()
        if csi is None:
            return
        
        pred, conf = self.engine.predict(csi, agc, fft)
        
        if pred is not None:
            self.pred_history.append(pred)
            
            # Majority vote smoothing
            if len(self.pred_history) >= 3:
                from collections import Counter
                smoothed = Counter(self.pred_history).most_common(1)[0][0]
            else:
                smoothed = pred
            
            label = LABELS.get(smoothed, f"Class {smoothed}")
            color = PREDICTION_COLORS.get(smoothed, TEXT_PRIMARY)
            
            self.pred_label.setText(label.upper())
            self.pred_label.setStyleSheet(f"""
                color: {color};
                font-size: 72px;
                font-weight: bold;
                background-color: {CARD_BG};
                border-radius: 16px;
                padding: 40px;
            """)
            
            conf_pct = int(conf * 100)
            self.conf_bar.setValue(conf_pct)
            self.conf_value.setText(f"{conf_pct}%")
            
            # Color bar based on confidence
            if conf_pct >= 80:
                bar_color = GREEN
            elif conf_pct >= 60:
                bar_color = YELLOW
            else:
                bar_color = RED
            
            self.conf_bar.setStyleSheet(f"""
                QProgressBar {{
                    background-color: {CARD_BG};
                    border-radius: 4px;
                }}
                QProgressBar::chunk {{
                    background-color: {bar_color};
                    border-radius: 4px;
                }}
            """)
    
    def closeEvent(self, event):
        for reader in self.readers:
            reader.stop()
            reader.wait()
        event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-p', '--ports', nargs='+', required=True)
    parser.add_argument('--labels', nargs='+')
    args = parser.parse_args()
    
    global LABELS
    if args.labels:
        LABELS = {i: label for i, label in enumerate(args.labels)}
    
    app = QApplication(sys.argv)
    window = LiveWindow(args.model, args.ports)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
