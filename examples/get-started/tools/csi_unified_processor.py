import sys
import csv
import json
import argparse
import numpy as np
import serial
from io import StringIO
from threading import Lock


from PyQt5.QtWidgets import QApplication, QWidget
from pyqtgraph import PlotWidget
from PyQt5 import QtCore
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal, QThread


BAUDRATE = 921600
SERIAL_TIMEOUT_SEC = 1.0
CSI_MARKER = "CSI_DATA"

CSI_DATA_INDEX = 50
CSI_DATA_COLUMNS = 490
DATA_COLUMNS_NAMES_C5C6 = ["type", "id", "mac", "rssi", "rate","noise_floor","fft_gain","agc_gain", "channel", "local_timestamp",  "sig_len", "rx_state", "len", "first_word", "data"]
DATA_COLUMNS_NAMES = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding",
                      "sgi", "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word", "data"]


# Simple storage for each receiver's CSI data (for visualization only)
class ReceiverDataStore:
    def __init__(self):
        self.lock = Lock()
        self.receiver_arrays = {}  # receiver_id -> np.array
    
    def add_data(self, receiver_id, csi_complex):
        with self.lock:
            if receiver_id not in self.receiver_arrays:
                self.receiver_arrays[receiver_id] = np.zeros(
                    [CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.complex64)
            
            # Shift and add new data
            arr = self.receiver_arrays[receiver_id]
            arr[:-1] = arr[1:]
            padded = np.zeros(CSI_DATA_COLUMNS, dtype=np.complex64)
            padded[:len(csi_complex)] = csi_complex
            arr[-1] = padded
    
    def get_all_receiver_csi(self):
        with self.lock:
            return {rid: arr.copy() for rid, arr in self.receiver_arrays.items()}


# Global data store
data_store = ReceiverDataStore()


def csi_data_read_parse(port: str, receiver_id: str, csv_writer, subthread=None, callback=None):
    try:
        ser = serial.Serial(
            port=port,
            baudrate=BAUDRATE,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=SERIAL_TIMEOUT_SEC,
        )
    except Exception as e:
        print(f"Error opening serial port {port} for {receiver_id}: {e}")
        return

    if not ser.isOpen():
        print(f"Failed to open serial port {port} for {receiver_id}")
        return

    print(f"{receiver_id}: Connected on {port}")
    first_packet = True

    try:
        while True:
            if subthread is not None and subthread.isInterruptionRequested():
                break

            try:
                line = ser.readline()
                if not line:
                    continue

                text = line.decode("utf-8", errors="ignore").strip()
                if not text or CSI_MARKER not in text:
                    continue
            except Exception as e:
                print(f"{receiver_id}: Serial read error: {e}")
                continue

            try:
                csi_row = next(csv.reader(StringIO(text)))
                csi_data_len = int(csi_row[-3])

                if len(csi_row) not in (len(DATA_COLUMNS_NAMES), len(DATA_COLUMNS_NAMES_C5C6)):
                    continue

                csi_raw_data = json.loads(csi_row[-1])
                if csi_data_len != len(csi_raw_data):
                    continue
            except Exception:
                continue

            if subthread is not None and not subthread.header_written:
                csv_writer.writerow(
                    DATA_COLUMNS_NAMES_C5C6 if len(csi_row) == len(DATA_COLUMNS_NAMES_C5C6) else DATA_COLUMNS_NAMES
                )
                subthread.header_written = True

            csv_writer.writerow(csi_row)
            if subthread is not None:
                subthread.save_file_fd.flush()

            # Convert raw I/Q pairs into complex samples for plotting:
            # csi_raw_data is [imag0, real0, imag1, real1, ...]
            n_sc = csi_data_len // 2
            csi_complex = np.empty(n_sc, dtype=np.complex64)
            for i in range(n_sc):
                csi_complex[i] = complex(csi_raw_data[i * 2 + 1], csi_raw_data[i * 2])

            data_store.add_data(receiver_id, csi_complex)

            if first_packet:
                first_packet = False
                print(f"{receiver_id}: First CSI packet (len={csi_data_len})")
                if callback:
                    callback(receiver_id, csi_data_len)
    finally:
        try:
            ser.close()
        except Exception:
            pass


class SubThread(QThread):
    data_ready = pyqtSignal(str, int)
    
    def __init__(self, serial_port, receiver_id, save_file_name):
        super().__init__()
        self.serial_port = serial_port
        self.receiver_id = receiver_id
        self.save_file_name = save_file_name
        self.save_file_fd = None
        self.csv_writer = None
        self.header_written = False

    def run(self):
        with open(self.save_file_name, "w", newline="") as f:
            self.save_file_fd = f
            self.csv_writer = csv.writer(f)
            self.header_written = False
            csi_data_read_parse(
                self.serial_port,
                self.receiver_id,
                self.csv_writer,
                subthread=self,
                callback=lambda rid, dlen: self.data_ready.emit(rid, dlen),
            )
        self.save_file_fd = None
        self.csv_writer = None

    def stop(self):
        self.requestInterruption()


class CSIWindow(QWidget):
    def __init__(self, receiver_ids):
        super().__init__()
        
        plot_height = 200
        total_height = plot_height * len(receiver_ids)
        self.resize(1280, total_height)
        self.setWindowTitle("Multi-Receiver CSI Data Collection")

        self.rx_colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]  # Red, Green, Blue
        self.data_len = 0
        
        # Create plots for each receiver
        self.rx_curves = {}
        
        for idx, rid in enumerate(receiver_ids):
            y_pos = plot_height * idx
            color = self.rx_colors[idx % len(self.rx_colors)]
            
            plot_widget = PlotWidget(self)
            plot_widget.setGeometry(QtCore.QRect(0, y_pos, 1280, plot_height))
            plot_widget.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)
            plot_widget.setTitle(f"{rid}")
            plot_widget.setLabel('left', 'Amplitude')
            
            curves = []
            for i in range(0, min(CSI_DATA_COLUMNS, 200), 16):
                curve = plot_widget.plot([], pen=pg.mkPen(color=color, width=1))
                curves.append(curve)
            
            self.rx_curves[rid] = curves

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(500)

    def on_data_ready(self, receiver_id, data_len):
        self.data_len = data_len

    def update_data(self):
        if self.data_len == 0:
            return
       
        subcarrier_count = self.data_len // 2
        curve_indices = list(range(0, min(CSI_DATA_COLUMNS, 200), 16))
        
        rx_arrays = data_store.get_all_receiver_csi()
        
        for rid, curves in self.rx_curves.items():
            if rid in rx_arrays:
                rx_amp = np.abs(rx_arrays[rid])
                for idx, i in enumerate(curve_indices):
                    if idx < len(curves) and i < subcarrier_count:
                        curves[idx].setData(rx_amp[:, i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Multi-receiver CSI data collector")
    parser.add_argument('-p', '--ports', dest='ports', nargs='+', required=True,
                        help="Serial ports (e.g., COM3 COM6 COM7)")
    parser.add_argument('-s', '--store', dest='store_prefix', action='store', default='./csi_data',
                        help="Prefix for CSV files")
    parser.add_argument('-d', '--duration', dest='duration', type=int, default=0,
                        help="Duration in seconds (0 = run indefinitely)")

    args = parser.parse_args()
    ports = args.ports
    store_prefix = args.store_prefix
    duration = args.duration
   
    receiver_ids = [f"RX{i+1}" for i in range(len(ports))]
   
    print(f"Starting collection from {len(ports)} receivers: {receiver_ids}")
    print(f"Data will be saved to: {store_prefix}_RX1.csv, {store_prefix}_RX2.csv, ...")

    app = QApplication(sys.argv)
    window = CSIWindow(receiver_ids)
   
    threads = []
    for rid, port in zip(receiver_ids, ports):
        subthread = SubThread(port, rid, f"{store_prefix}_{rid}.csv")
        subthread.data_ready.connect(window.on_data_ready)
        subthread.start()
        threads.append(subthread)
   
    window.show()

    if duration > 0:
        print(f"Will automatically stop after {duration} seconds...")
        close_timer = QtCore.QTimer()
        close_timer.setSingleShot(True)
        close_timer.timeout.connect(app.quit)
        close_timer.start(duration * 1000)

    def _shutdown_threads():
        for t in threads:
            t.stop()
        for t in threads:
            t.wait(2000)

    app.aboutToQuit.connect(_shutdown_threads)

    sys.exit(app.exec())
