import sys
import csv
import json
import argparse
import numpy as np
import psutil

import serial
from io import StringIO
from threading import Lock


from PyQt5.QtWidgets import QApplication, QWidget
from pyqtgraph import PlotWidget
from PyQt5 import QtCore
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal, QThread


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
    global data_store
   
    try:
        ser = serial.Serial(port=port, baudrate=921600, bytesize=8, parity='N', stopbits=1, timeout=1.0)
        count = 0
       
        if ser.isOpen():
            print(f"Serial port {port} opened successfully for {receiver_id}")
        else:
            print(f"Failed to open serial port {port} for {receiver_id}")
            return
    except Exception as e:
        print(f"Error opening serial port {port} for {receiver_id}: {e}")
        return
   
    print(f"Starting data reading loop for {receiver_id} on {port}")
    data_count = 0
   
    while True:
        try:
            line = ser.readline()
            if not line:
                continue
               
            strings = line.decode('utf-8', errors='ignore').strip()
            if not strings:
                continue
               
            # Debug: Print first few lines
            if data_count < 3:
                print(f"{receiver_id}: Received line {data_count}: {strings[:80]}...")
                data_count += 1
           
            if strings.find('CSI_DATA') == -1:
                continue
           
        except Exception as e:
            print(f"Error reading from {receiver_id}: {e}")
            continue

        try:
            csv_reader = csv.reader(StringIO(strings))
            csi_data = next(csv_reader)
            csi_data_len = int(csi_data[-3])
           
            if len(csi_data) != len(DATA_COLUMNS_NAMES) and len(csi_data) != len(DATA_COLUMNS_NAMES_C5C6):
                continue
        except Exception as e:
            print(f"{receiver_id}: Error parsing CSV: {e}")
            continue

        try:
            csi_raw_data = json.loads(csi_data[-1])
        except json.JSONDecodeError:
            continue
           
        if csi_data_len != len(csi_raw_data):
            continue

        # Write CSV header on first packet
        if subthread and not subthread.header_written:
            if len(csi_data) == len(DATA_COLUMNS_NAMES_C5C6):
                csv_writer.writerow(DATA_COLUMNS_NAMES_C5C6)
            else:
                csv_writer.writerow(DATA_COLUMNS_NAMES)
            subthread.header_written = True

        # Save to CSV
        csv_writer.writerow(csi_data)
        subthread.save_file_fd.flush()

        # Parse CSI complex data for visualization
        csi_complex = np.zeros(csi_data_len // 2, dtype=np.complex64)
        for i in range(csi_data_len // 2):
            csi_complex[i] = complex(csi_raw_data[i * 2 + 1], csi_raw_data[i * 2])
       
        # Add to data store for visualization
        data_store.add_data(receiver_id, csi_complex)
       
        if count == 0:
            count = 1
            print(f"{receiver_id}: First CSI packet! Data length: {csi_data_len}")
            if callback:
                callback(receiver_id, csi_data_len)

    ser.close()
    return


class SubThread(QThread):
    data_ready = pyqtSignal(str, int)
    
    def __init__(self, serial_port, receiver_id, save_file_name):
        super().__init__()
        self.serial_port = serial_port
        self.receiver_id = receiver_id
        self.save_file_fd = open(save_file_name, 'w', newline='')
        self.csv_writer = csv.writer(self.save_file_fd)
        self.header_written = False

    def run(self):
        csi_data_read_parse(self.serial_port, self.receiver_id, self.csv_writer,
                           subthread=self, callback=lambda rid, dlen: self.data_ready.emit(rid, dlen))

    def __del__(self):
        self.wait()
        if hasattr(self, 'save_file_fd'):
            self.save_file_fd.close()


class CSIWindow(QWidget):
    def __init__(self, receiver_ids):
        super().__init__()
        self.receiver_ids = receiver_ids
        self.num_receivers = len(receiver_ids)
        
        plot_height = 200
        total_height = plot_height * self.num_receivers
        self.resize(1280, total_height)
        self.setWindowTitle("Multi-Receiver CSI Data Collection")

        self.csi_amplitude_array = np.zeros([CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.float64)
        self.rx_colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]  # Red, Green, Blue
        self.data_len = 0
        
        # Create plots for each receiver
        self.rx_plots = {}
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
                curve = plot_widget.plot(self.csi_amplitude_array[:, i], pen=pg.mkPen(color=color, width=1))
                curves.append(curve)
            
            self.rx_plots[rid] = plot_widget
            self.rx_curves[rid] = curves

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(500)
        self.update_count = 0

    def on_data_ready(self, receiver_id, data_len):
        self.data_len = data_len

    def update_data(self):
        global data_store
       
        self.update_count += 1
       
        # Memory monitoring
        if self.update_count % 20 == 0:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 80:
                print("CRITICAL: Memory usage too high")
                self.timer.stop()
                return
       
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
    if sys.version_info < (3, 6):
        print("Python version should >= 3.6")
        exit()

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

    threads = []
    window = CSIWindow(receiver_ids)
   
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

    sys.exit(app.exec())
