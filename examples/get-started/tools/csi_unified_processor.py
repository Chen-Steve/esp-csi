import sys
import csv
import json
import argparse
import numpy as np
import psutil
import gc

import serial
from io import StringIO
from threading import Lock
from collections import deque

from PyQt5.QtWidgets import QApplication, QWidget
from pyqtgraph import PlotWidget
from PyQt5 import QtCore
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal, QThread

CSI_DATA_INDEX = 50  # Reduced from 200 to prevent memory issues
CSI_DATA_COLUMNS = 490
DATA_COLUMNS_NAMES_C5C6 = ["type", "seq", "mac", "rssi", "rate","noise_floor","fft_gain","agc_gain", "channel", "local_timestamp",  "sig_len", "rx_state", "len", "first_word", "data"]
DATA_COLUMNS_NAMES = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding",
                      "sgi", "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word", "data"]

# Global arrays for combined data (matching original structure)
csi_data_complex = np.zeros([CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.complex64)
agc_gain_data = np.zeros([CSI_DATA_INDEX], dtype=np.float64)
fft_gain_data = np.zeros([CSI_DATA_INDEX], dtype=np.float64)

# Synchronized data storage
class SynchronizedData:
    def __init__(self, max_size=CSI_DATA_INDEX):
        self.max_size = max_size
        self.lock = Lock()
        self.data_buffer = deque(maxlen=max_size)
        self.receiver_data = {}  # Latest data from each receiver
        self.last_timestamp = {}
        self.combined_csv_writer = None
        self.combined_csv_file = None
        self.last_processed_timestamp = 0  # Track last processed timestamp
        
    def set_csv_writer(self, csv_writer, csv_file):
        self.combined_csv_writer = csv_writer
        self.combined_csv_file = csv_file
        
    def add_data(self, receiver_id, timestamp, csi_complex, agc_gain, fft_gain, rssi):
        global csi_data_complex, agc_gain_data, fft_gain_data
        
        # Reduced debug frequency
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1
            
        if self._debug_count % 10 == 0:  # Only print every 10th packet
            print(f"SyncData: Received data from {receiver_id}, timestamp={timestamp}")
        
        with self.lock:
            self.receiver_data[receiver_id] = {
                'timestamp': timestamp,
                'csi_complex': csi_complex,
                'agc_gain': agc_gain,
                'fft_gain': fft_gain,
                'rssi': rssi
            }
            self.last_timestamp[receiver_id] = timestamp
            
            # Try to update with individual receiver data if no sync yet
            if len(self.receiver_data) == 1:
                # Only one receiver so far, use its data directly
                # Only rotate if this is new data
                if timestamp != self.last_processed_timestamp:
                    self.last_processed_timestamp = timestamp
                    rx_data = self.receiver_data[receiver_id]
                    
                    # Rotate arrays
                    csi_data_complex[:-1] = csi_data_complex[1:]
                    agc_gain_data[:-1] = agc_gain_data[1:]
                    fft_gain_data[:-1] = fft_gain_data[1:]
                    
                    # Pad CSI to match array size
                    combined_padded = np.zeros(CSI_DATA_COLUMNS, dtype=np.complex64)
                    rx_csi = rx_data['csi_complex']
                    combined_padded[:len(rx_csi)] = rx_csi
                    
                    csi_data_complex[-1] = combined_padded
                    agc_gain_data[-1] = rx_data['agc_gain']
                    fft_gain_data[-1] = rx_data['fft_gain']
            
            # Find synchronized pairs (within time window)
            if self._debug_count % 10 == 0:  # Reduce debug noise
                print(f"SyncData: Current receivers: {list(self.receiver_data.keys())}")
            self._find_synchronized_pairs()
    
    def _find_synchronized_pairs(self):
        """Find data pairs from different receivers that are close in time"""
        if len(self.receiver_data) < 2:
            return
        
        receiver_ids = list(self.receiver_data.keys())
        if len(receiver_ids) < 2:
            return
        
        # Use most recent timestamps
        timestamps = {rid: self.receiver_data[rid]['timestamp'] for rid in receiver_ids}
        
        # Check if timestamps are close (within larger window for initial sync)
        time_window = 50000  # microseconds (50ms) - increased for better sync
        synchronized = True
        base_time = timestamps[receiver_ids[0]]
        
        for rid in receiver_ids[1:]:
            time_diff = abs(timestamps[rid] - base_time)
            if time_diff > time_window:
                synchronized = False
                break
        
        if synchronized:
            print(f"SyncData: Found synchronized pair! Base time: {base_time}")
            combined_entry = {
                'timestamp': base_time,
                'receivers': self.receiver_data.copy()
            }
            self.data_buffer.append(combined_entry)
            
            # Save to CSV
            self._save_combined_to_csv(combined_entry)
        else:
            print(f"SyncData: No synchronization - time differences too large")
    
    def _save_combined_to_csv(self, combined_entry):
        """Save combined CSI data to CSV file"""
        if self.combined_csv_writer is None:
            return
        
        receivers = combined_entry['receivers']
        if len(receivers) < 2:
            return
        
        receiver_ids = list(receivers.keys())
        
        # Get minimum CSI length across ALL receivers
        min_len = min(len(receivers[rid]['csi_complex']) for rid in receiver_ids)
        combined_csi = []
        
        for i in range(min_len):
            # Average in complex plane across ALL receivers (better than averaging amplitude and phase separately)
            complex_values = [receivers[rid]['csi_complex'][i] for rid in receiver_ids]
            z_avg = np.mean(complex_values)  # Average all receivers
            
            # Convert back to I/Q format (convert to Python float for JSON)
            # Format: [Imaginary, Real] pairs as per original convention
            combined_csi.append(float(z_avg.imag))  # Imaginary
            combined_csi.append(float(z_avg.real))  # Real
        
        # Average metadata across ALL receivers
        avg_rssi = int(np.mean([receivers[rid]['rssi'] for rid in receiver_ids]))
        avg_fft_gain = int(np.mean([receivers[rid]['fft_gain'] for rid in receiver_ids]))
        avg_agc_gain = int(np.mean([receivers[rid]['agc_gain'] for rid in receiver_ids]))
        
        # Create combined MAC field showing all receivers
        combined_mac = "+".join(receiver_ids)
        
        # Create CSV row (use format from first receiver, mark as combined)
        # We'll use a simplified format: just the essential fields + combined data
        combined_row = [
            "CSI_DATA_COMBINED",
            0,  # id
            combined_mac,  # combined MAC showing all receivers
            avg_rssi,  # avg RSSI across all receivers
            11,  # rate (placeholder)
            -95,  # noise_floor (placeholder)
            avg_fft_gain,  # avg fft_gain across all receivers
            avg_agc_gain,  # avg agc_gain across all receivers
            11,  # channel
            combined_entry['timestamp'],  # local_timestamp
            0,  # sig_len
            0,  # rx_state
            len(combined_csi),  # len
            0,  # first_word
            json.dumps(combined_csi)  # data
        ]
        
        self.combined_csv_writer.writerow(combined_row)
        if self.combined_csv_file:
            self.combined_csv_file.flush()
    
    def get_combined_csi(self):
        """Get combined CSI data from all receivers and update global arrays"""
        global csi_data_complex, agc_gain_data, fft_gain_data
        
        with self.lock:
            if len(self.data_buffer) == 0:
                return None
            
            latest = self.data_buffer[-1]
            if 'receivers' not in latest:
                return None
            
            receivers = latest['receivers']
            if len(receivers) < 2:
                return None
            
            # Only process if this is new data (different timestamp)
            current_timestamp = latest['timestamp']
            if current_timestamp == self.last_processed_timestamp:
                return None  # Already processed this data
            
            self.last_processed_timestamp = current_timestamp
            
            # Combine CSI data (average amplitude, combine phase info)
            receiver_ids = list(receivers.keys())
            csi_data = []
            
            # Get minimum length across receivers
            min_len = min(len(receivers[rid]['csi_complex']) for rid in receiver_ids)
            
            for i in range(min_len):
                # Average in complex plane (better than averaging amplitude and phase separately)
                complex_values = [receivers[rid]['csi_complex'][i] for rid in receiver_ids]
                combined = np.mean(complex_values)  # Direct complex averaging
                csi_data.append(combined)
            
            # Rotate data arrays ONLY when new data arrives
            csi_data_complex[:-1] = csi_data_complex[1:]
            agc_gain_data[:-1] = agc_gain_data[1:]
            fft_gain_data[:-1] = fft_gain_data[1:]
            
            # Average AGC and FFT gains
            avg_agc = int(np.mean([receivers[rid]['agc_gain'] for rid in receiver_ids]))
            avg_fft = int(np.mean([receivers[rid]['fft_gain'] for rid in receiver_ids]))
            
            # Pad combined CSI to match array size
            combined_padded = np.zeros(CSI_DATA_COLUMNS, dtype=np.complex64)
            combined_padded[:len(csi_data)] = np.array(csi_data)
            
            csi_data_complex[-1] = combined_padded
            agc_gain_data[-1] = avg_agc
            fft_gain_data[-1] = avg_fft
            
            return {
                'csi_complex': np.array(csi_data),
                'receivers': receivers,
                'timestamp': current_timestamp
            }

# Global synchronized data
synced_data = SynchronizedData()

def generate_subcarrier_colors(red_range, green_range, yellow_range, total_num, interval=1):
    colors = []
    for i in range(total_num):
        if red_range and red_range[0] <= i <= red_range[1]:
            intensity = int(255 * (i - red_range[0]) / (red_range[1] - red_range[0])) 
            colors.append((intensity, 0, 0)) 
        elif green_range and green_range[0] <= i <= green_range[1]:
            intensity = int(255 * (i - green_range[0]) / (green_range[1] - green_range[0])) 
            colors.append((0, intensity, 0)) 
        elif yellow_range and yellow_range[0] <= i <= yellow_range[1]:
            intensity = int(255 * (i - yellow_range[0]) / (yellow_range[1] - yellow_range[0])) 
            colors.append((0, intensity, intensity)) 
        else:
            colors.append((200, 200, 200))  
    return colors

def csi_data_read_parse(port: str, receiver_id: str, csv_writer, log_file_fd, subthread=None, callback=None):
    global synced_data
    
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
    loop_count = 0
    
    while True:
        try:
            loop_count += 1
            line = ser.readline()
            if not line:
                continue  # Timeout occurred, try again
                
            strings = line.decode('utf-8', errors='ignore').strip()
            if not strings:
                continue
                
            # Debug: Print first few lines to see what we're receiving
            if data_count < 5:
                print(f"{receiver_id}: Received line {data_count}: {strings[:100]}...")
                data_count += 1
            
            index = strings.find('CSI_DATA')

            if index == -1:
                log_file_fd.write(strings + '\n')
                log_file_fd.flush()
                continue
            
            print(f"{receiver_id}: Found CSI_DATA packet!")
            
        except Exception as e:
            print(f"Error reading from {receiver_id}: {e}")
            continue

        try:
            csv_reader = csv.reader(StringIO(strings))
            csi_data = next(csv_reader)
            csi_data_len = int(csi_data[-3])
            print(f"{receiver_id}: Parsed CSI data, length={csi_data_len}, columns={len(csi_data)}")
            
            if len(csi_data) != len(DATA_COLUMNS_NAMES) and len(csi_data) != len(DATA_COLUMNS_NAMES_C5C6):
                print(f"{receiver_id}: Column count mismatch - got {len(csi_data)}, expected {len(DATA_COLUMNS_NAMES)} or {len(DATA_COLUMNS_NAMES_C5C6)}")
                log_file_fd.write("element number is not equal\n")
                log_file_fd.write(strings + '\n')
                log_file_fd.flush()
                continue
        except Exception as e:
            print(f"{receiver_id}: Error parsing CSV data: {e}")
            continue

        try:
            csi_raw_data = json.loads(csi_data[-1])
            print(f"{receiver_id}: Successfully parsed JSON data, raw_data_len={len(csi_raw_data)}")
        except json.JSONDecodeError as e:
            print(f"{receiver_id}: JSON decode error: {e}")
            log_file_fd.write("data is incomplete\n")
            log_file_fd.write(strings + '\n')
            log_file_fd.flush()
            continue
            
        if csi_data_len != len(csi_raw_data):
            print(f"{receiver_id}: CSI length mismatch - header says {csi_data_len}, got {len(csi_raw_data)}")
            log_file_fd.write("csi_data_len is not equal\n")
            log_file_fd.write(strings + '\n')
            log_file_fd.flush()
            continue

        # Write CSV header on first packet (now that we know the format)
        if subthread and not subthread.header_written:
            if len(csi_data) == len(DATA_COLUMNS_NAMES_C5C6):
                csv_writer.writerow(DATA_COLUMNS_NAMES_C5C6)
                print(f"{receiver_id}: Writing ESP32-C5/C6 format header")
            else:
                csv_writer.writerow(DATA_COLUMNS_NAMES)
                print(f"{receiver_id}: Writing standard ESP32 format header")
            subthread.header_written = True

        # Extract metadata
        timestamp_idx = 9 if len(csi_data) == len(DATA_COLUMNS_NAMES_C5C6) else 18
        rssi_idx = 3
        timestamp = int(csi_data[timestamp_idx])
        rssi = int(csi_data[rssi_idx])
        
        fft_gain = int(csi_data[6])
        agc_gain = int(csi_data[7])

        csv_writer.writerow(csi_data)

        # Parse CSI complex data
        csi_complex = np.zeros(csi_data_len // 2, dtype=np.complex64)
        for i in range(csi_data_len // 2):
            csi_complex[i] = complex(csi_raw_data[i * 2 + 1], csi_raw_data[i * 2])
        
        # Add to synchronized data structure
        print(f"{receiver_id}: Adding data to sync buffer - timestamp={timestamp}, rssi={rssi}")
        synced_data.add_data(receiver_id, timestamp, csi_complex, agc_gain, fft_gain, rssi)
        
        if count == 0:
            count = 1
            print(f"{receiver_id}: First CSI packet processed! Data length: {csi_data_len}")
            if csi_data_len == 106:
                colors = generate_subcarrier_colors((0,25), (27,53), None, len(csi_raw_data))
            elif csi_data_len == 114:
                colors = generate_subcarrier_colors((0,27), (29,56), None, len(csi_raw_data))
            elif csi_data_len == 52:
                colors = generate_subcarrier_colors((0,12), (13,26), None, len(csi_raw_data))   
            elif csi_data_len == 234:
                colors = generate_subcarrier_colors((0,28), (29,56), (60,116), len(csi_raw_data))
            elif csi_data_len == 490:
                colors = generate_subcarrier_colors((0,61), (62,122), (123,245), len(csi_raw_data))
            elif csi_data_len == 128:
                colors = generate_subcarrier_colors((0,31), (32,63), None, len(csi_raw_data))
            elif csi_data_len == 256:
                colors = generate_subcarrier_colors((0,32), (32,63), (64,128), len(csi_raw_data))
            elif csi_data_len == 512:
                colors = generate_subcarrier_colors((0,63), (64,127), (128,256), len(csi_raw_data))
            elif csi_data_len == 384:
                colors = generate_subcarrier_colors((0,63), (64,127), (128,192), len(csi_raw_data))
            else:
                print("Please add more color schemes.")
                count = 0
                continue
            
            if callback:
                callback(receiver_id, colors)

    ser.close()
    return

class SubThread (QThread):
    data_ready = pyqtSignal(str, object)
    def __init__(self, serial_port, receiver_id, save_file_name, log_file_name):
        super().__init__()
        self.serial_port = serial_port
        self.receiver_id = receiver_id
        save_file_fd = open(save_file_name, 'w')
        self.log_file_fd = open(log_file_name, 'w')
        self.csv_writer = csv.writer(save_file_fd)
        # Note: Header will be written after detecting data format in first packet
        self.save_file_fd = save_file_fd
        self.header_written = False

    def run(self):
        csi_data_read_parse(self.serial_port, self.receiver_id, self.csv_writer, self.log_file_fd,
                           subthread=self, callback=lambda rid, colors: self.data_ready.emit(rid, colors))

    def __del__(self):
        self.wait()
        if hasattr(self, 'log_file_fd'):
            self.log_file_fd.close()
        if hasattr(self, 'save_file_fd'):
            self.save_file_fd.close()

class unified_csi_window(QWidget):
    def __init__(self, receiver_ids):
        super().__init__()
        self.receiver_ids = receiver_ids
        self.resize(1280, 400)  # Much smaller height with only one plot
        self.setWindowTitle("Unified Multi-Receiver CSI Data")

        # Top plot removed to reduce memory usage
        # Initialize arrays to be reused later (prevent memory allocation in update loop)
        self.csi_amplitude_array = np.zeros_like(csi_data_complex, dtype=np.float64)
        self.csi_phase_array = np.zeros_like(csi_data_complex, dtype=np.float64)
        np.abs(csi_data_complex, out=self.csi_amplitude_array)
        self.csi_phase_array[:] = np.angle(csi_data_complex)  # angle() doesn't support out parameter

        # Subcarrier Amplitude Data (only plot, full window)
        self.plotWidget_multi_data = PlotWidget(self)
        self.plotWidget_multi_data.setGeometry(QtCore.QRect(0, 0, 1280, 400))
        self.plotWidget_multi_data.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)
        self.plotWidget_multi_data.addLegend()
        self.plotWidget_multi_data.setTitle("Combined Subcarrier Amplitude Data")
        self.plotWidget_multi_data.setLabel('left', 'Amplitude') 
        self.plotWidget_multi_data.setLabel('bottom', 'Time (Cumulative Packet Count)') 

        self.curve_list = []
        agc_curve = self.plotWidget_multi_data.plot(
            agc_gain_data, name="AGC Gain", pen=[255,255,0])
        fft_curve = self.plotWidget_multi_data.plot(
            fft_gain_data, name="FFT Gain", pen=[255,255,0])
        self.curve_list.append(agc_curve)
        self.curve_list.append(fft_curve)

        # Limit curves to reduce memory usage (only show every 16th subcarrier)
        for i in range(0, min(CSI_DATA_COLUMNS, 200), 16):
            curve = self.plotWidget_multi_data.plot(
                self.csi_amplitude_array[:, i], name=str(i), pen=(255, 255, 255))
            self.curve_list.append(curve)

        # Bottom plot removed to reduce memory usage
        self.curve_phase_list = []
        
        # IQ Plot removed to reduce memory usage
        self.iq_colors = []

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(1000)  # every one second
        self.deta_len = 0
        self.update_count = 0

    def update_curve_colors(self, receiver_id, color_list):
        self.deta_len = len(color_list)
        self.iq_colors = color_list
        # Only update amplitude curves since other plots were removed
        curve_indices = list(range(0, min(CSI_DATA_COLUMNS, 200), 16))
        for idx, i in enumerate(curve_indices):
            if idx + 2 < len(self.curve_list) and i < len(color_list):  # +2 for AGC/FFT curves
                self.curve_list[idx + 2].setPen(color_list[i])

    def update_data(self):
        global csi_data_complex, agc_gain_data, fft_gain_data, synced_data
        
        self.update_count += 1
        
        # Memory monitoring every 10 updates (~10 seconds with 1s timer)
        if self.update_count % 10 == 0:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 70:
                print(f"WARNING: High memory usage: {memory_percent:.1f}%")
                gc.collect()  # Force garbage collection
            
            if memory_percent > 80:
                print("CRITICAL: Memory usage too high, stopping to prevent crash")
                self.timer.stop()
                return
        
        # Try to get combined CSI data (this will rotate arrays if new data available)
        combined = synced_data.get_combined_csi()
        # Note: Rotation happens inside get_combined_csi() when new data arrives
        # and inside add_data() for single receiver
        
        # Update plots (matching original update_data structure)
        if self.deta_len == 0:
            return  # Wait for color scheme to be set
        
        # Reuse existing arrays instead of creating new ones to prevent memory leaks
        if not hasattr(self, 'csi_amplitude_array') or self.csi_amplitude_array.shape != csi_data_complex.shape:
            self.csi_amplitude_array = np.zeros_like(csi_data_complex, dtype=np.float64)
            self.csi_phase_array = np.zeros_like(csi_data_complex, dtype=np.float64)
        
        np.abs(csi_data_complex, out=self.csi_amplitude_array)
        self.csi_phase_array[:] = np.angle(csi_data_complex)  # angle() doesn't support out parameter
        
        # Only show data up to deta_len//2 subcarriers (matching original)
        subcarrier_count = self.deta_len // 2 

        # AGC and FFT curves are at indices 0 and 1
        if len(self.curve_list) > 1:
            self.curve_list[0].setData(agc_gain_data)
            self.curve_list[1].setData(fft_gain_data)
        
        # Only update amplitude curves for valid subcarriers (matching reduced curve count)
        curve_indices = list(range(0, min(CSI_DATA_COLUMNS, 200), 16))
        for idx, i in enumerate(curve_indices):
            if idx + 2 < len(self.curve_list) and i < subcarrier_count:  # +2 for AGC/FFT curves
                self.curve_list[idx + 2].setData(self.csi_amplitude_array[:, i])   

if __name__ == '__main__':
    if sys.version_info < (3, 6):
        print(" Python version should >= 3.6")
        exit()

    parser = argparse.ArgumentParser(
        description="Unified multi-receiver CSI data processor")
    parser.add_argument('-p', '--ports', dest='ports', nargs='+', required=True,
                        help="Serial port numbers of receivers (e.g., COM3 COM6)")
    parser.add_argument('-s', '--store', dest='store_prefix', action='store', default='./csi_data',
                        help="Prefix for saving CSV files")
    parser.add_argument('-l', '--log', dest="log_prefix", action="store", default="./csi_data_log",
                        help="Prefix for log files")
    parser.add_argument('-c', '--combined', dest='combined_file', action='store', default='./csi_data_combined.csv',
                        help="Output file for combined unified CSI data")

    args = parser.parse_args()
    ports = args.ports
    store_prefix = args.store_prefix
    log_prefix = args.log_prefix
    combined_file = args.combined_file
    
    receiver_ids = [f"RX{i+1}" for i in range(len(ports))]

    # Setup combined CSV file
    combined_csv_fd = open(combined_file, 'w')
    combined_csv_writer = csv.writer(combined_csv_fd)
    combined_csv_writer.writerow(["type", "id", "mac", "rssi", "rate", "noise_floor", "fft_gain", "agc_gain", 
                                  "channel", "local_timestamp", "sig_len", "rx_state", "len", "first_word", "data"])
    synced_data.set_csv_writer(combined_csv_writer, combined_csv_fd)
    
    print(f"Combined data will be saved to: {combined_file}")

    app = QApplication(sys.argv)

    threads = []
    window = unified_csi_window(receiver_ids)
    
    for rid, port in zip(receiver_ids, ports):
        subthread = SubThread(port, rid, f"{store_prefix}_{rid}.csv", f"{log_prefix}_{rid}.txt")
        subthread.data_ready.connect(window.update_curve_colors)
        subthread.start()
        threads.append(subthread)
    
    window.show()

    sys.exit(app.exec())
