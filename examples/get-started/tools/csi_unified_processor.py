import sys
import csv
import json
import argparse
import numpy as np
import psutil
import gc
import pickle

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
DATA_COLUMNS_NAMES_C5C6 = ["type", "id", "mac", "rssi", "rate","noise_floor","fft_gain","agc_gain", "channel", "local_timestamp",  "sig_len", "rx_state", "len", "first_word", "data"]
DATA_COLUMNS_NAMES = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding",
                      "sgi", "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word", "data"]


# Global arrays for combined data (matching original structure)
csi_data_complex = np.zeros([CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.complex64)
agc_gain_data = np.zeros([CSI_DATA_INDEX], dtype=np.float64)
fft_gain_data = np.zeros([CSI_DATA_INDEX], dtype=np.float64)


# Synchronized data storage - uses arrival order instead of timestamps
# (ESP32 local_timestamps are NOT synchronized across devices)
class SynchronizedData:
    def __init__(self, max_size=CSI_DATA_INDEX, num_receivers=3):
        self.max_size = max_size
        self.num_receivers = num_receivers
        self.lock = Lock()
        
        # Pending packets from each receiver (arrival-order based matching)
        self.pending_packets = {}  # receiver_id -> deque of packets
        self.packet_counter = 0    # Global arrival counter
        
        # Matched/synchronized data buffer
        self.matched_buffer = deque(maxlen=max_size)
        
        # Individual receiver data (for visualization even without sync)
        self.latest_individual = {}  # receiver_id -> latest packet
        
        # CSV output
        self.combined_csv_writer = None
        self.combined_csv_file = None
        
        # Track what we've processed for visualization
        self.last_processed_id = -1
        self._debug_count = 0
        
        # Store individual receiver CSI arrays (for separate analysis)
        self.receiver_csi_arrays = {}  # receiver_id -> np.array
       
    def set_csv_writer(self, csv_writer, csv_file):
        self.combined_csv_writer = csv_writer
        self.combined_csv_file = csv_file
    
    def set_num_receivers(self, num):
        """Set expected number of receivers for synchronization"""
        self.num_receivers = num
       
    def add_data(self, receiver_id, timestamp, csi_complex, agc_gain, fft_gain, rssi):
        global csi_data_complex, agc_gain_data, fft_gain_data
        
        self._debug_count += 1
        
        with self.lock:
            # Initialize pending queue for this receiver if needed
            if receiver_id not in self.pending_packets:
                self.pending_packets[receiver_id] = deque(maxlen=20)  # Buffer last 20 packets
                self.receiver_csi_arrays[receiver_id] = np.zeros(
                    [CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.complex64)
            
            # Create packet with arrival order
            packet = {
                'receiver_id': receiver_id,
                'timestamp': timestamp,
                'csi_complex': csi_complex.copy(),
                'agc_gain': agc_gain,
                'fft_gain': fft_gain,
                'rssi': rssi,
                'arrival_id': self.packet_counter
            }
            self.packet_counter += 1
            
            # Store as latest individual data
            self.latest_individual[receiver_id] = packet
            
            # Update individual receiver's CSI array (for per-receiver analysis)
            rx_array = self.receiver_csi_arrays[receiver_id]
            rx_array[:-1] = rx_array[1:]
            padded = np.zeros(CSI_DATA_COLUMNS, dtype=np.complex64)
            padded[:len(csi_complex)] = csi_complex
            rx_array[-1] = padded
            
            # Add to pending queue
            self.pending_packets[receiver_id].append(packet)
            
            # Try to match packets across receivers
            matched = self._try_match_packets()
            
            if matched:
                # We have synchronized data from all receivers
                self._process_matched_packets(matched)
            else:
                # No sync yet - update global arrays with latest individual data
                # This ensures visualization works even before full sync
                self._update_with_individual(packet)
            
            if self._debug_count % 50 == 0:
                active_receivers = list(self.pending_packets.keys())
                pending_counts = {rid: len(self.pending_packets[rid]) for rid in active_receivers}
                print(f"SyncData: Active receivers: {active_receivers}, pending: {pending_counts}")
    
    def _try_match_packets(self):
        """
        Try to match packets from all receivers by arrival order.
        Returns dict of matched packets or None if not enough receivers have data.
        """
        active_receivers = list(self.pending_packets.keys())
        
        # Need at least 2 receivers, or wait for expected number
        min_receivers = min(2, self.num_receivers)
        if len(active_receivers) < min_receivers:
            return None
        
        # Check if all active receivers have pending packets
        if not all(len(self.pending_packets[rid]) > 0 for rid in active_receivers):
            return None
        
        # Match by taking oldest packet from each receiver
        # (assumes packets from same sender transmission arrive in similar order)
        matched = {}
        for rid in active_receivers:
            if self.pending_packets[rid]:
                matched[rid] = self.pending_packets[rid].popleft()
        
        return matched if len(matched) >= min_receivers else None
    
    def _update_with_individual(self, packet):
        """Update global arrays with individual packet (before sync established)"""
        global csi_data_complex, agc_gain_data, fft_gain_data
        
        # Rotate arrays
        csi_data_complex[:-1] = csi_data_complex[1:]
        agc_gain_data[:-1] = agc_gain_data[1:]
        fft_gain_data[:-1] = fft_gain_data[1:]
        
        # Pad CSI to match array size
        padded = np.zeros(CSI_DATA_COLUMNS, dtype=np.complex64)
        csi = packet['csi_complex']
        padded[:len(csi)] = csi
        
        csi_data_complex[-1] = padded
        agc_gain_data[-1] = packet['agc_gain']
        fft_gain_data[-1] = packet['fft_gain']
    
    def _process_matched_packets(self, matched):
        """Process a set of matched packets from multiple receivers"""
        global csi_data_complex, agc_gain_data, fft_gain_data
        
        receiver_ids = list(matched.keys())
        
        if self._debug_count % 50 == 0:
            print(f"SyncData: Matched packets from {receiver_ids}")
        
        # Store matched entry
        entry_id = self.packet_counter
        matched_entry = {
            'id': entry_id,
            'receivers': matched,
            'timestamp': matched[receiver_ids[0]]['timestamp']
        }
        self.matched_buffer.append(matched_entry)
        
        # Calculate combined CSI (complex averaging)
        min_len = min(len(matched[rid]['csi_complex']) for rid in receiver_ids)
        combined_csi = np.zeros(min_len, dtype=np.complex64)
        
        for i in range(min_len):
            complex_values = [matched[rid]['csi_complex'][i] for rid in receiver_ids]
            combined_csi[i] = np.mean(complex_values)
        
        # Update global arrays
        csi_data_complex[:-1] = csi_data_complex[1:]
        agc_gain_data[:-1] = agc_gain_data[1:]
        fft_gain_data[:-1] = fft_gain_data[1:]
        
        # Pad and store
        padded = np.zeros(CSI_DATA_COLUMNS, dtype=np.complex64)
        padded[:len(combined_csi)] = combined_csi
        csi_data_complex[-1] = padded
        
        # Average gains
        avg_agc = np.mean([matched[rid]['agc_gain'] for rid in receiver_ids])
        avg_fft = np.mean([matched[rid]['fft_gain'] for rid in receiver_ids])
        agc_gain_data[-1] = avg_agc
        fft_gain_data[-1] = avg_fft
        
        # Save to CSV
        self._save_to_csv(matched_entry, combined_csi, receiver_ids)
        
        self.last_processed_id = entry_id
    
    def _save_to_csv(self, matched_entry, combined_csi, receiver_ids):
        """Save combined and individual CSI data to CSV file"""
        if self.combined_csv_writer is None:
            return
        
        matched = matched_entry['receivers']
        
        # Convert combined CSI to I/Q format for CSV
        combined_iq = []
        for z in combined_csi:
            combined_iq.append(float(z.imag))  # Imaginary first (original format)
            combined_iq.append(float(z.real))  # Real second
        
        # Average metadata
        avg_rssi = int(np.mean([matched[rid]['rssi'] for rid in receiver_ids]))
        avg_fft_gain = int(np.mean([matched[rid]['fft_gain'] for rid in receiver_ids]))
        avg_agc_gain = int(np.mean([matched[rid]['agc_gain'] for rid in receiver_ids]))
        
        # Combined row
        combined_row = [
            "CSI_DATA_COMBINED",
            matched_entry['id'],
            "+".join(receiver_ids),
            avg_rssi,
            11,  # rate placeholder
            -95,  # noise_floor placeholder
            avg_fft_gain,
            avg_agc_gain,
            11,  # channel
            matched_entry['timestamp'],
            0,  # sig_len
            0,  # rx_state
            len(combined_iq),
            0,  # first_word
            json.dumps(combined_iq)
        ]
        self.combined_csv_writer.writerow(combined_row)
        
        # Also save individual receiver data (for separate analysis)
        for rid in receiver_ids:
            pkt = matched[rid]
            individual_iq = []
            for z in pkt['csi_complex']:
                individual_iq.append(float(z.imag))
                individual_iq.append(float(z.real))
            
            individual_row = [
                f"CSI_DATA_{rid}",
                matched_entry['id'],
                rid,
                pkt['rssi'],
                11,
                -95,
                pkt['fft_gain'],
                pkt['agc_gain'],
                11,
                pkt['timestamp'],
                0,
                0,
                len(individual_iq),
                0,
                json.dumps(individual_iq)
            ]
            self.combined_csv_writer.writerow(individual_row)
        
        if self.combined_csv_file:
            self.combined_csv_file.flush()
   
    def get_combined_csi(self):
        """Get latest combined CSI data (called by visualization)"""
        with self.lock:
            if len(self.matched_buffer) == 0:
                return None
            
            latest = self.matched_buffer[-1]
            if latest['id'] == self.last_processed_id:
                return None  # Already processed
            
            return {
                'csi_complex': csi_data_complex[-1].copy(),
                'receivers': latest['receivers'],
                'timestamp': latest['timestamp'],
                'id': latest['id']
            }
    
    def get_receiver_csi(self, receiver_id):
        """Get CSI array for a specific receiver (for individual analysis)"""
        with self.lock:
            if receiver_id in self.receiver_csi_arrays:
                return self.receiver_csi_arrays[receiver_id].copy()
            return None
    
    def get_all_receiver_csi(self):
        """Get CSI arrays for all receivers"""
        with self.lock:
            return {rid: arr.copy() for rid, arr in self.receiver_csi_arrays.items()}


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
       
        # Debug: Print gain values to see if they're changing
        if data_count <= 10:  # Print first 10 packets
            print(f"{receiver_id}: FFT_gain={fft_gain}, AGC_gain={agc_gain}")


        csv_writer.writerow(csi_data)


        # Parse CSI complex data
        csi_complex = np.zeros(csi_data_len // 2, dtype=np.complex64)
        for i in range(csi_data_len // 2):
            csi_complex[i] = complex(csi_raw_data[i * 2 + 1], csi_raw_data[i * 2])
       
        # Add to synchronized data structure
        print(f"{receiver_id}: Adding data to sync buffer - timestamp={timestamp}, rssi={rssi}, fft_gain={fft_gain}, agc_gain={agc_gain}")
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
    def __init__(self, receiver_ids, plot_phase=False):
        super().__init__()
        self.receiver_ids = receiver_ids
        self.plot_phase = plot_phase
        self.num_receivers = len(receiver_ids)
        
        # Layout: Combined plot on top, individual receiver plots below
        plot_height = 200
        total_height = plot_height * (1 + self.num_receivers)  # Combined + individual
        self.resize(1280, total_height)
        self.setWindowTitle("Multi-Receiver CSI Data (Combined + Individual)")

        # Initialize arrays
        self.csi_amplitude_array = np.zeros([CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.float64)
        
        # Colors for each receiver
        self.rx_colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]  # Red, Green, Blue
        
        # === COMBINED PLOT (top) ===
        self.plotWidget_combined = PlotWidget(self)
        self.plotWidget_combined.setGeometry(QtCore.QRect(0, 0, 1280, plot_height))
        self.plotWidget_combined.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)
        self.plotWidget_combined.addLegend()
        self.plotWidget_combined.setTitle("Combined (All Receivers Averaged)")
        self.plotWidget_combined.setLabel('left', 'Amplitude')

        self.curve_list = []
        # AGC and FFT gain curves
        agc_curve = self.plotWidget_combined.plot(agc_gain_data, name="AGC", pen=pg.mkPen(color='y', width=2))
        fft_curve = self.plotWidget_combined.plot(fft_gain_data, name="FFT", pen=pg.mkPen(color='orange', width=2))
        self.curve_list.append(agc_curve)
        self.curve_list.append(fft_curve)

        # Subcarrier curves (every 16th)
        for i in range(0, min(CSI_DATA_COLUMNS, 200), 16):
            curve = self.plotWidget_combined.plot(self.csi_amplitude_array[:, i], pen=(255, 255, 255))
            self.curve_list.append(curve)

        # === INDIVIDUAL RECEIVER PLOTS ===
        self.rx_plots = {}
        self.rx_curves = {}
        
        for idx, rid in enumerate(receiver_ids):
            y_pos = plot_height * (idx + 1)
            color = self.rx_colors[idx % len(self.rx_colors)]
            
            plot_widget = PlotWidget(self)
            plot_widget.setGeometry(QtCore.QRect(0, y_pos, 1280, plot_height))
            plot_widget.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)
            plot_widget.setTitle(f"{rid} (Individual)")
            plot_widget.setLabel('left', 'Amplitude')
            
            # Create curves for this receiver
            curves = []
            for i in range(0, min(CSI_DATA_COLUMNS, 200), 16):
                curve = plot_widget.plot(self.csi_amplitude_array[:, i], pen=pg.mkPen(color=color, width=1))
                curves.append(curve)
            
            self.rx_plots[rid] = plot_widget
            self.rx_curves[rid] = curves

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(500)  # update every 500ms
        self.deta_len = 0
        self.update_count = 0


    def update_curve_colors(self, receiver_id, color_list):
        self.deta_len = len(color_list)


    def update_data(self):
        global csi_data_complex, agc_gain_data, fft_gain_data, synced_data
       
        self.update_count += 1
       
        # Memory monitoring
        if self.update_count % 20 == 0:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 80:
                print("CRITICAL: Memory usage too high")
                self.timer.stop()
                return
       
        if self.deta_len == 0:
            return
       
        subcarrier_count = self.deta_len // 2
        curve_indices = list(range(0, min(CSI_DATA_COLUMNS, 200), 16))
        
        # === UPDATE COMBINED PLOT ===
        with synced_data.lock:
            csi_snapshot = csi_data_complex.copy()
            agc_snapshot = agc_gain_data.copy()
            fft_snapshot = fft_gain_data.copy()
       
        np.abs(csi_snapshot, out=self.csi_amplitude_array)
        
        # AGC and FFT curves
        if len(self.curve_list) > 1:
            self.curve_list[0].setData(agc_snapshot)
            self.curve_list[1].setData(fft_snapshot)
       
        # Combined subcarrier curves
        for idx, i in enumerate(curve_indices):
            if idx + 2 < len(self.curve_list) and i < subcarrier_count:
                self.curve_list[idx + 2].setData(self.csi_amplitude_array[:, i])
        
        # === UPDATE INDIVIDUAL RECEIVER PLOTS ===
        rx_arrays = synced_data.get_all_receiver_csi()
        
        for rid, curves in self.rx_curves.items():
            if rid in rx_arrays:
                rx_amp = np.abs(rx_arrays[rid])
                for idx, i in enumerate(curve_indices):
                    if idx < len(curves) and i < subcarrier_count:
                        curves[idx].setData(rx_amp[:, i])  


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
    parser.add_argument('--plot-phase', dest='plot_phase', action='store_true',
                        help="Display an additional plot for subcarrier phase data")


    args = parser.parse_args()
    ports = args.ports
    store_prefix = args.store_prefix
    log_prefix = args.log_prefix
    combined_file = args.combined_file
   
    receiver_ids = [f"RX{i+1}" for i in range(len(ports))]


    # Setup combined CSV file
    combined_csv_fd = open(combined_file, 'w', newline='')
    combined_csv_writer = csv.writer(combined_csv_fd)
    # Header matches DATA_COLUMNS_NAMES_C5C6 format
    # type will be "CSI_DATA_COMBINED" for averaged data, "CSI_DATA_RX1" etc. for individual
    combined_csv_writer.writerow(["type", "id", "mac", "rssi", "rate", "noise_floor", "fft_gain", "agc_gain",
                                  "channel", "local_timestamp", "sig_len", "rx_state", "len", "first_word", "data"])
    synced_data.set_csv_writer(combined_csv_writer, combined_csv_fd)
    synced_data.set_num_receivers(len(ports))  # Set expected number of receivers
   
    print(f"Combined data will be saved to: {combined_file}")
    print(f"Expected {len(ports)} receivers: {receiver_ids}")


    app = QApplication(sys.argv)


    threads = []
    window = unified_csi_window(receiver_ids, plot_phase=args.plot_phase)
   
    for rid, port in zip(receiver_ids, ports):
        subthread = SubThread(port, rid, f"{store_prefix}_{rid}.csv", f"{log_prefix}_{rid}.txt")
        subthread.data_ready.connect(window.update_curve_colors)
        subthread.start()
        threads.append(subthread)
   
    window.show()


    sys.exit(app.exec())