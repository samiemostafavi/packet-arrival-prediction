import os, sys
from pathlib import Path
from loguru import logger
import pandas as pd
from edaf.core.uplink.analyze_packet import ULPacketAnalyzer
from edaf.core.uplink.analyze_scheduling import ULSchedulingAnalyzer
from edaf.core.uplink.analyze_channel import ULChannelAnalyzer
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter, medfilt, butter, filtfilt
from scipy.stats import norm

if not os.getenv('DEBUG'):
    logger.remove()
    logger.add(sys.stdout, level="INFO")

config = {
    'db_address' : 'data/240928_082545_results/database.db',
    'time_mask' : [0.2, 0.9], # portion of interest from the whole time series
    'filter_packet_sizes' : [ 128 ], # only allows these sizes in bytes (note that this turns into a discrete event problem)
    'history_window_size' : 20, # number of events to consider in the past (both packet arrival and frame start events)
    'dataset_size_max' : 10000, # maximum number of events series to export
    'split_ratios' : [0.7,0.15,0.15], # split dataset [train, dev, test]
    'experiment' : {
        'total_prbs_num' : 106,
        'symbols_per_slot' : 14,
        'slots_per_frame' : 20,
        'slots_duration_ms' : 0.5,
        'scheduling_map_num_integers' : 4,
        'max_num_frames' : 1024,
    }
}

if __name__ == "__main__":

    if len(sys.argv) != 2:
        logger.error("Usage: python channel.py <source_db_address> <result_files>")
        sys.exit(1)

    # read configuration
    results_folder_addr = Path(sys.argv[1])
    result_database_file = config['db_address']
    time_mask = config['time_mask']
    filter_packet_sizes = config['filter_packet_sizes']
    history_window_size = config['history_window_size']
    dataset_size_max = config['dataset_size_max']
    split_ratios = config['split_ratios']
    slots_duration_ms = config['experiment']['slots_duration_ms']
    num_slots_per_frame = config['experiment']['slots_per_frame']
    total_prbs_num = config['experiment']['total_prbs_num']
    symbols_per_slot = config['experiment']['symbols_per_slot']
    scheduling_map_num_integers = config['experiment']['scheduling_map_num_integers']
    max_num_frames = config['experiment']['max_num_frames']

    # initiate analyzers
    pacekt_analyzer = ULPacketAnalyzer(result_database_file)
    scheduling_analyzer = ULSchedulingAnalyzer(
        total_prbs_num = total_prbs_num, 
        symbols_per_slot = symbols_per_slot,
        slots_per_frame = num_slots_per_frame, 
        slots_duration_ms = slots_duration_ms, 
        scheduling_map_num_integers = scheduling_map_num_integers,
        max_num_frames = max_num_frames,
        db_addr = result_database_file
    )

    experiment_length_ts = pacekt_analyzer.last_ueip_ts - pacekt_analyzer.first_ueip_ts
    logger.info(f"Experiment duration: {(experiment_length_ts)} seconds")
    logger.info(f"Filtering packets from {pacekt_analyzer.first_ueip_ts+experiment_length_ts*time_mask[0]} to {pacekt_analyzer.first_ueip_ts+experiment_length_ts*time_mask[1]}, length: {experiment_length_ts*time_mask[1]-experiment_length_ts*time_mask[0]} seconds")
    packets = pacekt_analyzer.figure_packettx_from_ts(pacekt_analyzer.first_ueip_ts+experiment_length_ts*time_mask[0], pacekt_analyzer.first_ueip_ts+experiment_length_ts*time_mask[1])
    logger.info(f"Number of packets for this duration: {len(packets)}")

    # sort the packets in case they are not sorted
    packets = sorted(packets, key=lambda x: x['ip.in_t'], reverse=False)

    packet_size_arr = []
    packet_inp_ts_arr = []
    frame_strt_ts_arr = []
    slot_num_arr = []
    for packet in packets:
        if packet['len'] not in filter_packet_sizes:
            continue
        frame_start_ts, frame_num, slot_num = scheduling_analyzer.find_frame_slot_from_ts(
            timestamp=packet['ip.in_t'], 
            #SCHED_OFFSET_S=0.002 # 2ms which is 4*slot_duration_ms
            SCHED_OFFSET_S=0.004 # 4ms which is 8*slot_duration_ms
        )
        slot_num_arr.append(slot_num)
        frame_strt_ts_arr.append(frame_start_ts)
        packet_inp_ts_arr.append(packet['ip.in_t'])
        packet_size_arr.append(packet['len'])

    # Plot timeseries of slot numbers against packet_inp_ts
    plt.figure()
    plt.xlabel('time')
    plt.ylabel('slot number')
    plt.title('Timeseries of packet arrival events')
    plt.plot(packet_inp_ts_arr, slot_num_arr, color='red', linestyle='None', marker='o')
    plt.plot(frame_strt_ts_arr, np.ones(len(frame_strt_ts_arr)), color='blue', linestyle='None', marker='o')
    plt.show()

    # Plot timeseries of (packet_inp_ts-framest_ts_arr)*1000 against packet_inp_ts
    plt.figure()
    plt.xlabel('packet_inp_ts')
    plt.ylabel('(packet_inp_ts-framest_ts_arr)*1000')
    plt.title('Timeseries of (packet_inp_ts-framest_ts_arr)*1000')
    plt.plot(packet_inp_ts_arr, (np.array(packet_inp_ts_arr)-np.array(frame_strt_ts_arr))*1000, color='green', linestyle='None', marker='o')
    plt.show()

    # Plot timeseries of packet_size_arr
    plt.figure()
    plt.xlabel('packet_inp_ts')
    plt.ylabel('packet sizes [bytes]')
    plt.title('Timeseries of packet sizes')
    plt.plot(packet_inp_ts_arr, packet_size_arr, color='green', linestyle='None', marker='o')
    plt.show()






