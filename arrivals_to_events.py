import os, sys, json
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
import random, pickle

if not os.getenv('DEBUG'):
    logger.remove()
    logger.add(sys.stdout, level="INFO")

config = {
    'db_address' : 'data/s20-2/database.db', #240928_082545_results
    'time_mask' : [0.1, 0.9], # portion of interest from the whole time series #[0.2, 0.9]
    'filter_packet_sizes' : [ 128 ], # only allows these sizes in bytes (note that this turns into a discrete event problem)
    'dim_process' : 1,
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
        logger.error("Usage: python arrivals_to_events.py <target-training-data-folder>")
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
    dim_process = config['dim_process']

    # Save configuration dictionary to a json file
    results_folder_addr.mkdir(parents=True, exist_ok=True)
    with open(results_folder_addr / 'config.json', 'w') as f:
        json_obj = json.dumps(config, indent=4)
        f.write(json_obj)

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

    packet_arrival_events = []
    last_event_ts = 0
    for packet in packets:
        if packet['len'] not in filter_packet_sizes:
            continue
        frame_start_ts, frame_num, slot_num = scheduling_analyzer.find_frame_slot_from_ts(
            timestamp=packet['ip.in_t'], 
            #SCHED_OFFSET_S=0.002 # 2ms which is 4*slot_duration_ms
            SCHED_OFFSET_S=0.004 # 4ms which is 8*slot_duration_ms
        )

        time_since_frame0 = frame_num*num_slots_per_frame*slots_duration_ms + slot_num*slots_duration_ms
        time_since_last_event = time_since_frame0-last_event_ts
        if time_since_last_event < 0:
            time_since_last_event = time_since_frame0
        last_event_ts = time_since_frame0
        packet_arrival_events.append(
            {
                'type_event' : 0,
                'time_since_start' : time_since_frame0,
                'time_since_last_event' : time_since_last_event,
                'timestamp' : packet['ip.in_t']
            }
        )
    
    dataset = []
    for idx,_ in enumerate(packet_arrival_events):
        if idx+history_window_size >= len(packet_arrival_events):
            break
        events_window = []
        for pos, event in enumerate(packet_arrival_events[idx:idx+history_window_size]):
            idx_event = pos
            events_window.append(
                {
                    'idx_event' : pos,
                    'type_event': event['type_event'],
                    'time_since_start' : event['time_since_start'],
                    'time_since_last_event' : event['time_since_last_event'],
                }
            )        

        #print(events)
        dataset.append(events_window)
        if len(dataset) > dataset_size_max:
            break

    # shuffle
    random.shuffle(dataset)

    # print length of dataset
    logger.info(f"Number of total entries in dataset: {len(dataset)}")
    print(dataset[0])

    # split
    train_num = int(len(dataset)*split_ratios[0])
    dev_num = int(len(dataset)*split_ratios[1])
    print("train: ", train_num, " - dev: ", dev_num)
    # train
    train_ds = {
        'dim_process' : dim_process,
        'train' : dataset[0:train_num],
    }

    # Save the dictionary to a pickle file
    with open(results_folder_addr / 'train.pkl', 'wb') as f:
        pickle.dump(train_ds, f)
    # dev
    dev_ds = {
        'dim_process' : dim_process,
        'dev' : dataset[train_num:train_num+dev_num],
    }
    # Save the dictionary to a pickle file
    with open(results_folder_addr / 'dev.pkl', 'wb') as f:
        pickle.dump(dev_ds, f)
    # test
    test_ds = {
        'dim_process' : dim_process,
        'test' : dataset[train_num+dev_num:-1],
    }
    # Save the dictionary to a pickle file
    with open(results_folder_addr / 'test.pkl', 'wb') as f:
        pickle.dump(test_ds, f)