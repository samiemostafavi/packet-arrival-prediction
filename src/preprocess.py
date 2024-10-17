import os, sys, gzip, json, sqlite3, random, pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from loguru import logger

from edaf.core.uplink.preprocess import preprocess_ul
from edaf.core.uplink.analyze_packet import ULPacketAnalyzer
from edaf.core.uplink.analyze_scheduling import ULSchedulingAnalyzer

if not os.getenv('DEBUG'):
    logger.remove()
    logger.add(sys.stdout, level="INFO")

# in case you have offline parquet journey files, you can use this script to decompose delay
# pass the address of a folder in argv with the following structure:
# FOLDER_ADDR/
# -- gnb/
# ---- latseq.*.lseq
# -- ue/
# ---- latseq.*.lseq
# -- upf/
# ---- se_*.json.gz

# create database file by running
# python preprocess.py results/240928_082545_results

# it will result a database.db file inside the given directory

def preprocess_edaf(args):

    folder_path = Path(args.source)
    result_database_file = folder_path / 'database.db'

    # GNB
    gnb_path = folder_path.joinpath("gnb")
    gnb_lseq_file = list(gnb_path.glob("*.lseq"))[0]
    logger.info(f"found gnb lseq file: {gnb_lseq_file}")
    gnb_lseq_file = open(gnb_lseq_file, 'r')
    gnb_lines = gnb_lseq_file.readlines()
    
    # UE
    ue_path = folder_path.joinpath("ue")
    ue_lseq_file = list(ue_path.glob("*.lseq"))[0]
    logger.info(f"found ue lseq file: {ue_lseq_file}")
    ue_lseq_file = open(ue_lseq_file, 'r')
    ue_lines = ue_lseq_file.readlines()

    # NLMT
    nlmt_path = folder_path.joinpath("upf")
    nlmt_file = list(nlmt_path.glob("se_*.json.gz"))[0]
    with gzip.open(nlmt_file, 'rt', encoding='utf-8') as file:
        nlmt_records = json.load(file)['oneway_trips']
    logger.info(f"found nlmt json file: {nlmt_file}")

    # Open a connection to the SQLite database
    conn = sqlite3.connect(result_database_file)
    # process the lines
    preprocess_ul(conn, gnb_lines, ue_lines, nlmt_records)
    # Close the connection when done
    conn.close()
    logger.success(f"Tables successfully saved to '{result_database_file}'.")
    

def plot_arrival_data(args):

    # read configuration from args.config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # read configuration
    folder_addr = Path(args.source)
    result_database_file = folder_addr / 'database.db'

    # read exp configuration from args.config
    with open(folder_addr / 'experiment_config.json', 'r') as f:
        exp_config = json.load(f)

    time_mask = config['time_mask']
    filter_packet_sizes = config['filter_packet_sizes']
    history_window_size = config['history_window_size']
    dataset_size_max = config['dataset_size_max']
    split_ratios = config['split_ratios']
    
    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']
    total_prbs_num = exp_config['total_prbs_num']
    symbols_per_slot = exp_config['symbols_per_slot']
    scheduling_map_num_integers = exp_config['scheduling_map_num_integers']
    max_num_frames = exp_config['max_num_frames']
    scheduling_time_ahead_ms = exp_config['scheduling_time_ahead_ms']

    # prepare the results folder
    results_folder_addr = folder_addr / 'arrivals_plots' / args.name
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
            #SCHED_OFFSET_S=0.004 # 4ms which is 8*slot_duration_ms
            SCHED_OFFSET_S=scheduling_time_ahead_ms/1000 # 4ms which is 8*slot_duration_ms
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
    plt.savefig(results_folder_addr / 'slot_numbers.png')

    # Plot timeseries of (packet_inp_ts-framest_ts_arr)*1000 against packet_inp_ts
    plt.figure()
    plt.xlabel('packet_inp_ts')
    plt.ylabel('(packet_inp_ts-framest_ts_arr)*1000')
    plt.title('Timeseries of (packet_inp_ts-framest_ts_arr)*1000')
    plt.plot(packet_inp_ts_arr, (np.array(packet_inp_ts_arr)-np.array(frame_strt_ts_arr))*1000, color='green', linestyle='None', marker='o')
    plt.savefig(results_folder_addr / 'time_difference.png')

    # Plot timeseries of packet_size_arr
    plt.figure()
    plt.xlabel('packet_inp_ts')
    plt.ylabel('packet sizes [bytes]')
    plt.title('Timeseries of packet sizes')
    plt.plot(packet_inp_ts_arr, packet_size_arr, color='green', linestyle='None', marker='o')
    plt.savefig(results_folder_addr / 'packet_sizes.png')


def create_training_dataset(args):

    # read configuration from args.config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # read configuration
    folder_addr = Path(args.source)
    result_database_file = folder_addr / 'database.db'

    # read exp configuration from args.config
    with open(folder_addr / 'experiment_config.json', 'r') as f:
        exp_config = json.load(f)

    time_mask = config['time_mask']
    filter_packet_sizes = config['filter_packet_sizes']
    history_window_size = config['history_window_size']
    dataset_size_max = config['dataset_size_max']
    split_ratios = config['split_ratios']
    dim_process = config['dim_process']

    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']
    total_prbs_num = exp_config['total_prbs_num']
    symbols_per_slot = exp_config['symbols_per_slot']
    scheduling_map_num_integers = exp_config['scheduling_map_num_integers']
    max_num_frames = exp_config['max_num_frames']
    scheduling_time_ahead_ms = exp_config['scheduling_time_ahead_ms']

    # Save configuration dictionary to a json file
    results_folder_addr = folder_addr / 'training_datasets' / args.name
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
            #SCHED_OFFSET_S=0.004 # 4ms which is 8*slot_duration_ms
            SCHED_OFFSET_S=scheduling_time_ahead_ms/1000 # 4ms which is 8*slot_duration_ms
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