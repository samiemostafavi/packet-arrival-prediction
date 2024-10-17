# README

This repository contains the code for packet arrival prediction using EDAF and EasyTemporalPointProcess projects.

## Dependencies

This code is tested with Python 3.9. 
To create a Python 3.9 environment with Conda, you can use the following command:

```shell
conda create --name pap python=3.9
```
This command will create a new Conda environment with Python 3.9 installed.

```shell
conda activate pap
```

## Installation

To install the required dependencies and create a Conda environment, follow these steps:

1. Clone the repository:

    ```shell
    git clone https://github.com/samiemostafavi/packet-arrival-prediction.git
    ```

2. Change into the project directory:

    ```shell
    cd packet-arrival-prediction
    ```

3. Create a new Conda environment:

    ```shell
    conda create --name pap python=3.9
    ```

4. Activate the Conda environment:

    ```shell
    conda activate pap
    ```

5. Install the required packages:

    ```shell
    pip install -r requirements.txt
    ```

## Usage

Once the dependencies are installed, you need to bring/create the database file.

To create the `database.db` file if it is not created, you can insert edaf raw results into `data` folder.

Here's an example of how the directory structure would look like:
```
├── data
│   └── 240928_082545_results
│       ├── gnb
│       ├── ue
│       └── upf
```

Run the following command to process the raw data and create `database.db` file:
```shell 
python main.py -t preprocess -s data/240928_082545_results
```

Create an experiment coonfiguration file using json format and insert it in this folder under the name `experiment_config.json`.
The file should be like:
```json
{
    "total_prbs_num": 106,
    "symbols_per_slot": 14,
    "slots_per_frame": 20,
    "slots_duration_ms": 0.5,
    "scheduling_map_num_integers": 4,
    "max_num_frames": 1024
}
```

Now you can remove gnb, ue, and upf folders to save space. Then the directory structure should look like:
```
├── data
│   └── 240928_082545_results
│       ├── experiment_config.json
│       └── database.db
```

### Create Training Dataset


Make sure to update/create the training dataset configuration file and insert it in the `config` folder under the name `training_dataset_config.json`.
The file should be like:
```json
{
    "time_mask": [0.1,0.9],
    "filter_packet_sizes": [128],
    "dim_process": 1,
    "history_window_size": 20,
    "dataset_size_max": 10000,
    "split_ratios": [0.7,0.15,0.15]
}
```

Then the directory structure should look like:
```
├── config
│   └── training_dataset_config.json
├── data
│   └── 240928_082545_results
│       ├── experiment_config.json
│       └── database.db
```

Now you can create a training dataset by executing the script:
```shell
python main.py -t plot_arrival_data -s data/240928_082545_results
python main.py -t create_training_dataset -s data/240928_082545_results -c config/training_dataset_config.json -n test0
```
