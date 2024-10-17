import argparse
from src.preprocess import preprocess_edaf
from src.preprocess import plot_arrival_data
from src.preprocess import create_training_dataset

# python main.py -t preprocess -s data/240928_082545_results
# python main.py -t plot_arrival_data -s data/240928_082545_results -c config/training_dataset_config.json -n test0
# python main.py -t create_training_dataset -s data/240928_082545_results -c config/training_dataset_config.json -n test0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Packet Arrival Prediction")
    parser.add_argument("-t", "--task", choices=[
            "preprocess", 
            "plot_arrival_data", 
            "create_training_dataset"
        ], 
        help="Specify the task to run"
    )
    parser.add_argument("-s", "--source", help="Specify the source directory")
    parser.add_argument("-c", "--config", help="Specify the json configuration file")
    parser.add_argument("-n", "--name", help="Specify the name of the dataset")
    args = parser.parse_args()

    if args.task == "preprocess":
        preprocess_edaf(args)
    elif args.task == "plot_arrival_data":
        plot_arrival_data(args)
    elif args.task == "create_training_dataset":
        create_training_dataset(args)
    else:
        print("Invalid task specified")

        

        

