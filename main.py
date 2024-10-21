import argparse
from src.preprocess import preprocess_edaf, plot_arrival_data, create_training_dataset
from src.train import train_model
from src.predict import generate_predictions, plot_predictions

# python main.py -t preprocess -s data/240928_082545_results
# python main.py -t plot_arrival_data -s data/240928_082545_results -c config/dataset_config.json -n test0
# python main.py -t create_training_dataset -s data/240928_082545_results -c config/dataset_config.json -n test0
# python main.py -t train_model -c config/training_config.yaml -i THP_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Packet Arrival Prediction")
    parser.add_argument("-t", "--task", choices=[
            "preprocess", 
            "plot_arrival_data", 
            "create_training_dataset",
            "train_model",
            "generate_predictions",
            "plot_predictions",
        ], 
        help="Specify the task to run"
    )
    parser.add_argument("-s", "--source", help="Specify the source directory")
    parser.add_argument("-c", "--config", help="Specify the configuration file")
    parser.add_argument("-n", "--name", help="Specify the name of the dataset")
    parser.add_argument("-i", "--id", help="Specify the training id")
    args = parser.parse_args()

    if args.task == "preprocess":
        preprocess_edaf(args)
    elif args.task == "plot_arrival_data":
        plot_arrival_data(args)
    elif args.task == "create_training_dataset":
        create_training_dataset(args)
    elif args.task == "train_model":
        train_model(args)
    elif args.task == "generate_predictions":
        generate_predictions(args)
    elif args.task == "plot_predictions":
        plot_predictions(args)
    else:
        print("Invalid task specified")

        

        

