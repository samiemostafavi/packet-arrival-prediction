from easy_tpp.config_factory import Config
from src.tpprunner import TPPRunner
from pathlib import Path
import yaml, pickle
import numpy as np

def generate_predictions(args):
    model_path = Path(args.source) / "training_results" / args.name / args.id
    yaml_file = next(model_path.glob("*.yaml"))
    with open(yaml_file, 'r') as file:
        training_output_config = yaml.load(file, Loader=yaml.FullLoader)

    # fix the base_dir for the generation stage
    training_base_dir = training_output_config['base_config']['base_dir']
    prediction_base_dir = training_base_dir.replace("training_results", "prediction_results")

    experiment_id = "THP_predict"
    # Transform the dict to match training configuration format
    config = {
        "pipeline_config_id": "runner_config",
        "data": {
            training_output_config['base_config']['dataset_id']: {
                "data_format": training_output_config['data_config']['data_format'],
                "train_dir": training_output_config['data_config']['train_dir'],
                "valid_dir": training_output_config['data_config']['valid_dir'],
                "test_dir": training_output_config['data_config']['test_dir'],
                "data_specs": {
                    "num_event_types": training_output_config['data_config']['data_specs']['num_event_types'],
                    "pad_token_id": training_output_config['data_config']['data_specs']['pad_token_id'],
                    "padding_side": training_output_config['data_config']['data_specs']['padding_side'],
                    "truncation_side": training_output_config['data_config']['data_specs']['truncation_side'],
                }
            }
        },
        experiment_id: {
            "base_config": {
                "stage": "gen",
                "backend": training_output_config['base_config']['backend'],
                "dataset_id": training_output_config['base_config']['dataset_id'],
                "runner_id": training_output_config['base_config']['runner_id'],
                "model_id": training_output_config['base_config']['model_id'],
                "base_dir": prediction_base_dir,
            },
            "trainer_config": {
                "batch_size": training_output_config['trainer_config']['batch_size'],
                "max_epoch": training_output_config['trainer_config']['max_epoch'],
                "shuffle": training_output_config['trainer_config']['shuffle'],
                "optimizer": training_output_config['trainer_config']['optimizer'],
                "learning_rate": training_output_config['trainer_config']['learning_rate'],
                "valid_freq": training_output_config['trainer_config']['valid_freq'],
                "use_tfb": training_output_config['trainer_config']['use_tfb'],
                "metrics": training_output_config['trainer_config']['metrics'],
                "seed": training_output_config['trainer_config']['seed'],
                "gpu": -1,#training_output_config['trainer_config']['gpu'],
            },
            "model_config": {
                "hidden_size": training_output_config['model_config']['hidden_size'],
                "num_layers": training_output_config['model_config']['num_layers'],
                "loss_integral_num_sample_per_step": training_output_config['model_config']['loss_integral_num_sample_per_step'],
                "use_ln": training_output_config['model_config']['use_ln'],
                "pretrained_model_dir": training_output_config['base_config']['specs']['saved_model_dir'],
                "thinning": {
                    "num_seq": 10,
                    "num_sample": 1000,
                    "num_exp": 1000, # number of i.i.d. Exp(intensity_bound) draws at one time in thinning algorithm
                    "look_ahead_time": 10,
                    "patience_counter": 5, # the maximum iteration used in adaptive thinning
                    "over_sample_rate": 5,
                    "num_samples_boundary": 100,
                    "dtime_max": 20,
                    "num_step_gen": 1
                }
            }
        }
    }
    config = Config.build_from_dict(config, experiment_id=experiment_id)
    model_runner = TPPRunner(config)
    model_runner.run()


def plot_predictions(args):
    model_path = Path(args.source) / "prediction_results" / args.name / args.id
    pkl_file = next(model_path.glob("*.pkl"))
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)
    print(np.array(data['pred']).shape)
    print(np.array(data['label']).shape)

    pred = np.array(data['pred'])
    label = np.array(data['label'])
    
    import plotly.graph_objects as go

    # Get the prediction data
    y_vals = np.array(pred)[0,101,0,:]
    print(y_vals)
    x_vals = np.linspace(0, 20, 1000)
    print(x_vals)

    # Create a scatter plot using plotly
    fig = go.Figure(data=go.Scatter(x=x_vals, y=y_vals, mode='markers'))
    # Cap the y-axis to 1
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(range=[0, 15])
    # Set the layout of the plot
    fig.update_layout(title='Predictions', xaxis_title='Time', yaxis_title='Probability')

    # Save the plot as an HTML file
    fig.write_html(model_path / "prob_delta_times.html")

