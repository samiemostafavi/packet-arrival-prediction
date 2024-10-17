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
    git clone https://github.com/username/repo.git
    ```

2. Change into the project directory:

    ```shell
    cd repo
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

To use the `database.db` file created by EDAF, you can insert edaf files and filders into `data` folder.
Here's an example of how the directory structure would look like:
```
├── data
│   └── 240928_082545_results
│       ├── gnb
│       ├── ue
│       ├── upf
│       └── database.db
```

Make sure to update the code in your project to reference the correct path to the `database.db` file.

Then run the code by executing the main script:

```shell
python arrivals_plot.py
python arrivals_to_events.py data/240928_082545_results/training_dataset
```
