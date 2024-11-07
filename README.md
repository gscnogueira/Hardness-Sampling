# Hardness Sampling

This repository contains the code and data used for the experiments described in the paper **"Hardness Sampling: Exploring Instance Hardness in Pool-based Active Learning"**. 
The repository structure is organized to facilitate reproducing the experiments and analyzing the results.

## Repository Structure

Below is an overview of the main folders and files in the repository:
- [datasets](datasets/): Datasets used in the experiments.
- [experiments](experiments/): Scripts and configurations for running experiments.
- [logs](logs/): Log files generated during experiment runs.
- [notebooks](notebooks/): Jupyter notebooks for exploratory analysis of the results.
- [results](results/): Results generated after running the experiments.
- [scripts](scripts/): Helper scripts for data processing.

## Requirements

- **Python Version**: 3.8.19.
- **Libraries**: Listed in the `requirements.txt` files within the `experiments` and `notebooks` directories.

## Setting Up the Environment

Set up the environment for running the experiments by cloning this repository and navigating to its main folder:

```bash
git clone 'https://github.com/gscnogueira/Hardness-Sampling'
cd Hardness-Sampling
```

## Datasets


For the experiments, a diverse collection of 90 classification datasets was used. These datasets are located in the `datasets/csv` directory and represent a subset of the datasets used in the work of [Pereira-Santos et al. (2019)](https://doi.org/10.1016/j.neucom.2017.05.105), which are stored in the `datasets/arff` directory.

## Running the Experiments

1. Go to the `experiments` directory:
```bash
cd experiments
```
2. Create and activate the virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Ajusted the desired experimental settings in [`experiments/config.py`](experiments/config.py), if needed
5. Run the main script:
```bash
python main.py
```

## Results

The results for a given configuration tested are stored in a file within the `results` directory, identified by the file name in the format `dataset#algorithm#method.csv`.
Each CSV file contains 100 rows and 5 columns.
Thus, a cell located in row i and column j represents the Cohen's kappa coefficient obtained by the configuration in the i-th iteration of the active learning process, using the j-th fold of cross-validation.

# Logs


The log files are located in the `logs/` directory and are used to analyze the behavior of the strategies during the experiments.
Initially, it was proposed to have one log file per configuration, following a naming convention similar to that of the results files.
However, this approach had to be changed due to concurrency issues, so the current implementation stores all logs in the `logs/experiments.log` file.
Nevertheless, the logs of previously run experiments have been preserved to avoid redundant work.

## Jupyter notebooks

Refer to the notebooks in [`notebooks/`](notebooks) for detailed analysis of the results and logs generated during the experiments.

## Credits

This project was developed to support the paper **Hardness Sampling: Exploring Instance Hardness in Pool-based Active Learning** by Gabriel da S. C. Nogueira, Davi, P. dos Santos, and Luís P. F. Garcia.
