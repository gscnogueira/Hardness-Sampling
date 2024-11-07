# Hardness Sampling

This repository contains the code and data used for the experiments described in the paper **"Hardness Sampling: Exploring Instance Hardness in Pool-based Active Learning"**. 
The repository structure is organized to facilitate reproducing the experiments and analyzing the results.

## Repository Structure

Below is an overview of the main folders and files in the repository:
- [datasets](datasets/): The datasets used in the experiments.
- [experiments](experiments/): Scripts and configurations for running experiments.
- [logs](logs/): Log files generated during experiment runs.
- [notebooks](notebooks/): Jupyter notebooks for exploratory analysis of the results.
- [results](results/): Results generated after running the experiments.
- [scripts](scripts/): Helper scripts for data processing.

## Requirements

- **Python Version**: 3.8.19.
- **Libraries**: Listed in the `requirements.txt` files in the `experiments` and `notebooks` directories.

## Setting Up the Environment

Setup the environment by simply cloning this repository and cding to its main folder:

```bash
git clone 'https://github.com/gscnogueira/Hardness-Sampling'
cd Hardness-Sampling
```

## Running experiments

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
4. Ajusted the dessired experimental settings if needed in `experiments/config.py`
5. Run the main script:
```bash
python main.py
```

## Jupyter notebooks

Refer to the notebooks in `notebooks/` for detailed analysis of the results and logs generated during the experiments.

## Credits

This project was developed to support the paper **Hardness Sampling: Exploring Instance Hardness in Pool-based Active Learning** by Gabriel da S. C. Nogueira, Davi, P. dos Santos, and Luís P. F. Garcia.
