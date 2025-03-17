from functools import partial
import os

from sklearn.svm import SVC
import pandas as pd
from modAL.uncertainty import margin_sampling
from tqdm import tqdm

from active_learning_experiment import ActiveLearningExperiment
from strategies.random import random_sampling


N_EXECS = 5
query_strategy = margin_sampling


def simulate(method):
    results_gen = (exp.run_strategy(estimator=learner,
                                    query_strategy=query_strategy)
                   for _ in range(N_EXECS))

    results_df = pd.concat(tqdm(results_gen, total=5), axis=1)

    dataset_file_name = os.path.split(dataset_file)[-1]

    results_df.to_csv(
        f'../notebooks/{method.__name__}_{N_EXECS}_{dataset_file_name}')


if __name__ == '__main__':
    dataset_file = '../datasets/csv/statlog-image-segmentation.csv'

    # Dataset que vai muito mal:
    dataset_file = '../datasets/csv/ozone-eighthr.csv'

    data = pd.read_csv(dataset_file)

    exp = ActiveLearningExperiment(data, n_queries=100, random_state=0)

    learner = partial(SVC, probability=True)

    print(f"Running random sampling...")
    simulate(random_sampling)

    print(f"Running {query_strategy.__name__}...")
    simulate(query_strategy)

