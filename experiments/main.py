from os import environ
# environ['OMP_NUM_THREADS'] = '1'

import os

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from modAL.uncertainty import margin_sampling
from tqdm import tqdm

from config import CSV_DIR, RESULTS_DIR
from active_learning_experiment import ActiveLearningExperiment
from strategies.random import random_sampling
from strategies.expected_error import expected_error_reduction
from strategies.information_density import (density_weighted_sampling,
                                            training_utility_sampling)


def run_exeriments(dataset_file, estimator, query_strategy,
                   n_queries=100, initial_labeled_size=5,
                   random_satate=None):

    df = pd.read_csv(os.path.join(CSV_DIR, dataset_file))

    exp = ActiveLearningExperiment(data=df,
                                   initial_labeled_size=initial_labeled_size,
                                   n_queries=n_queries,
                                   random_state=random_satate)

    scores = exp.run_strategy(estimator=estimator,
                              query_strategy=query_strategy)

    dataset_name, _ = os.path.splitext(dataset_file)

    file_name = (f'{dataset_name}#'
                 f'{estimator.__name__}#'
                 f'{query_strategy.__name__}.csv')

    scores.to_csv(os.path.join(RESULTS_DIR, file_name))


if __name__ == '__main__':

    sampling_methods = [
        random_sampling,
        margin_sampling,
        density_weighted_sampling,
        training_utility_sampling,
        expected_error_reduction,
    ]

    datasets = (f for f in os.listdir(CSV_DIR))

    dataset = next(datasets)
    estimator = KNeighborsClassifier

    for method in (pbar := tqdm(sampling_methods)):

        pbar.set_description("["
                             f"{dataset}, "
                             f"{estimator.__name__}, "
                             f"{method.__name__}"
                             "]")

        run_exeriments(dataset,
                       estimator=estimator,
                       query_strategy=method,
                       n_queries=10)
