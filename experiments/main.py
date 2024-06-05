from os import environ
from functools import partial
# environ['OMP_NUM_THREADS'] = '1'

import os

import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from modAL.uncertainty import margin_sampling
from tqdm import tqdm

from config import CSV_DIR, RESULTS_DIR
from active_learning_experiment import ActiveLearningExperiment
from strategies.random import random_sampling
from strategies.expected_error import expected_error_reduction
from strategies.information_density import (density_weighted_sampling,
                                            training_utility_sampling)

import strategies.hardness as ih

def run_exeriments(dataset_file, estimator, query_strategy,
                   n_queries=100, initial_labeled_size=5,
                   random_satate=None, n_splits=5):

    df = pd.read_csv(os.path.join(CSV_DIR, dataset_file))

    exp = ActiveLearningExperiment(data=df,
                                   initial_labeled_size=initial_labeled_size,
                                   n_queries=n_queries,
                                   random_state=random_satate)

    scores = exp.run_strategy(estimator=estimator,
                              query_strategy=query_strategy,
                              n_splits=n_splits)

    dataset_name, _ = os.path.splitext(dataset_file)

    file_name = (f'{dataset_name}#'
                 f'{estimator.__name__}#'
                 f'{query_strategy.__name__}.csv')

    scores.to_csv(os.path.join(RESULTS_DIR, file_name))


if __name__ == '__main__':

    hardness_methods = [
        ih.borderline_points_sampling,
        ih.class_balance_sampling,
        ih.class_likelihood_sampling,
        ih.class_likeliood_diff_sampling,
        ih.disjunct_class_percentage_sampling,
        ih.disjunct_size_sampling,
        ih.f1_sampling,
        ih.f2_sampling,
        ih.f3_sampling,
        ih.f4_sampling,
        ih.harmfulness_sampling,
        ih.intra_extra_ratio_sampling,
        ih.k_disagreeing_neighbors_sampling,
        ih.local_set_cardinality_sampling,
        ih.ls_radius_sampling,
        ih.minority_value_sampling,
        ih.tree_depth_pruned_sampling,
        ih.tree_depth_unpruned_sampling,
        ih.usefulness_sampling,
    ]

    classic_methods = [
        random_sampling,
        margin_sampling,
        density_weighted_sampling,
        training_utility_sampling,
        # expected_error_reduction,
    ]

    sampling_methods = hardness_methods + classic_methods

    datasets = (f for f in os.listdir(CSV_DIR))

    dataset = next(datasets)
    estimator = partial(SVC, probability=True)
    estimator.__name__ = 'SVC'

    for method in (pbar := tqdm(sampling_methods)):

        pbar.set_description("["
                             f"{dataset}, "
                             f"{estimator.__name__}, "
                             f"{method.__name__}"
                             "]")

        try:
            run_exeriments(dataset,
                           estimator=estimator,
                           query_strategy=method,
                           n_queries=2,
                           n_splits=2)

        except Exception as e:
            print('Erro:', type(e), e)
