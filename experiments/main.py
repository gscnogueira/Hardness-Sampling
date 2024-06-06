from os import environ
# environ['OMP_NUM_THREADS'] = '1'

import os
import sys
from functools import partial
import logging
import warnings
from multiprocessing import Pool
from itertools import product

import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from modAL.uncertainty import margin_sampling
from tqdm import tqdm

from config import CSV_DIR, RESULTS_DIR, N_WORKERS
from active_learning_experiment import ActiveLearningExperiment
from strategies.random import random_sampling
from strategies.expected_error import expected_error_reduction
from strategies.information_density import (density_weighted_sampling,
                                            training_utility_sampling)

import strategies.hardness as ih

classifier_dict = {
    "SVC": partial(SVC, probability=True)
}
def setup_logger(logger_name):

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger

def run_experiments(args, n_queries=100, initial_labeled_size=5,
                   random_satate=None, n_splits=5):

    dataset_file, estimator_name, query_strategy = args
    estimator = classifier_dict[estimator_name]

    logger_name = (
        "["
        f"{dataset_file}, "
        f"{estimator_name}, "
        f"{query_strategy.__name__}"
        "]")

    logger = setup_logger(logger_name)

    logger.info("Inciando processo")

    df = pd.read_csv(os.path.join(CSV_DIR, dataset_file))

    exp = ActiveLearningExperiment(data=df,
                                   initial_labeled_size=initial_labeled_size,
                                   n_queries=n_queries,
                                   random_state=random_satate)

    with warnings.catch_warnings():
        # warnings.simplefilter('error', UserWarning)

        try:
            scores = exp.run_strategy(estimator=estimator,
                                      query_strategy=query_strategy,
                                      n_splits=n_splits)
        except UserWarning as w:
            logger.warning(str(w))
            return 

        except Exception as e:
            logger.error(str(e))
            return 

    dataset_name, _ = os.path.splitext(dataset_file)

    file_name = (f'{dataset_name}#'
                 f'{estimator_name}#'
                 f'{query_strategy.__name__}.csv')

    scores.to_csv(os.path.join(RESULTS_DIR, file_name))

    logger.info("Processo finalizado")


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

    datasets = [f for f in os.listdir(CSV_DIR)]
    datasets = datasets[:1]

    args = product(datasets, classifier_dict.keys(), sampling_methods)

    with Pool(N_WORKERS) as p:

        run_experiments_partial = partial(run_experiments,
                                          n_queries=2,
                                          n_splits=2)

        experiments_pool = p.imap_unordered(run_experiments_partial, args)
        pbar = tqdm(experiments_pool, total=len(sampling_methods), file=sys.stdout)
        list(pbar)
        

