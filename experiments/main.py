from os import environ; environ['OMP_NUM_THREADS'] = '1'

import os
import sys
from functools import partial, reduce
import logging
import warnings
from multiprocessing import Pool
from itertools import product

import pandas as pd
from tqdm import tqdm

from active_learning_experiment import ActiveLearningExperiment
import config


def setup_logger(dataset_file, estimator_name, query_strategy):

    logger_name = (
        "["
        f"{dataset_file}, "
        f"{estimator_name}, "
        f"{query_strategy.__name__}"
        "]")

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(
        os.path.join(config.LOG_DIR, f'experiments_{estimator_name}.log'))

    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def run_experiments(args, n_queries=100, initial_labeled_size=5,
                   random_satate=None, n_splits=5):

    dataset_file, estimator_name, query_strategy = args
    estimator = config.CLASSIFIER_DICT[estimator_name]

    logger = setup_logger(dataset_file,
                          estimator_name,
                          query_strategy)

    dataset_name, _ = os.path.splitext(dataset_file)

    file_name = (f'{dataset_name}#'
                 f'{estimator_name}#'
                 f'{query_strategy.__name__}.csv')

    results_path = os.path.join(config.RESULTS_DIR, file_name)

    if os.path.exists(results_path):
        logger.info("Experimento já havia sido realizado")
        return 


    def custom_show_warning(message, category, filename, lineno,
                            file=None, line=None):
        logger.warning(str(message))

    logger.info("Processo iniciado")

    df = pd.read_csv(os.path.join(config.CSV_DIR, dataset_file))

    exp = ActiveLearningExperiment(data=df,
                                   initial_labeled_size=initial_labeled_size,
                                   n_queries=n_queries,
                                   random_state=random_satate)

    with warnings.catch_warnings():

        warnings.showwarning = custom_show_warning

        try:
            scores = exp.run_strategy(estimator=estimator,
                                      query_strategy=query_strategy,
                                      n_splits=n_splits)
        except Exception as e:
            logger.error(str(e))
            return


    scores.to_csv(results_path)

    logger.info("Processo finalizado")


if __name__ == '__main__':

    datasets = [f for f in os.listdir(config.CSV_DIR)]

    args = (datasets, config.CLASSIFIER_DICT, config.SAMPLING_METHODS)

    with Pool(config.N_WORKERS) as p:

        run_experiments_partial = partial(run_experiments,
                                          n_queries=config.N_QUERIES,
                                          n_splits=config.N_SPLITS)

        experiments_pool = p.imap_unordered(run_experiments_partial,
                                            product(*args))

        pbar = tqdm(experiments_pool, file=sys.stdout,
                    total=reduce(lambda x, y: x*y, map(len, args)))
        list(pbar)
