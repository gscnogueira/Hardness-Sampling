from os import environ; environ['OMP_NUM_THREADS'] = '1'

import os
import sys
from functools import partial, reduce
import logging
import warnings
from multiprocessing import Pool, Process
from multiprocessing import Queue, current_process
from itertools import product

import pandas as pd
from tqdm import tqdm

from active_learning_experiment import ActiveLearningExperiment
import config

formatter = logging.Formatter(
    '%(asctime)s - %(process)d - %(processName)s - %(levelname)s - %(message)s')

def logger_process():
    global queue
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler = logging.FileHandler(os.path.join(config.LOG_DIR,
                                               'experiments.log'))
    logger.addHandler(handler)
    handler.setFormatter(formatter)

    while True:
        message = queue.get()

        if message is None:
            break

        logger.handle(message)


def run_experiments(args, n_queries=100, initial_labeled_size=5,
                    random_satate=None, n_splits=5):

    logger = logging.getLogger()


    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():
        global queue
        logger.addHandler(logging.handlers.QueueHandler(queue))

    dataset_file, estimator_name, query_strategy = args

    estimator = config.CLASSIFIER_DICT[estimator_name]

    dataset_name, _ = os.path.splitext(dataset_file)


    # Muda o nome do processo
    process = current_process()
    process.name = f"({dataset_name}, {estimator_name}, {query_strategy.__name__})"

    # Nome do arquivo que irá registrar resultados
    file_name = (f'{dataset_name}#'
                 f'{estimator_name}#'
                 f'{query_strategy.__name__}.csv')

    # Caminho dos resultados
    results_path = os.path.join(config.RESULTS_DIR, file_name)

    if os.path.exists(results_path):
        logger.info("Experimento já havia sido realizado")
        return

    logger.info("Processo iniciado")

    df = pd.read_csv(os.path.join(config.CSV_DIR, dataset_file))

    exp = ActiveLearningExperiment(data=df,
                                   initial_labeled_size=initial_labeled_size,
                                   n_queries=n_queries,
                                   random_state=random_satate)

    with warnings.catch_warnings():

        # Mostra warnings nos logs
        def custom_show_warning(message, category, filename, lineno,
                                file=None, line=None):
            logger.warning(str(message))

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

    datasets = sorted([f for f in os.listdir(config.CSV_DIR)])

    args = (datasets, config.CLASSIFIER_DICT, config.SAMPLING_METHODS)

    global queue
    queue = Queue()

    logger_p = Process(target=logger_process)
    logger_p.start()

    main_logger = logging.getLogger()
    main_logger.setLevel(logging.DEBUG)

    handler = logging.handlers.QueueHandler(queue)
    # handler.setFormatter(formatter)

    main_logger.addHandler(handler)

    main_logger.info('Iniciando experimentos')

    with Pool(config.N_WORKERS) as p:

        run_experiments_partial = partial(run_experiments,
                                          n_queries=config.N_QUERIES,
                                          n_splits=config.N_SPLITS)

        experiments_pool = p.imap_unordered(run_experiments_partial,
                                            product(*args))

        pbar = tqdm(experiments_pool,
                    total=reduce(lambda x, y: x*y, map(len, args)))

        list(pbar)

    main_logger.info('Experimentos finalizados.')
    queue.put(None)
