# from os import environ; environ['OMP_NUM_THREADS'] = '1'

import os
from datetime import datetime
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


def logger_process():
    global queue
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # handler = logging.StreamHandler()

    # Configura arquivo de logging
    time_stamp = datetime.now().strftime("%Y-%m-d_%H-%M%S")
    log_file_name = os.path.join(config.LOG_DIR,
                                 f'run_{time_stamp}.log')
    handler = logging.FileHandler(log_file_name)
    logger.addHandler(handler)

    # Configura formatação dos logs
    formatter = logging.Formatter(
        '%(asctime)s - %(process)d - %(processName)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    while True:
        message = queue.get()

        if message is None:
            break

        logger.handle(message)


def run_experiments(args, n_queries,
                    results_dir,
                    initial_labeled_size=None,
                    random_state=None, n_runs=1, n_folds=5):

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

    logger.info("Processo iniciado")

    exp = ActiveLearningExperiment(dataset_file,
                                   estimator=estimator,
                                   query_strategy=query_strategy,
                                   n_queries=n_queries,
                                   n_runs=n_runs,
                                   n_folds=n_folds,
                                   results_dir=results_dir,
                                   random_state=random_state,
                                   estimator_name=estimator_name)

    with warnings.catch_warnings():

        # Mostra warnings nos logs
        def custom_show_warning(message, category, filename, lineno,
                                file=None, line=None):
            logger.warning(str(message))

        warnings.showwarning = custom_show_warning

        try:
            exp.run_strategy()

        except Exception as e:
            logger.error(str(e))
            return

    logger.info("Processo finalizado")


if __name__ == '__main__':

    # Configurando o Logger:
    global queue
    queue = Queue()

    if not os.path.exists(config.LOG_DIR):
        os.mkdir(config.LOG_DIR)

    logger_p = Process(target=logger_process)
    logger_p.start()

    main_logger = logging.getLogger()
    main_logger.setLevel(logging.DEBUG)

    handler = logging.handlers.QueueHandler(queue)
    # handler.setFormatter(formatter)

    main_logger.addHandler(handler)

    # Iniciando experimentos
    main_logger.info('Iniciando experimentos')

    datasets = sorted([os.path.join(config.CSV_DIR, f)
                       for f in os.listdir(config.CSV_DIR)])

    args = (datasets, config.CLASSIFIER_DICT, config.SAMPLING_METHODS)

    # TODO: verificar RANDOM_STATE para garantir que a pool de dados
    # inicial dos experimentos é a mesma
    with Pool(config.N_WORKERS) as p:

        run_experiments_partial = partial(run_experiments,
                                          results_dir=config.RESULTS_DIR,
                                          n_queries=config.N_QUERIES,
                                          n_folds=config.N_SPLITS,
                                          n_runs=config.N_RUNS)

        experiments_pool = p.imap_unordered(run_experiments_partial,
                                            product(*args))

        pbar = tqdm(experiments_pool,
                    total=reduce(lambda x, y: x*y, map(len, args)))

        list(pbar)

    main_logger.info('Experimentos finalizados.')
    queue.put(None)
