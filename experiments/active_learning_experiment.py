from functools import partial
import warnings
from datetime import datetime
import os
import csv
from typing import Callable


from modAL.models import ActiveLearner
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator

from dataclasses import dataclass
from sklearn.metrics import cohen_kappa_score
import numpy as np
import pandas as pd


@dataclass
class ActiveLearningExperiment:
    csv_file: str
    estimator: Callable
    query_strategy: Callable
    n_queries: int
    n_runs: int
    n_folds: int
    results_dir: str
    initial_labeled_size: int = 5
    batch_size: int = 1
    random_state: int = None
    estimator_name: str = None

    def __post_init__(self):

        os.environ["PYHARD_SEED"] = str(self.random_state)
        # Gerador aleatório para reproducibilidade dos resultados
        self.rng = np.random.default_rng(seed=self.random_state)

        # Carrega dados
        data = pd.read_csv(self.csv_file)
        self.X = data.iloc[:, :-1].to_numpy()
        self.y = data.iloc[:, -1].to_numpy()

        # Estabelece arquivo para armazenar resultados:
        self.dataset_name = os.path.split(self.csv_file)[-1][:-4]

        # Resolve nome do classificador
        if self.estimator_name is None:
            self.estimator_name = self.estimator.__name__

        result_file_name = "_".join([
            self.dataset_name,
            f'{self.n_runs}x{self.n_folds}',
            self.estimator_name,
            self.query_strategy.__name__]) + ".csv"

        self.result_file = os.path.join(self.results_dir, result_file_name)

    def run_strategy(self):

        skf = StratifiedKFold(n_splits=self.n_folds,
                              shuffle=True,
                              random_state=self.random_state)

        # TODO: tratar warning para quando a LPC tiver menos que
        # fold_number instâncias
        for run_number in range(self.n_runs):
            for fold_number, (train_index, test_index) in enumerate(
                    skf.split(self.X, self.y)):

                self.__run_fold(self.estimator, self.query_strategy,
                                train_index, test_index,
                                run_number, fold_number)

    def __run_fold(self, estimator: BaseEstimator, query_strategy,
                   train_index, test_index, run_number, fold_number):


        X_train, y_train = self.X[train_index], self.y[train_index]
        X_test, y_test = self.X[test_index], self.y[test_index]

        # Seleciona uma instância de cada classe para serem rotuladas
        unique_classes = np.unique(y_train)
        l_index = []

        rng = np.random.default_rng(seed=self.random_state + fold_number)

        for cls in unique_classes:
            cls_idxs = np.where(y_train == cls)[0]
            random_idx = rng.choice(cls_idxs)
            l_index.append(random_idx)

        # Se houver um numero de instâncias rotuladas menor que o
        # necessário, mais instâncias até que esse número seja
        # atingido
        if (n_missing := self.initial_labeled_size - len(l_index)) > 0:

            # Indices dos dados que ainda não foram slecionados para
            # rotulação
            missing_indexes = np.setdiff1d(np.arange(y_train.size),
                                           l_index)

            additional_index = rng.choice(missing_indexes,
                                          size=n_missing,
                                          replace=False)

            l_index.extend(additional_index)

        l_X_pool = X_train[l_index]
        l_y_pool = y_train[l_index]

        u_X_pool = np.delete(X_train, l_index, 0)
        u_y_pool = np.delete(y_train, l_index)

        args = dict()
        args['estimator'] = estimator()
        args['X_training'] = l_X_pool
        args['y_training'] = l_y_pool
        args['query_strategy'] = partial(self.query_strategy,
                                         n_instances=self.batch_size)

        learner = ActiveLearner(**args)
        scores = []

        # Afere desempenho inicial do modelo
        initial_y_pred = learner.predict(X_test)
        initial_score = cohen_kappa_score(y_test, initial_y_pred)
        scores.append(initial_score)

        result_row = {
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            'dataset': self.dataset_name,
            'classifier': self.estimator_name,
            'method': self.query_strategy.__name__,
            'run': run_number,
            'fold': fold_number,
            'query': 0,
            'kappa': initial_score
            }

        self.__write_row(result_row)

        # Active Learning Loop
        for query_number in range(1, self.n_queries + 1):

            if np.size(u_y_pool) <= 0:
                break

            # Verifica se pool saturou número de queries
            query_index = (learner.query(u_X_pool)[0]
                           if len(u_X_pool) > self.batch_size
                           else np.arange(len(u_X_pool)))

            learner.teach(X=u_X_pool[query_index], y=u_y_pool[query_index])

            u_X_pool = np.delete(u_X_pool, query_index, 0)
            u_y_pool = np.delete(u_y_pool, query_index)

            y_pred = learner.predict(X_test)
            kappa_score = cohen_kappa_score(y_test, y_pred)

            scores.append(kappa_score)

            result_row['time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            result_row['kappa'] = kappa_score
            result_row['query'] = query_number

            self.__write_row(result_row)

    def __write_row(self, row: dict):
        with open(self.result_file, mode='a', newline='') as f:

            fieldnames = ["time", "dataset", "classifier",
                          "method", "run", "fold", "query", "kappa"]

            csv_writer = csv.DictWriter(f, fieldnames=fieldnames)

            # Escreve header se necessário
            if f.tell() == 0:
                csv_writer.writeheader()

            csv_writer.writerow(row)


if __name__ == '__main__':

    from sklearn.svm import SVC
    from strategies.random import random_sampling
    from strategies.hardness import intra_extra_ratio_sampling
    from strategies.expected_error import expected_error_reduction
    from modAL.uncertainty import margin_sampling
    from strategies import hardness as h

    estimator=partial(SVC, probability=True)
    estimator.__name__ = SVC.__name__

    exp = ActiveLearningExperiment('../datasets/csv/iris.csv',
                                   estimator=estimator,
                                   query_strategy=h.disjunct_class_percentage_sampling,
                                   n_queries=100,
                                   n_runs=1,
                                   n_folds=5,
                                   results_dir=".",
                                   random_state=42)

    exp.run_strategy()
