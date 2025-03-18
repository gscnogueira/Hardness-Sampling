from functools import partial
import warnings
import time
import os


from modAL.models import ActiveLearner
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator

from sklearn.metrics import cohen_kappa_score
import numpy as np
import pandas as pd
import sqlite3


class ActiveLearningExperiment:
    def __init__(self, csv_file, result_db, n_queries: int,
                 batch_size=1, initial_labeled_size: int = 5,
                 random_state=None):

        self.dataset_name = os.path.split(csv_file)[-1]
        self.batch_size = batch_size
        self.n_queries = n_queries
        self.random_state = random_state
        self.initial_labeled_size = initial_labeled_size
        self.rng = np.random.default_rng(random_state)

        self.conn = sqlite3.connect(result_db, check_same_thread=False)

        data = pd.read_csv(csv_file)

        self.X = data.iloc[:, :-1].to_numpy()
        self.y = data.iloc[:, -1].to_numpy()

    def run_strategy(self, estimator: BaseEstimator,
                     query_strategy, n_runs=1, n_folds=5) -> pd.DataFrame:

        skf = StratifiedKFold(n_splits=n_folds,
                              shuffle=True,
                              random_state=self.random_state)

        for run_number in range(n_runs):
            for fold_number, (train_index, test_index) in enumerate(skf.split(self.X, self.y)):
                print(run_number, fold_number)
                self.__run_fold(estimator, query_strategy, train_index,
                                test_index, run_number, fold_number)

    def __run_fold(self, estimator: BaseEstimator, query_strategy,
                   train_index, test_index, run_number, fold_number):

        X_train, y_train = self.X[train_index], self.y[train_index]
        X_test, y_test = self.X[test_index], self.y[test_index]

        # Seleciona uma instância de cada classe para serem rotuladas
        unique_classes = np.unique(y_train)
        l_index = []
        query_strategy_name = query_strategy.__name__

        for cls in unique_classes:
            cls_idxs = np.where(y_train == cls)[0]

            random_idx = self.rng.choice(cls_idxs)

            l_index.append(random_idx)

        # Se houver um numero de instâncias rotuladas menor que o
        # necessário, mais instâncias até que esse número seja
        # atingido
        if (n_missing := self.initial_labeled_size - len(l_index)) > 0:
            additional_index = self.rng.choice(y_train, size=n_missing)
            l_index.extend(additional_index)

        l_X_pool = X_train[l_index]
        l_y_pool = y_train[l_index]

        u_X_pool = np.delete(X_train, l_index, 0)
        u_y_pool = np.delete(y_train, l_index)

        args = dict()
        args['estimator'] = estimator()
        args['X_training'] = l_X_pool
        args['y_training'] = l_y_pool
        args['query_strategy'] = partial(query_strategy,
                                         n_instances=self.batch_size)

        learner = ActiveLearner(**args)
        scores = []

        # Afere desempenho inicial do modelo
        initial_y_pred = learner.predict(X_test)
        initial_score = cohen_kappa_score(y_test, initial_y_pred)
        scores.append(initial_score)

        # Active Learning Loop
        for query_number in range(self.n_queries):

            if np.size(u_y_pool) <= 0:
                break

            query_index, _ = learner.query(u_X_pool)
            execution_time = time.time()

            learner.teach(X=u_X_pool[query_index], y=u_y_pool[query_index])

            u_X_pool = np.delete(u_X_pool, query_index, 0)
            u_y_pool = np.delete(u_y_pool, query_index)

            y_pred = learner.predict(X_test)
            kappa_score = cohen_kappa_score(y_test, y_pred)

            scores.append(kappa_score)

            db_row = (self.dataset_name,
                      estimator.__name__,
                      query_strategy_name,
                      run_number,
                      fold_number,
                      query_number,
                      kappa_score,
                      execution_time)

            self.__write_results(db_row)

        return scores

    def __write_results(self,db_row):
        cursor = self.conn.cursor()

        cursor.execute("""
        INSERT INTO results
        (dataset, classifier, method, run, fold, query, kappa, execution_time) 
        VALUES (?, ? , ?, ?, ?, ?, ?,?)
        """, db_row)

        self.conn.commit()


def setup_database(db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset TEXT NOT NULL,
    classifier TEXT NOT NULL,
    method TEXT NOT NULL,
    run INTEGER NOT NULL,
    fold INTEGER NOT NULL,
    query INTEGER NOT NULL,
    kappa REAL NOT NULL,
    execution_time REAL NOT NULL)
    ''')

    conn.close()

if __name__ == '__main__':

    from sklearn.svm import SVC
    from strategies.random import random_sampling
    from strategies.hardness import intra_extra_ratio_sampling
    from modAL.uncertainty import margin_sampling

    database_file = "results.db"

    setup_database(database_file)

    exp = ActiveLearningExperiment('../datasets/csv/horse-colic-surgical.csv',
                                   result_db=database_file,
                                   n_queries=200,
                                   initial_labeled_size=5,
                                   random_state=42)

    estimator=partial(SVC, probability=True)
    estimator.__name__ = SVC.__name__
    scores = exp.run_strategy(estimator=estimator,
                              query_strategy=margin_sampling,
                              n_runs=2)

    print(scores)
