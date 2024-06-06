from functools import partial
import warnings


from modAL.models import ActiveLearner
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator

from sklearn.metrics import f1_score
import numpy as np
import pandas as pd


class ActiveLearningExperiment:
    def __init__(self, data: pd.DataFrame, n_queries: int,
                 batch_size=1, initial_labeled_size: int = 5,
                 random_state=None):

        self.batch_size = batch_size
        self.n_queries = n_queries
        self.data = data
        self.random_state = random_state
        self.initial_labeled_size = initial_labeled_size

        self.X, self.y = data.iloc[:, :-1], data.iloc[:, -1]

    def run_strategy(self, estimator: BaseEstimator,
                     query_strategy, n_splits=5) -> pd.DataFrame:

        skf = StratifiedKFold(n_splits=n_splits,
                              shuffle=True,
                              random_state=self.random_state)
        results = [
            self.__run_fold(estimator, query_strategy, train_index, test_index)
            for train_index, test_index in skf.split(self.X, self.y)
        ]

        return pd.DataFrame(results).T

    def __run_fold(self, estimator: BaseEstimator, query_strategy,
                   train_index, test_index):

        X_train, y_train = self.X.iloc[train_index], self.y.iloc[train_index]
        X_test, y_test = self.X.iloc[test_index], self.y.iloc[test_index]

        # Seleciona uma instância de cada classe para serem rotuladas
        groups = y_train.groupby(y_train)
        l_index = groups.sample(1, random_state=self.random_state).index

        # Se houver um numero de instâncias rotuladas menor que o
        # necessário, mais instâncias até que esse número seja
        # atingido
        if (n_missing := self.initial_labeled_size - len(l_index)) > 0:

            new_index = y_train.drop(l_index).sample(
                n_missing, random_state=self.random_state).index

            l_index = l_index.append(new_index)

        l_X_pool = X_train.loc[l_index].values
        l_y_pool = y_train.loc[l_index].values

        u_X_pool = X_train.drop(l_index)
        u_y_pool = y_train.drop(l_index)

        args = dict()
        args['estimator'] = estimator()
        args['X_training'] = l_X_pool
        args['y_training'] = l_y_pool
        args['query_strategy'] = partial(query_strategy,
                                         n_instances=self.batch_size)

        scores = []
        learner = ActiveLearner(**args)

        # Active Learning Loop
        for idx in range(self.n_queries):

            if np.size(u_y_pool) <= 0:
                break

            query_index, _ = learner.query(u_X_pool.values)

            learner.teach(X=u_X_pool.iloc[query_index].values,
                          y=u_y_pool.iloc[query_index].values)

            query_index = u_X_pool.index[query_index]
            u_X_pool = u_X_pool.drop(query_index)
            u_y_pool = u_y_pool.drop(query_index)

            y_pred = learner.predict(X_test.values)

            score = f1_score(y_test, y_pred, average='macro')
            scores.append(score)

        return scores
