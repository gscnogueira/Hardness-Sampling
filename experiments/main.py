import os
from functools import partial

import pandas as pd
from modAL.uncertainty import uncertainty_sampling
from pyhard.measures import ClassificationMeasures

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from config import CSV_DIR, RESULTS_DIR
from active_learning_experiment import ActiveLearningExperiment


def run_exeriments(dataset_file, estimator, query_strategy):

    df = pd.read_csv(os.path.join(CSV_DIR, dataset_file))

    exp = ActiveLearningExperiment(df, initial_labeled_size=5,
                                   n_queries=100, random_state=0)

    scores = exp.run_strategy(estimator=estimator,
                              query_strategy=query_strategy)

    dataset_name, _ = os.path.splitext(dataset_file)
    file_name = (f'{dataset_name}_{estimator.__name__}'
                 f'_{query_strategy.__name__}.csv')

    scores.to_csv(os.path.join(RESULTS_DIR, file_name))


if __name__ == '__main__':

    datasets = (f for f in os.listdir(CSV_DIR))

    run_exeriments(next(datasets),
                   KNeighborsClassifier,
                   uncertainty_sampling)
