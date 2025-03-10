from functools import partial

from sklearn.svm import SVC
import pandas as pd
from active_learning_experiment import ActiveLearningExperiment
from strategies.random import random_sampling

from modAL.uncertainty import margin_sampling


if __name__ == '__main__':
    # dataset_file = '../datasets/csv/statlog-image-segmentation.csv'
    dataset_file = '../datasets/csv/ozone-eighthr.csv'

    data = pd.read_csv(dataset_file)

    exp = ActiveLearningExperiment(data, n_queries=100, random_state=42)

    learner = partial(SVC, probability=True)

    exp.run_strategy(estimator=learner, query_strategy=margin_sampling)
    # exp.run_strategy(estimator=learner, query_strategy=random_sampling)

