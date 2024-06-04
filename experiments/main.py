import os
from functools import partial

import pandas as pd
from modAL.uncertainty import uncertainty_sampling
from pyhard.measures import ClassificationMeasures

from sklearn.svm import SVC

from config import CSV_DIR
from ..active_learning_experiment import ActiveLearningExperiment

if __name__ == '__main__':

    datasets = (f for f in os.listdir(CSV_DIR))

    df = pd.read_csv(os.path.join(CSV_DIR, next(datasets)))

    exp = ActiveLearningExperiment(df,
                                   initial_labeled_size=5,
                                   n_queries=100,
                                   random_state=0)

    scores = exp.run_strategy(estimator=partial(SVC, probability=True),
                              query_strategy=uncertainty_sampling)
