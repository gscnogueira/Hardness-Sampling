from functools import partial

from modAL.uncertainty import margin_sampling
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import strategies.hardness as ih
from strategies.random import random_sampling
from strategies.expected_error import expected_error_reduction
from strategies.information_density import (density_weighted_sampling,
                                            training_utility_sampling)

# -----------EXPERIMENTAL SETTINGS-----------------

# number of queries for the active learning process
N_QUERIES = 100

# n_splits for cross-validation
N_SPLITS = 5

RESULTS_DIR = '../results/'

CLASSIFIER_DICT = {
    # "SVC": partial(SVC, probability=True),
    # "5NN": KNeighborsClassifier,
    "Decision Tree": DecisionTreeClassifier,
    # "Gaussian Naive Bayes",
}

SAMPLING_METHODS = [
    random_sampling,
    margin_sampling,
    density_weighted_sampling,
    training_utility_sampling,
    # expected_error_reduction,
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

# -----------DATA SETTINGS-------------------------
ARFF_DIR = '../datasets/arff/'
CSV_DIR = '../datasets/csv'

# -----------MULTIPROCESSING----------------------
N_WORKERS = 5

# -----------LOGGING-----------------------------
LOG_FILE = 'experiments.log'
