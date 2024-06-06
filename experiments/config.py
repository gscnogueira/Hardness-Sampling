from functools import partial
from sklearn.svm import SVC

ARFF_DIR = '../datasets/arff/'
CSV_DIR = '../datasets/csv'
RESULTS_DIR = '../results/'
N_WORKERS = 11
LOG_FILE = 'experiments.log'

CLASSIFIER_DICT = {
    "SVC": partial(SVC, probability=True)
}
