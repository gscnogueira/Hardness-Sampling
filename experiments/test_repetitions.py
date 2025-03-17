from functools import partial
import os


from scipy.io import arff
from sklearn.svm import SVC
import pandas as pd
from modAL.uncertainty import margin_sampling
from tqdm import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from active_learning_experiment import ActiveLearningExperiment
from strategies.random import random_sampling


N_EXECS = 5
query_strategy = margin_sampling


def simulate_dataset(dataset_name, method=random_sampling):

    dataset_file = f'../datasets/arff/{dataset_name}.arff'

    data, meta_data = arff.loadarff(dataset_file)
    data_df = pd.DataFrame(data)

    nominal_mask = [x == 'nominal' for x in meta_data.types()][:-1]
    numeric_mask = [not x for x in nominal_mask]

    X = data_df.iloc[:, :-1]
    y = data_df.iloc[:, -1]

    one_hot_encoder = OneHotEncoder(sparse_output=False)
    standard_scaler = StandardScaler()

    column_transformers = [
        ('one-hot-encoder', one_hot_encoder, nominal_mask),
        ('standard-scaler', standard_scaler, numeric_mask),
    ]

    preprocessor = ColumnTransformer(column_transformers, remainder='passthrough')
    preprocessor.set_output(transform="pandas")

    X_new = preprocessor.fit_transform(X)
    y_new = LabelEncoder().fit_transform(y)

    data = X_new.copy()
    data["Class"] = y_new

    exp = ActiveLearningExperiment(data, n_queries=200)
    learner = partial(SVC, probability=True)

    results_gen = (exp.run_strategy(estimator=learner,
                                    query_strategy=query_strategy)
                   for _ in range(N_EXECS))

    results_df = pd.concat(tqdm(results_gen, total=5), axis=1)

    return results_df

if __name__ == '__main__':

    for dataset_name in open('../datasets/datasets.txt'):
        dataset_name = dataset_name.strip()
        print(dataset_name)
        df = simulate_dataset(dataset_name)
        print(df.mean(axis=1).max())
        df.to_csv(f'test_reps_results/{dataset_name}.csv')
