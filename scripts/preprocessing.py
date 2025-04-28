"""
Script que realiza o preprocessamento dos dados tabulares e
armazena num arquivo csv
"""

import os

import pandas as pd
from tqdm import tqdm
from scipy.io import arff
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def arff_to_csv(file_path):

    data, meta_data = arff.loadarff(file_path)
    data_df = pd.DataFrame(data)

    nominal_mask = [x == 'nominal' for x in meta_data.types()][:-1]
    numeric_mask = [not x for x in nominal_mask]

    X = data_df.iloc[:, :-1]
    y = data_df.iloc[:, -1]

    binarizer = OneHotEncoder(handle_unknown='ignore',
                              sparse_output=False)
    standardizer = StandardScaler()

    transformer_list = [
        ('binzarization',   binarizer,    nominal_mask),
        ('standardization', standardizer, numeric_mask),
    ]

    preprocessor = ColumnTransformer(transformer_list, remainder='passthrough')
    preprocessor.set_output(transform='pandas')

    data_df = preprocessor.fit_transform(X)
    data_df["Class"] = LabelEncoder().fit_transform(y)

    assert not data_df.isnull().values.any(),\
        "Dataset cannot contain NaNs or missing values"

    return data_df


if __name__ == '__main__':

    ARFF_DIR = '../datasets/arff/'
    CSV_DIR = '../datasets/csv/'
    DATASET_LIST_PATH = '../datasets/datasets.txt'

    arff_files = [f'{f_name.strip()}.arff'
                  for f_name in open(DATASET_LIST_PATH)]

    for arff_file in tqdm(arff_files):
        df = arff_to_csv(os.path.join(ARFF_DIR, arff_file))

        dataset_name, _ = os.path.splitext(arff_file)
        csv_path = os.path.join(CSV_DIR, f'{dataset_name}.csv')

        df.to_csv(csv_path, index=False)
