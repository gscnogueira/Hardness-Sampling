import os

import pandas as pd
from tqdm import tqdm
from scipy.io import arff
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer



def arff_to_csv(file_path):
    with open(file_path) as f:
        data, meta_data = arff.loadarff(f)

        df = pd.DataFrame(data)

        one_hot_encoder = OneHotEncoder(handle_unknown='ignore',
                                        sparse_output=False)

        cat_indicator = [dtype == 'nominal' and i != len(meta_data.types()) - 1
                         for i, dtype in enumerate(meta_data.types())]

        transformers = [
            ('one-hot-encoder', one_hot_encoder, cat_indicator),
        ]

        preprocessor = ColumnTransformer(transformers, remainder='passthrough')
        preprocessor.set_output(transform='pandas')

        dataset = preprocessor.fit_transform(df)
        dataset.iloc[:, -1] = LabelEncoder().fit_transform(dataset.iloc[:, -1])

        assert not dataset.isnull().values.any(),\
            "Dataset cannot contain NaNs or missing values"

        return dataset


if __name__ == '__main__':

    DATASET_DIR = 'datasets/arff/'
    DATASET_LIST_PATH = 'datasets.txt'

    arff_files = [f'{f_name.strip()}.arff'
                  for f_name in open(DATASET_LIST_PATH)]

    for arff_file in tqdm(arff_files):
        df = arff_to_csv(os.path.join(DATASET_DIR, arff_file))

        dataset_name, _ = os.path.splitext(arff_file)
        csv_path = os.path.join('datasets', 'csv', f'{dataset_name}.csv')

        df.to_csv(csv_path, index=False)
