import os
import numpy as np
import pandas as pd
from scipy.io import arff

DATASET_DIR = '../datasets/arff/'
DATASET_LIST_FILE = '../datasets/datasets.txt'

if __name__ == '__main__':

    # arff_files = os.listdir(DATASET_DIR)

    dataset_names = sorted([f.strip() for f in open(DATASET_LIST_FILE)])

    records = []
    for dataset in dataset_names:
        data , meta_data = arff.loadarff(os.path.join(DATASET_DIR,
                                                      dataset + ".arff"))

        row = {}
        # Nome do dataset (sem -, e extensao)
        row['Name'] = dataset

        # Quantidade de instancias
        row['#in'] = len(data)
        
        # Quantidade de classes
        row['#cl'] = len(np.unique(data['Class']))

        # Quantidade de atributos
        ats = list(meta_data.types())[:-1]
        row['#at'] = len(ats)

        # Quantidade de nominais
        row['#no'] = len([at for at in ats if at == 'nominal'])

        records.append(row)

    split_point = 45
    df = pd.DataFrame().from_records(records)
    df['Name'] = df['Name'].apply(lambda x: x[:5])

    print(df.iloc[:split_point].to_latex())
    print('--------------------')
    print(df.iloc[split_point:].to_latex())
