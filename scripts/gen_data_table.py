import os
import numpy as np
import pandas as pd
from scipy.io import arff

DATASET_DIR = '../datasets/arff/'
DATASET_LIST_FILE = '../datasets/datasets.txt'


def truncate_text(text, max_lenght=22):
    return text if len(text) < max_lenght else text[:max_lenght - 3] + '...'


if __name__ == '__main__':

    # arff_files = os.listdir(DATASET_DIR)

    dataset_names = sorted([f.strip() for f in open(DATASET_LIST_FILE)])

    records = []
    for i, dataset in enumerate(dataset_names, 1):
        data , meta_data = arff.loadarff(os.path.join(DATASET_DIR,
                                                      dataset + ".arff"))

        row = {}
        # Nome do dataset (sem -, e extensao)
        row['Name'] = f'{str(i)}-{dataset}'

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

    # import pdb; pdb.set_trace()
    formatters = {"Name": truncate_text}
    df.iloc[:split_point].to_latex(index=False, formatters=formatters,
                                   buf=open('datasets_table.tex', 'w'))
    df.iloc[split_point:].to_latex(index=False, formatters=formatters,
                                   buf=open('datasets_table.tex', 'a'))
    df.describe().to_latex(buf=open('datasets_stats_table.tex',
                                    'w'),
                           float_format="%.2f",
                           escape=True)
