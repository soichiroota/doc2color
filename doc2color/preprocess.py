from models import BertModelWithTokenizer

import pandas as pd
import numpy as np
import torch

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
# logging.basicConfig(level=logging.INFO)


def preprocess(name_series, method):
    if method == 'capitalize':
        return name_series.str.capitalize()
    elif method == 'lower':
        return name_series.str.lower()
    elif method ==  'upper':
        return name_series.str.upper()
    elif method == 'title':
        return name_series.str.title()
    return name_series


if __name__ == '__main__':
    df = pd.read_csv('doc2color/data/color.csv')

    rgb_dicts = []
    for rgb in df['Hex (RGB)']:
        rgb_dict = {
            'R': int(rgb[1:3], 16),
            'G': int(rgb[3:5], 16),
            'B': int(rgb[5:], 16)
        }
        print(rgb, rgb_dict)
        rgb_dicts.append(rgb_dict)
    rgb_df = pd.DataFrame(rgb_dicts)
    rgb_df.to_csv('doc2color/data/rgb.csv', sep=',', index=False)

    model = BertModelWithTokenizer("bert-base-multilingual-cased")

    doc_types = ('capitalize', 'lower', 'upper', 'title')
    for doc_type in doc_types:
        names = preprocess(df['Name'], doc_type)
        vecs = np.array([model.get_sentence_embedding(name) for name in names])
        vec_df = pd.DataFrame(vecs)
        print(vec_df.shape)
        vec_df.to_csv('doc2color/data/{file_name}.csv'.format(file_name=doc_type), sep=',', index=False)
