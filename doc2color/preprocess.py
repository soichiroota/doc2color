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


def get_name_hex_rgb_df(df):
    return df[['Name', 'Hex (RGB)']]


if __name__ == '__main__':
    hex_rgb = 'Hex (RGB)'
    main_df = pd.read_csv('doc2color/data/color.csv')
    df1 = pd.read_csv('doc2color/data/crayon1.csv').rename(columns={'Hexadecimal in their website depiction[b]': hex_rgb})
    df2 = pd.read_csv('doc2color/data/crayon2.csv').rename(columns={'Hexadecimal': hex_rgb})
    df5 = pd.read_csv('doc2color/data/crayon5.csv').rename(columns={'Hexadecimal': hex_rgb})
    df15 = pd.read_csv('doc2color/data/crayon15.csv').rename(columns={'Hex Code': hex_rgb})
    df17 = pd.read_csv('doc2color/data/crayon17.csv').rename(columns={'Hex Code': hex_rgb})
    scent_df17 = df17.rename(columns={'Scent Name': 'Name'})
    color_df17 = df17.rename(columns={'Color Name': 'Name'})
    df19 = pd.read_csv('doc2color/data/crayon19.csv').rename(columns={'Hex Code': hex_rgb})
    color_dic_df = pd.read_csv('doc2color/data/color_dic.csv')

    concat_df = pd.concat([get_name_hex_rgb_df(d) for d in (main_df, df1, df2, df5, df15, scent_df17, color_df17, df19, color_dic_df)]).dropna()
    df = concat_df[concat_df['Hex (RGB)'] != 'unknown']

    rgb_dicts = []
    for i, row in df.iterrows():
        rgb = row[hex_rgb]
        print(row['Name'], rgb)
        rgb_dict = {
            'R': int(rgb[1:3], 16),
            'G': int(rgb[3:5], 16),
            'B': int(rgb[5:7], 16)
        }
        print(rgb, rgb_dict)
        rgb_dicts.append(rgb_dict)
    rgb_df = pd.DataFrame(rgb_dicts)
    print(rgb_df.shape)
    rgb_df.to_csv('doc2color/data/rgb.csv', sep=',', index=False)

    model = BertModelWithTokenizer("bert-base-multilingual-cased")

    doc_types = ('capitalize', 'lower', 'upper', 'title')
    for doc_type in doc_types:
        names = preprocess(df['Name'], doc_type)
        vecs = np.array([model.get_sentence_embedding(name) for name in names])
        vec_df = pd.DataFrame(vecs)
        print(vec_df.shape)
        vec_df.to_csv('doc2color/data/{file_name}.csv'.format(file_name=doc_type), sep=',', index=False)
