import requests
import lxml.html
import pandas as pd

import os


def scrape(url):
    response = requests.get(url)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    dfs = pd.read_html(response.text)
    if url == 'https://en.wikipedia.org/wiki/X11_color_names':
        cols = ['Name', 'Hex (RGB)', 'Red (RGB)', 'Green (RGB)', 'Blue (RGB)',
            'Hue (HSL/HSV)', 'Satur. (HSL)', 'Light (HSL)', 'Satur. (HSV)',
            'Value (HSV)']
        return pd.concat([dfs[2][cols], dfs[5][cols]])
    if dfs:
        return dfs[0]
    else:
        return None


if __name__ == '__main__':
    urls = (
        'https://en.wikipedia.org/wiki/X11_color_names',
        'https://en.wikipedia.org/wiki/List_of_colors:_A%E2%80%93F',
        'https://en.wikipedia.org/wiki/List_of_colors:_G%E2%80%93M',
        'https://en.wikipedia.org/wiki/List_of_colors:_N%E2%80%93Z')
    dfs = [scrape(url) for url in urls]
    df = pd.concat([d for d in dfs if d is not None])
    print(df.head(), df.columns, df.shape)
    df.to_csv('doc2color/data/color.csv', sep=',', index=False)
