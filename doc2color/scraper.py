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


def scrape_crayon(url):
    response = requests.get(url)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    dfs = pd.read_html(response.text)
    for i, df in enumerate(dfs):
        print(df.head(), df.columns, df.shape, i)
    for n in (1, 2, 5, 15, 17, 19):
        dfs[n].to_csv('doc2color/data/crayon{}.csv'.format(n), sep=',', index=False)


def scrape_color_dic(url):
    response = requests.get(url)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    html = lxml.html.fromstring(response.text)
    names1 = html.xpath('//table[@class="colortable"]//td/a/text()[1]')
    names2 = html.xpath('//table[@class="colortable"]//td/a/span/text()')
    color_codes = html.xpath('//table[@class="colortable"]//td/a/text()[2]')
    print(names1, names2, color_codes)
    dicts = []
    for i in range(len(names1)):
        record = dict(name=names1[i], color_code=color_codes[i])
        dicts.append(record)
    for i in range(len(names2)):
        record = dict(name=names2[i], color_code=color_codes[i])
        dicts.append(record)
    if dicts:
        return pd.DataFrame(dicts).rename(columns={'name': 'Name', 'color_code': 'Hex (RGB)'})
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

    crayon_url = 'https://en.wikipedia.org/wiki/List_of_Crayola_crayon_colors'
    scrape_crayon(crayon_url)

    color_dic_urls = (
        'https://www.colordic.org/',
        'https://www.colordic.org/w',
        'https://www.colordic.org/y',
        'https://www.colordic.org/m')
    color_dic_dfs = [scrape_color_dic(url) for url in color_dic_urls]
    color_dic_df = pd.concat([d for d in color_dic_dfs if d is not None])
    print(color_dic_df.head(), color_dic_df.columns, color_dic_df.shape)
    color_dic_df.to_csv('doc2color/data/color_dic.csv', sep=',', index=False)
