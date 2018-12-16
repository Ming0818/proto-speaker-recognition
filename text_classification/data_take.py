#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 22:14:32 2018

@author: ricardo
"""
import pandas as pd
import json
import glob

path = '/home/ricardo/Documents/TFM/clasification_news/dataset/noticia_sp'
out = '/home/ricardo/Documents/TFM/clasification_news/transformation'
newss = glob.glob(path+'/*.json')
frag_e = 500
frag_s = 0
# int(len(news)*000.1)]


def get_labels(row):
    labels = row['url'].split('/')[3:5]
    row['label'] = labels
    return row


while (frag_e < len(newss)):
    df_list = []
    news = newss[frag_s:frag_e]
    for new in news:
        with open(new) as fil:
            df = pd.DataFrame(json.load(fil).items())
        df = df.set_index(0).T
        df = df[['language', 'url', 'title', 'text']]
        df = df.apply(get_labels, axis=1)
        df_list.append(df)
    try:
        df = pd.concat(df_list, ignore_index=True)
    except Exception:
        print(len(df_list))
    df.to_csv('{}/news{}-{}.csv'.format(out, frag_s, frag_e), encoding='utf-8')
    print('done')
    frag_e += 500
    frag_s += 500
