#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 13:16:25 2018

@author: ricardo
"""

import pandas as pd 

def news_in_one_line(row):
    if isinstance(row['text'], float):
        print('hey')
    else:
        row['text'] = row['text'].replace('\n', ' ')
    return row

def read_write_csv():
    df = pd.read_csv('data_all_filter.csv',usecols=['language', 'url', 'title',
                                                    'text'])
    df = df.apply(news_in_one_line, axis=1)
    df.to_csv('data_all_text_one_line.csv', encoding='utf-8')

if __name__ == '__main__':
    read_write_csv()