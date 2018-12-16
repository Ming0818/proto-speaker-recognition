#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 14:44:10 2018

@author: ricardo
"""
import csv
import re
from nltk.corpus import stopwords
import pandas as pd


def preprocessing_words(row):
    if isinstance(row['text'], str):
        words = row['text'].lower()
        row['text'] = re.sub("[^a-záéíóú0-9]", " ", row['text'])
        # headlines_onlyletters = [''.join(ps.stem(word))
        # for word in headlines_onlyletters][0]
        # Remove everything other than letters
        words = row['text'].split()
        # Convert to lower case, split into individual words
        stops = set(stopwords.words("spanish"))
        # Convert the stopwords to a set for improvised performance
        meaningful_words = [w for w in words if w not in stops]
        # Removing stopwords
        row['text'] = (" ".join(meaningful_words))
    return row


def get_words(news):
    bow = {}
    for _, row in news.iterrows():
        if isinstance(row['text'], str):
            words = row['text'].split()
            for word in words:
                if word in bow:
                    bow[word] += 1
                else:
                    bow[word] = 1
    return bow


def get_idftf(news, bow):
    cats_list = news['tag'].unique().tolist()[1:-2]
    for cat in cats_list:
        category = {}
        cat = news[news['tag'] == cat].copy()
        for _, row in cat.iterrows():
            words = row['text'].split()
            for word in words:
                if word in category:
                    category[word] += 1
                else:
                    category[word] = 1
        tfidf = {}
        for word in category:
            if (category[word]/bow[word] < 0.5
                    and category[word]/bow[word] > 0.01):
                tfidf[word] = category[word]/bow[word]
    return tfidf


if __name__ == '__main__':
    # result.csv
    news = pd.read_csv('./result.csv')
#    news = news
    news = news.apply(preprocessing_words, axis=1)
    bow = get_words(news)
    bow_not = get_idftf(news, bow)
#    with open('bow_not.csv', 'w') as f:  # Just use 'w' mode in 3.x
#        w = csv.DictWriter(f, bow_not.keys())
#        w.writeheader()
#        w.writerow(bow_not)
    bow_not = pd.DataFrame(bow_not, index=['value']).T
    bow_not.to_csv('bow_not.csv')
    print('Done')
