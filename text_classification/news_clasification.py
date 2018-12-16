#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 12:05:51 2018

@author: ricardo
"""

import re
import pandas as pd  # CSV file I/O (pd.read_csv)
from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize
import numpy as np
import sklearn
import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score  # , confusion_matrix
# from sklearn import preprocessing
# from sklearn.model_selection import cross_val_score
# from skmultilearn.adapt import MLkNN
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

def get_words(headlines, stops):
    # ps = PorterStemmer()
    headlines_onlyletters = re.sub("[^a-zA-Z]", " ", headlines)
    # headlines_onlyletters = [''.join(ps.stem(word))
    # for word in headlines_onlyletters][0]
    # Remove everything other than letters
    words = headlines_onlyletters.lower().split()
    # Convert to lower case, split into individual words
    # Convert the stopwords to a set for improvised performance
    meaningful_words = [w for w in words if w not in stops]
    # Removing stopwords
    return(" ".join(meaningful_words))
    # Joining the words


def read_and_split():
    # news = pd.read_csv("./uci-news-aggregator.csv")
    # news = (news.loc[news['CATEGORY'].isin(['b', 'e'])])
    # news = news[:int(len(news)*0.1)]
    # X_train, X_test, Y_train, Y_test = sklearn.model_selection\
    #    .train_test_split(news["TITLE"], news["CATEGORY"], test_size=0.2)
    news = pd.read_csv('./result.csv')
    news = news[:int(len(news)*0.1)]
    news.dropna(inplace=True)
    X_train, X_test, Y_train, Y_test = sklearn.model_selection\
        .train_test_split(news["text"], news["tag"], test_size=0.2)
#    le = preprocessing.LabelEncoder()
#    le.fit(Y_train)
#    X_train = np.array(X_train)
#    X_test = np.array(X_test)
##    Y_train = le.transform(Y_train)
#    Y_train = np.array(Y_train)
##    Y_test = le.transform(Y_test)
#    Y_test = np.array(Y_test)
    return X_train, X_test, Y_train, Y_test



def clean_headline(X_train, X_test):
    cleanHeadlines_train = []  # To append processed headlines
    cleanHeadlines_test = []  # To append processed headlines
    number_reviews_train = len(X_train)  # Calculating the number of reviews
    number_reviews_test = len(X_test)  # Calculating the number of reviews
    not_bow = pd.read_csv('./bow_not.csv')
    not_bow = not_bow['Unnamed: 0'].tolist()
    stops = stopwords.words("spanish")
    stops = stops + not_bow
    for i in range(0, number_reviews_train):
        cleanHeadline = get_words(X_train[i], stops)
        cleanHeadlines_train.append(cleanHeadline)
    for i in range(0, number_reviews_test):
        cleanHeadline = get_words(X_test[i], stops)
        cleanHeadlines_test.append(cleanHeadline)
    return cleanHeadlines_train, cleanHeadlines_test


def bag_of_words(cleanHeadlines_train, cleanHeadlines_test):
    bagOfWords_train = vectorize.fit_transform(cleanHeadlines_train)
    X_train = bagOfWords_train.toarray()
    bagOfWords_test = vectorize.transform(cleanHeadlines_test)
    X_test = bagOfWords_test.toarray()
    return X_train, X_test


if __name__ == '__main__':
#    X_train, X_test, Y_train, Y_test = read_and_split()
    cleanHeadlines_train, cleanHeadlines_test = clean_headline(X_train, X_test)
    vectorize = sklearn.feature_extraction.text.TfidfVectorizer(
            max_features=1700)
    # vectorize = sklearn.feature_extraction.text.CountVectorizer(
    #         analyzer="word", max_features=1700)
    X_train, X_test = bag_of_words(cleanHeadlines_train, cleanHeadlines_test)
    # vocab = vectorize.get_feature_names()
#    nb = MultinomialNB()
#    nb.fit(X_train, Y_train)
#    print(nb.score(X_test, Y_test))
    logistic_Regression = LogisticRegression(multi_class='auto',
                                             solver='lbfgs')
    logistic_Regression.fit(X_train, Y_train)
    Y_predict = logistic_Regression.predict(X_test)
    print(accuracy_score(Y_test, Y_predict))
#    classifier = MLkNN(k=20)
#    classifier.fit(X_train, Y_train)
#    Y_predict = classifier.predict(X_test)
#    print(accuracy_score(Y_test, Y_predict))
#    support_machine = SVC(kernel='sigmoid')
#    support_machine.fit(X_train, Y_train)
#    Y_predict = support_machine.predict(X_test)
    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                          alpha=1e-3, random_state=42,
                                          max_iter=5, tol=None)),
                    ])
    sgd.fit(X_train, Y_train)
    Y_predict = sgd.predict(X_test)

#    pickle.dump(classifier, open('svr.out', 'wb'))
    print(accuracy_score(Y_test, Y_predict))

# support_machine = pickle.load( open( "svr.out", "rb" ) )
