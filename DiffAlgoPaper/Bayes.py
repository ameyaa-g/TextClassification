from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups

import numpy as np
import sys
from time import time

categories = ['alt.atheism', 'talk.religion.misc',
              'comp.graphics', 'sci.space']

newsgroups_test = fetch_20newsgroups(subset='test',
                                    remove=('headers', 'footers', 'quotes'),
                                    categories=categories)
newsgroups_train = fetch_20newsgroups(subset='train',
                                    remove=('headers', 'footers', 'quotes'),
                                    categories=categories)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
vectors_test = vectorizer.transform(newsgroups_test.data)


clf = MultinomialNB(alpha=.01)
clf.fit(vectors, newsgroups_train.target)
pred = clf.predict(vectors_test)

ytt = newsgroups_test.target
print("Bayes F1 score: ", metrics.f1_score(newsgroups_test.target,
pred, average='macro'))

from sklearn.metrics import classification_report, confusion_matrix
conf_mx = confusion_matrix(ytt, pred)
print(classification_report(ytt, pred))


plt.matshow(conf_mx, cmap=plt.cm.get_cmap('pink'))

plt.show()

