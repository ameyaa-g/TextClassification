import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import LinearSVC
import numpy as np
import sys
from time import time

from sklearn.preprocessing import StandardScaler

categories = ['alt.atheism', 'talk.religion.misc',
              'comp.graphics', 'sci.space']

data_train = fetch_20newsgroups(subset='train', remove=('headers',
'footers', 'quotes'), categories=categories)
data_test = fetch_20newsgroups(subset='test', remove=('headers',
'footers', 'quotes'), categories=categories)


vectorizer = TfidfVectorizer()
data_train_vectors = vectorizer.fit_transform(data_train.data)
data_test_vectors = vectorizer.transform(data_test.data)

Xtr = data_train_vectors
ytr = data_train.target
Xtt = data_test_vectors
ytt = data_test.target

# Implementing classification model- using LinearSVC

# Instantiate the estimator
clf_svc = LinearSVC()

# Fit the model with data (aka "model training")
clf_svc.fit(Xtr, ytr)

# Predict the response for a new observation
y_pred = clf_svc.predict(Xtt)
#print "Predicted Class Labels:",y_pred

# Predict the response score for a new observation
y_pred_score_svc = clf_svc.decision_function(Xtt)
print("Predicted Score:\n",y_pred_score_svc)

from sklearn.metrics import classification_report, confusion_matrix
conf_mx = confusion_matrix(ytt, y_pred)
print(classification_report(ytt, y_pred))

plt.matshow(conf_mx, cmap=plt.cm.get_cmap('pink'))

plt.show()

