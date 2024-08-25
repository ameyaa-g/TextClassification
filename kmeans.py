
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier

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

clf_knn = KNeighborsClassifier(n_neighbors=4)
clf_knn.fit(Xtr, ytr)
y_pred = clf_knn.predict(Xtt)

y_pred_score_knn = clf_knn.predict_proba(Xtt)

from sklearn.metrics import classification_report, confusion_matrix
conf_mx = confusion_matrix(ytt, y_pred)
print(classification_report(ytt, y_pred))

plt.matshow(conf_mx, cmap=plt.cm.get_cmap('Greys'))

plt.show()

