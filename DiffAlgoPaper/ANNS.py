import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups



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


from sklearn.neural_network import MLPClassifier


clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(15,), random_state=1)

clf.fit(Xtr, ytr)

y_pred = clf.predict(Xtt)

from sklearn.metrics import classification_report, confusion_matrix
conf_mx = confusion_matrix(ytt, y_pred)
print(classification_report(ytt, y_pred))


plt.matshow(conf_mx, cmap=plt.cm.get_cmap('pink'))

plt.show()

