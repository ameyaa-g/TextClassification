import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df = pd.read_csv('/Users/ameya/Downloads/exoplanet_find - TOI_2022.07.09_05.07.28 (1).csv', names=['tfopwg_disp','ra', 'dec','st_pmra','st_pmdec',
                                                                                               'pl_tranmid','pl_orbper','pl_trandurh','pl_trandep',
                                                                                               'pl_rade','pl_insol','pl_eqt','st_tmag','st_dist',
                                                                                               'st_teff','st_logg','st_rad'])

# Dropping blank rows
df = df.dropna(axis=0)

print(df)



# Changing tfopwg_disp into int
le = preprocessing.LabelEncoder()
le.fit(df.tfopwg_disp)
df['tfopwg_disp'] = le.transform(df.tfopwg_disp)

print(df)


# splitting training and test datasets
datasets = train_test_split(df, df.tfopwg_disp, test_size=0.33, random_state=42)
train_data, test_data, train_labels, test_labels = datasets

# scaling training and test datasets
scaler = StandardScaler()
scaler.fit(train_data)

train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

# Training the Model
# creating an classifier from the model:
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000)

# fit the training data
mlp.fit(train_data, train_labels)

predictions_train = mlp.predict(train_data)
print(accuracy_score(predictions_train, train_labels))

predictions_test = mlp.predict(test_data)
print(accuracy_score(predictions_test, test_labels))


print(classification_report(predictions_test, test_labels))

conf_mx = confusion_matrix(predictions_test, test_labels)

plt.matshow(conf_mx, cmap=plt.cm.get_cmap('pink'))

plt.show()

