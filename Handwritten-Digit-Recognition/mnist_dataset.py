# -*- coding: utf-8 -*-
"""lab02.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Ele5Nciip-MtrVb0a3phZKn3wrx8ERNS
"""

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
# fetch_openml - pobiera zbiory danych z otwartego repozytorium zbiorow danych

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print((np.array(mnist.data.loc[42]).reshape(28,28) > 0).astype(int))

# print(mnist.target[42]) # 7

pixels = np.array(mnist.data.loc[42]).reshape(28,28)
plt.imshow(pixels, cmap='gray')
plt.show()

# 3
X, y = mnist["data"], mnist["target"].astype(np.uint8)
X.shape

y.shape

y_sorted = y.sort_values(ascending=True)
y_sorted.index

y_sorted

X_sorted = X.reindex(y_sorted.index)

X_train_bad, X_test_bad = X_sorted[:56000], X_sorted[56000:]
y_train_bad, y_test_bad = y_sorted[:56000], y_sorted[56000:]
print(X_train_bad.shape)
print(y_train_bad.shape)
print(X_test_bad.shape)
print(y_test_bad.shape)

print(np.unique(y_train_bad))
print(np.unique(y_test_bad))
# [0 1 2 3 4 5 6 7]
# [7 8 9]
# zbiory sa niepoprawne bo w zbiorze tranginowym nie ma zadnych 8 ani 9, nie nauczy sie model o nich nic
# testowy tez analogicznie niepoprawny

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

print(np.unique(y_train))
print(np.unique(y_test))

# 4
y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)
print(y_train_0)
print(np.unique(y_train_0))
print(len(y_train_0))

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(n_jobs=-1,  random_state=42)
sgd_clf.fit(X_train, y_train_0)

print(y_test.head(10))
print(sgd_clf.predict(X_test.head(10)))

# dokladnosc
y_train_pred = sgd_clf.predict(X_train)
y_test_pred = sgd_clf.predict(X_test)

acc_train = sum(y_train_pred == y_train_0) / len(y_train_0)
acc_test = sum(y_test_pred == y_test_0) / len(y_test_0)

print(acc_train, acc_test)
# 0.9918571428571429 0.9903571428571428

acc_list = [acc_train, acc_test]

import pickle

with open("sgd_acc.pkl", "wb") as file1:  # wb - zapis w trybie binarnym
  pickle.dump(acc_list, file1)

# 3-punktowa walidacja krzyzowa dokladnosci (accuracy)
from sklearn.model_selection import cross_val_score
score = cross_val_score(sgd_clf, X_train, y_train_0, cv=3, scoring="accuracy", n_jobs=-1)
# cv=3 - liczba k, model jest dzielony na 3 czesci, na dwoch trenuje i testuje na jednym, potem zamienia zestawy
# scoring="accuracy" - dokladnosc jako metryka oceny
# n=-1 - wykrozystuje wszystkei rdzenie procesora (przyspiesza to obliczenia)
print(score)
# [0.98687523 0.98762522 0.98649952]

print(score.astype)

with open("sgd_cva.pkl", "wb") as file2:
  pickle.dump(score, file2)

# 5
# klasyfikacja wieloklasowa
sgd_m_clf = SGDClassifier(n_jobs=-1, random_state=42)
sgd_m_clf.fit(X_train, y_train)

print(y_test.head(10).values)
print(sgd_m_clf.predict(X_test.head(10)))

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

# walidacja krzyzowa (dla siebie nie rtzeba w zadaniu)
y_train_pred_cross = cross_val_predict(sgd_m_clf, X_train, y_train, cv=3, n_jobs=-1)
conf_mx_cross = confusion_matrix(y_train, y_train_pred_cross)
print(conf_mx_cross)

y_test_pred = sgd_m_clf.predict(X_test)
conf_mx = confusion_matrix(y_test, y_test_pred)
print(conf_mx)

print(conf_mx.astype)
print(conf_mx.shape)

print(conf_mx_cross.astype)
print(conf_mx_cross.shape)

with open("sgd_cmx.pkl", "wb") as file3:
  pickle.dump(conf_mx, file3)