# -*- coding: utf-8 -*-
"""lab06.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1AdWbOquNVayTwtdU4dWgbnyheoN0oWG3

# **Laboratorium: Metody zespołowe**

### Breast Cancer Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split

from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)

X = data_breast_cancer.data[['mean texture','mean symmetry']]
y = data_breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

dec_tree_clf = DecisionTreeClassifier()
dec_tree_clf.fit(X_train, y_train)

log_reg_clf = LogisticRegression()
log_reg_clf.fit(X_train, y_train)

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

acc_dec_tree_test = accuracy_score(y_test, dec_tree_clf.predict(X_test))
acc_dec_tree_train = accuracy_score(y_train, dec_tree_clf.predict(X_train))

acc_log_reg_test = accuracy_score(y_test, log_reg_clf.predict(X_test))
acc_log_reg_train = accuracy_score(y_train, log_reg_clf.predict(X_train))

acc_knn_test = accuracy_score(y_test, knn_clf.predict(X_test))
acc_knn_train = accuracy_score(y_train, knn_clf.predict(X_train))

"""### Ensemble dla glosowania typu hard i soft"""

from sklearn.ensemble import VotingClassifier

voting_clf_hard = VotingClassifier(
    estimators=[
        ('dc', dec_tree_clf),
        ('lr', log_reg_clf),
        ('knn', knn_clf),
    ],
    voting='hard'
)

voting_clf_soft = VotingClassifier(
    estimators=[
        ('dc', dec_tree_clf),
        ('lr', log_reg_clf),
        ('knn', knn_clf),
    ],
    voting='soft'
)

voting_clf_hard.fit(X_train, y_train)
voting_clf_soft.fit(X_train, y_train)

acc_voting_hard_test = accuracy_score(y_test, voting_clf_hard.predict(X_test))
acc_voting_hard_train = accuracy_score(y_train, voting_clf_hard.predict(X_train))

acc_voting_soft_test = accuracy_score(y_test, voting_clf_soft.predict(X_test))
acc_voting_soft_train = accuracy_score(y_train, voting_clf_soft.predict(X_train))

"""# Porownanie dokladnosci klasyfikatorow"""

print("Dokladnosc Decision Tree Classifier  train : ", acc_dec_tree_train)
print("Dokladnosc Decision Tree Classifier  test  : ", acc_dec_tree_test)
print()
print("Dokladnosc Logistic Regression       train : ", acc_log_reg_train)
print("Dokladnosc Logistic Regression       test  : ", acc_log_reg_test)
print()
print("Dokladnosc KNeighbors Classifier     train : ", acc_knn_train)
print("Dokladnosc KNeighbors Classifier     test  : ", acc_knn_test)
print()
print("Dokladnosc Ensemble Hard Voting      train : ", acc_voting_hard_train)
print("Dokladnosc Ensemble Hard Voting      test  : ", acc_voting_hard_test)
print()
print("Dokladnosc Ensemble Soft Voting      train : ", acc_voting_soft_train)
print("Dokladnosc Ensemble Soft Voting      test  : ", acc_voting_soft_test)

res_list1 = [(acc_dec_tree_train, acc_dec_tree_test), (acc_log_reg_train, acc_log_reg_test), (acc_knn_train, acc_knn_test), # Poprawiono tutaj
             (acc_voting_hard_train, acc_voting_hard_test), (acc_voting_soft_train, acc_voting_soft_test)]
print(res_list1)

with open('acc_vote.pkl', 'wb') as file:
  pickle.dump(res_list1, file)

res_list2 = [dec_tree_clf, log_reg_clf, knn_clf, voting_clf_hard, voting_clf_soft]
print(res_list2)
with open('vote.pkl', 'wb') as file:
  pickle.dump(res_list2, file)

"""### Bagging i Pasting wykorzystujac 30 drzew decyzyjnych"""

from sklearn.ensemble import BaggingClassifier

# bagging dla max_samples=1.0   = kazdy model trenuje na tyle probek ile wynosi rozmiar calego zbioru danych
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=30, # liczba klasyfikatorow (drzew w tym przypadku)
    max_samples=1.0, # liczba probek do trenowania pojedynczego klasyfikatora
                     # jesli int to liczba probek, jesli float to ile procent probek
    bootstrap=True,  # z powtorzeniami = bagging
    random_state=42
)

bag_clf.fit(X_train, y_train)

# Bagging z wykorzystaniem 50% instancji

bag_clf_50 = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=30,
    max_samples=0.5,
    bootstrap=True,
    random_state=42
)
bag_clf_50.fit(X_train, y_train)

# Pasting dla max_samples = 1.0

pas_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=30,
    max_samples=1.0,
    bootstrap=False, # bez powtorzen = Pasting
    random_state=42
)
pas_clf.fit(X_train, y_train)

# Pasting dla 50% instancji

pas_clf_50 = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=30,
    max_samples=0.5,
    bootstrap=False,
    random_state=42
)
pas_clf_50.fit(X_train, y_train)

"""# Random Forest"""

# Random forest
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(
    n_estimators=30,
    random_state=42
)
rnd_clf.fit(X_train, y_train)

""" # AdaBoost - zwiekszanie wag blednie sklasyfikwoanym probkom"""

from sklearn.ensemble import AdaBoostClassifier

ada_boost_clf = AdaBoostClassifier(
    DecisionTreeClassifier(),
    n_estimators=30,
    random_state=42
)
ada_boost_clf.fit(X_train, y_train)

"""# Gradient Boosting - korygowanie poprzednich predykcji"""

from sklearn.ensemble import GradientBoostingClassifier

grad_boost_clf = GradientBoostingClassifier( # nie trzeba podawac ze DecisionTreeClassifier
    n_estimators=30,
    random_state=42
)
grad_boost_clf.fit(X_train, y_train)

"""# Accuracy dla tych 7 klasyfikotrow"""

bag_acc_train = accuracy_score(y_train, bag_clf.predict(X_train))
bag_acc_test = accuracy_score(y_test, bag_clf.predict(X_test))

bag_50_acc_train = accuracy_score(y_train, bag_clf_50.predict(X_train))
bag_50_acc_test = accuracy_score(y_test, bag_clf_50.predict(X_test))

pas_acc_train = accuracy_score(y_train, pas_clf.predict(X_train))
pas_acc_test = accuracy_score(y_test, pas_clf.predict(X_test))

pas_50_acc_train = accuracy_score(y_train, pas_clf_50.predict(X_train))
pas_50_acc_test = accuracy_score(y_test, pas_clf_50.predict(X_test))

ran_for_acc_train = accuracy_score(y_train, rnd_clf.predict(X_train))
ran_for_acc_test = accuracy_score(y_test, rnd_clf.predict(X_test))

ada_boost_acc_train = accuracy_score(y_train, ada_boost_clf.predict(X_train))
ada_boost_acc_test = accuracy_score(y_test, ada_boost_clf.predict(X_test))

grad_boost_acc_train = accuracy_score(y_train, grad_boost_clf.predict(X_train))
grad_boost_acc_test = accuracy_score(y_test, grad_boost_clf.predict(X_test))

print(f"Bagging Classifier:             Train Accuracy = {bag_acc_train}, \n                                Test Accuracy = {bag_acc_test}\n")
print(f"Bagging Classifier 50% :        Train Accuracy = {bag_50_acc_train}, \n                                Test Accuracy = {bag_50_acc_test}\n")
print(f"Pasting Classifier:             Train Accuracy = {pas_acc_train}, \n                                Test Accuracy = {pas_acc_test}\n")
print(f"Pasting Classifier 50% :        Train Accuracy = {pas_50_acc_train}, \n                                Test Accuracy = {pas_50_acc_test}\n")
print(f"Random Forest Classifier:       Train Accuracy = {ran_for_acc_train}, \n                                Test Accuracy = {ran_for_acc_test}\n")
print(f"AdaBoost Classifier:            Train Accuracy = {ada_boost_acc_train}, \n                                Test Accuracy = {ada_boost_acc_test}\n")
print(f"Gradient Boosting Classifier:   Train Accuracy = {grad_boost_acc_train}, \n                                Test Accuracy = {grad_boost_acc_test}\n")

"""# Dlaczego Random Forest daje inne rezultaty niż Bagging + drzewa decyzyjne?
Poniewaz w Random Forest oprocz boostrapowania przy kazdej probie podzialu w drzewie wybierany jest losowa podgrupa cech co zmniejsza korelacje miedzy drzewami skutkujac lepsza wydajnoscia modelu
"""

accuracies = [
    (bag_acc_train, bag_acc_test),
    (bag_50_acc_train, bag_50_acc_test),
    (pas_acc_train, pas_acc_test),
    (pas_50_acc_train, pas_50_acc_test),
    (ran_for_acc_train, ran_for_acc_test),
    (ada_boost_acc_train, ada_boost_acc_test),
    (grad_boost_acc_train, grad_boost_acc_test)
]

with open('acc_bag.pkl', 'wb') as file:
  pickle.dump(accuracies, file)

classifiers = [bag_clf, bag_clf_50, pas_clf, pas_clf_50, rnd_clf, ada_boost_clf, grad_boost_clf]

with open('bag.pkl', 'wb') as file:
  pickle.dump(classifiers, file)

"""# Sampling 2 sech"""

bag_sampling_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=30,
    max_samples=0.5,
    max_features=2,
    bootstrap_features=False, # cechy bez powtorzen
    bootstrap=True,         # instancje z powtorzeniami
    random_state=42
)
bag_sampling_clf.fit(X_train, y_train)

sampling_acc_train = accuracy_score(y_train, bag_sampling_clf.predict(X_train))
sampling_acc_test = accuracy_score(y_test, bag_sampling_clf.predict(X_test))

print(f"Bagging with sampling:          Train Accuracy = {sampling_acc_train}, \n                                Test Accuracy = {sampling_acc_test}\n")

sampl_list = [sampling_acc_train, sampling_acc_test]

with open('acc_fea.pkl', 'wb') as file:
  pickle.dump(sampl_list, file)

sampl_clf_list = [bag_sampling_clf]

with open('fea.pkl', 'wb') as file:
  pickle.dump(sampl_clf_list, file)

"""# Ranking estymatorow"""

# print(data_breast_cancer.feature_names)

feature_names = X_train.columns.tolist()
print(feature_names)

estimator_data = []

for i, (estimator, feature_indices) in enumerate(zip(bag_sampling_clf.estimators_, bag_sampling_clf.estimators_features_)):
  current_feature_names = [feature_names[idx] for idx in feature_indices]

  X_train_subset = X_train.iloc[:, feature_indices]
  X_test_subset = X_test.iloc[:, feature_indices]

  acc_train = accuracy_score(y_train, estimator.predict(X_train_subset))
  acc_test = accuracy_score(y_test, estimator.predict(X_test_subset))

  estimator_data.append({
      'accuracy train' : acc_train,
      'accuracy test' : acc_test,
      'features' : current_feature_names
  })

rank_df = pd.DataFrame(estimator_data)

rank_df_sorted = rank_df.sort_values(by=['accuracy test', 'accuracy train'], ascending=[False, False])

print(rank_df_sorted.head())

with open('acc_fea_rank.pkl', 'wb') as file:
  pickle.dump(rank_df_sorted, file)