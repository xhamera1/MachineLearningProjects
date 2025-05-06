# %% [markdown]
# # Dimension Reduction

# %% [markdown]
# ## Preparing data

# %%
from sklearn import datasets
from sklearn.datasets import load_iris

data_breast_cancer = datasets.load_breast_cancer()
data_iris = load_iris()

X_bc = data_breast_cancer.data
X_iris = data_iris.data

# %% [markdown]
# ## PCA (Principal Component Analysis)

# %% [markdown]
# ### Without scaling

# %% [markdown]
# 1. Przeprowadź analizę PCA, tak aby tak zredukować liczbę wymiarów dla każdego z w/w
# zbiorów. Nowa przestrzeń ma pokrywać przynajmniej 90% różnorodności (zmienności) danych
# i ma mieć jak najmniej wymiarów.

# %%
from sklearn.decomposition import PCA

pca_bc = PCA(n_components=0.9)
X_bc_90 = pca_bc.fit_transform(X_bc)
print("Before scaling: ")
print('Breast cancer:')
print(X_bc.shape,'-->', X_bc_90.shape)
print(pca_bc.explained_variance_ratio_)

print()

pca_iris = PCA(n_components=0.90)
X_iris_90 = pca_iris.fit_transform(X_iris)
print('Iris:')
print(X_iris.shape,'-->', X_iris_90.shape)
print(pca_iris.explained_variance_ratio_)

# %% [markdown]
# 2. Ćwiczenia przeprowadź najpierw na oryginalnych danych, a później na danych
# przeskalowanych. Porównaj wyniki.
# W podanych zbiorach są istotnie różne zakresy dla poszczególnych cech. Aby je przeskalować,
# by były porównywalne, użyj StandardScaler(). Klasa PCA() centruje dane automatycznie,
# ale ich nie skaluje!
# 

# %% [markdown]
# ### With scaling

# %%
from sklearn.preprocessing import StandardScaler

scaler_bc = StandardScaler()
scaler_iris = StandardScaler()

X_bc_scaled = scaler_bc.fit_transform(X_bc)
X_iris_scaled = scaler_iris.fit_transform(X_iris)

pca_bc_sc = PCA(n_components=0.90)
X_bc_90_scaled = pca_bc_sc.fit_transform(X_bc_scaled)

pca_iris_sc = PCA(n_components=0.9)
X_iris_90_scaled = pca_iris_sc.fit_transform(X_iris_scaled)

print("After scaling: ")
print('Breast cancer:')
print(X_bc_scaled.shape,'-->', X_bc_90_scaled.shape)
print(pca_bc_sc.explained_variance_ratio_)

print()

print('Iris:')
print(X_iris_scaled.shape,'-->', X_iris_90_scaled.shape)
print(pca_iris_sc.explained_variance_ratio_)


# %% [markdown]
# ## Saving results

# %% [markdown]
# 3. Utwórz listę z współczynnikami zmienności nowych wymiarów (dla danych przeskalowanych).
# W przypadku data_breast_cancer listę zapisz w pliku Pickle o nazwie pca_bc.pkl
# 3 pkt.
# W przypadku data_iris listę zapisz w pliku Pickle o nazwie pca_ir.pkl
# 3 pkt.

# %%
import pickle

bc_scaled_evr = pca_bc_sc.explained_variance_ratio_
print(bc_scaled_evr)

iris_scaled_evr = pca_iris_sc.explained_variance_ratio_
print(iris_scaled_evr)

with open('pca_bc.pkl', 'wb') as file:
    pickle.dump(bc_scaled_evr, file)

with open('pca_ir.pkl', 'wb') as file:
    pickle.dump(iris_scaled_evr, file)



# %% [markdown]
# 4. Dla danych przeskalowanych utwórz listę indeksów cech oryginalnych wymiarów, w kolejności
# od cechy, która ma największy udział w nowych cechach, do tej, która ma najmniejszy.
# Podpowiedź: zob. atrybut components_ klasy PCA, użyj wartości w
# explained_variance_ratio_ jako wagę istotności udziałów starych cech w nowych
# cechach, czyli pomnóż components_ przez explained_variance_ratio. W otrzymanej
# macierzy oblicz wartość bezwzględne dla każdej wartości (bo mogą być zarówno dodatnie jak i ujemne). Im większa wartość tym większy udział starego wymiaru w nowym.
# Posortuj wartości rosnąco, znajdź odpowiadające im indeksy starych cech i zapisz jako listę
# bez powtórzeń. Przydatne funkcje: numpy.argsort(), ndarray.flatten(). Usunięcie
# powtórzeń z listy możesz zrealizować np. tak: list(dict.fromkeys([20,19,5,20,6])).

# %% [markdown]
# # Breast Cancer Dataset:

# %%
import numpy as np

weights = np.abs(pca_bc_sc.components_ * pca_bc_sc.explained_variance_ratio_.reshape(-1,1))
# print(weights)
ind = np.argsort(np.max(weights, axis=0))[::-1]
print(ind)

# %% [markdown]
# # Iris dataset:

# %%
weights2 = np.abs(pca_iris_sc.components_ * pca_iris_sc.explained_variance_ratio_.reshape(-1,1))
ind2 = np.argsort(np.max(weights2, axis=0))[::-1]
print(ind2)

# %%
with open('idx_bc.pkl', 'wb') as file:
    pickle.dump(ind, file)

with open('idx_ir.pkl', 'wb') as file:
    pickle.dump(ind2, file)


