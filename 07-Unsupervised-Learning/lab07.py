# %% [markdown]
# ## Unsupervised learning

# %% [markdown]
# ### Preparing data

# %%
from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]

print(X)
print(y)

# %% [markdown]
# # K-Means

# %% [markdown]
# ### Silhouette score (wskaznik sylwetkowy) 
# - mierzy jak dobrze dany punkt pasuje do klastra  
#   s = (b-a)/(max(a,b))      wartosc (-1,1)
# im blizej 1 tym lepszy, im blizej -1 tym gorszy
# a - srednia odlegosc do wszytskch punktow W TYM SAMYM klastrze
#       (im mniej tym lepiej)
# b - srednia odleglosc od wszystkich punktow W NAJBLIZSZYM SASIEDNIM klastrze (im wiecej tym lepiej)
# 

# %%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

k_number = [8,9,10,11,12]

silh_list = []


for k in k_number:
    kmeans = KMeans(n_init=10, n_clusters=k, random_state=42)
    y_pred = kmeans.fit_predict(X)
    silhouette_sc = silhouette_score(X, y_pred)
    print("Silhouette score for k = ", k, " : ", silhouette_sc)
    silh_list.append(silhouette_sc)

# %% [markdown]
# ### Mimo, ze wiadomo, ze jest 10 klastrow to KMeans nie dal najlepszego silhouete score dla k=10, lecz niewiele lepsze (tez bardzo male) dla k=8

# %%
import pickle

print(silh_list)

with open('kmeans_sil.pkl', 'wb') as file:
    pickle.dump(silh_list, file)

# %%
from sklearn.metrics import confusion_matrix

kmeans10 = KMeans(n_init=10, n_clusters=10, random_state=42)
kmeans10.fit_predict(X)

conf_matrix = confusion_matrix(y, kmeans10.predict(X))

# %%
max_ind_list = []

for row in conf_matrix:
    i = np.argmax(row)
    max_ind_list.append(i)

max_ind_list = np.unique(max_ind_list)
max_ind_list.sort()


print(max_ind_list)


# %% [markdown]
# lista: [0 1 2 3 5 6 8 9]
# pokazuje ze klastry K-Means o indeksach 4 i 7 nie byly glownymi reprezentantami dla zadnej z 10 prawdziwych cyfr
# 

# %%
with open('kmeans_argmax.pkl', 'wb') as file:
    pickle.dump(max_ind_list, file)

# %% [markdown]
# # DBSCAN

# %%
from sklearn.cluster import DBSCAN

n_subset = 300
all_distances = []

for i in range(n_subset):
    x1 = X[i]

    for j in range(X.shape[0]):
        x2 = X[j]
        dist = np.linalg.norm(x1 - x2)
        if dist < 1e-11:
            continue
        all_distances.append(dist)

all_distances.sort()

# %%
all_distances_10 = all_distances[:10]
print(all_distances_10)

with open('dist.pkl', 'wb') as file:
    pickle.dump(all_distances_10, file)

# %%
s = (all_distances_10[0] + all_distances_10[1] + all_distances_10[2])/3.0
print(s)


# %%
eps_list = []

# for eps in range(s, s+0.1*s, 0.04*s):
#     eps_list.append(eps)

i = s
stop = s+0.1*s
step = 0.04*s

while (i <= stop):
    eps_list.append(i)
    i += step

print(eps_list)

# %%
label_count_list = []

for eps in eps_list:
    db_scan = DBSCAN(eps=eps, n_jobs=-1)
    db_scan.fit(X)
    cluster_unique = np.unique(db_scan.labels_)
    label_count_list.append(len(cluster_unique))
    print("For eps=", eps, " label count : ", len(cluster_unique))
    print() 





# %%
with open('dbscan_len.pkl', 'wb') as file:
    pickle.dump(label_count_list, file)


