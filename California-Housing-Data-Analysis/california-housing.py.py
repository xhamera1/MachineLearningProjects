# -*- coding: utf-8 -*-
"""lab01.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wi_i1fgddBy4SJXNGJxF1gEQLpJV1tHd
"""

# prompt: !pip install scikit-learn==1.5.0 tensorflow==2.14.0 numpy==1.26.0 pandas==2.2.2 matplotlib==3.9.0 seaborn==0.13.2

# !pip install scikit-learn==1.5.0 tensorflow==2.14.0 numpy==1.26.0 pandas==2.2.2 matplotlib==3.9.0 seaborn==0.13.2

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('housing.csv.gz')
# df to struktura dataframe przypominajaca tabele

df.head() # pierwsze 5 wuerszy
df.info() # informacje o typach kolumn

pd.value_counts(df['ocean_proximity']) # podaje szczegoly kolumny ocean_proximity
# ile razy wystepouje dana wartosc

df['ocean_proximity'].describe() # podaje szczeogoly wartosci kolumny

df.hist(bins=50, figsize=(20,15))
# df.hist generuje histogramy - wykresy pokazujace ilosc wartosxi
# bins - okresla liczbe przedzialow- w histogramie
# figsize - ustawia rozmiar rusunku w calach (szerokosc, wysokosc)
plt.savefig("obraz1.png")

df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1, figsize=(7,4))
# funkcja plot opiera sie na matplotlib(rysownaie wykresow), kind=scatter-punktowy
# alpha - przezroczusyosc punktow 10%
# figsize - wielkosc w calach
plt.savefig("obraz2.png")

df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, figsize=(7,3),
        colorbar=True, s=df["population"]/100, label = "population",
        c="median_house_value",cmap=plt.get_cmap("jet"))
# colorbar - wyswietla pasek barw obok
# s=df["population"]/100 - ustala rozmiar punktow na wykrsie dzieli przez 100
#     zeby puntky dopasowaly sie odpowiednio do wykresu
# label - ustawia etykeite
# c="median_house_value" - przypisuje kolor puntkow w zaleznosci od wartosci mediany warotsic domu
# cmap=plt.get_cmap("jet") - ustawia kolory wedlug mapy kolorow jet
plt.savefig("obraz3.png")

# pd.get_dummies(df, columns=["ocean_proximity"]).corr()["median_house_value"].sort_values(ascending=False)

pd.get_dummies(df, columns=["ocean_proximity"]).corr()["median_house_value"].sort_values(ascending=False).reset_index().rename(columns={"index":"atrybut","median_house_value":"wspolczynnik_korelacji"}).to_csv("korelacja.csv", index=False)
# get_dummies() - kolumna ocean_proximity jest nienumeryczna wicc ta metoda zamienia wartosci na numeryczne
# corr().["median_house_value"] - oblicza macierz korealcji dla wszystkich kolumn,
#   wybieramy median_house_value co poakzuje wspolcznmnik korelacji z ta kolumna
# sort_values(ascending=False) - sortuje m,alejaco
# .reset_index() - przekstalcza ta serie w dataFrame,
#   nazwy atrybutow ktore byly indeksami  sa kolumna
# rename zamienia nazwy kolumn
# to_csv - zapisuje dataframe jako plik csv, index=False mowi ze dodatkowa kolumna indeksu nie zostanie zapisana

import seaborn as sns
sns.pairplot(df)

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
# ^ podzial zbioru danychb na treningowy i testowy
# test_size=0.2 oznacza ze 20% danycgh to testowe, 80% to treningowe
# random_state ustala ziarno losowosci
len(train_set), len(test_set)

train_set.head()

test_set.head()

pd.get_dummies(train_set, columns=["ocean_proximity"]).corr()["median_house_value"].sort_values(ascending=False).reset_index().rename(columns={"index":"atrybut","median_house_value":"wspolczynnik_korelacji"})

pd.get_dummies(test_set, columns=["ocean_proximity"]).corr()["median_house_value"].sort_values(ascending=False).reset_index().rename(columns={"index":"atrybut","median_house_value":"wspolczynnik_korelacji"})

# wyniki sa podobne ale oczywiscie nie identyczne
train_set.to_pickle("train_set.pkl")
test_set.to_pickle("test_set.pkl")
# pliki pickle to pliki binarne sluzace do serializacji obiekow w pyghtonie