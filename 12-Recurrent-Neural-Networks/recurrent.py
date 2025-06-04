# %% [markdown]
# # Laboratorium: Rekurencyjne sieci neuronowe
# 

# %% [markdown]
# ## 2.1 Pobieranie danych

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# %%
tf.keras.utils.get_file(
    "bike_sharing_dataset.zip",
    "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
    cache_dir=".",
    extract=True
)


# %% [markdown]
# ## 2.2 Przygotowanie danych

# %%
df = pd.read_csv('datasets/bike_sharing_dataset_extracted/hour.csv')
df['datetime'] = pd.to_datetime(df['dteday'] + ' ' + df['hr'].astype(str).str.zfill(2), format = '%Y-%m-%d %H')
df.set_index('datetime', inplace=True)



# %%
print(df[:1])

# %%
print((df.index.min(), df.index.max()))


# %%
print('dataset rows count: ', len(df))
print('should be: ', (365 + 366) * 24)
print('missing: ', (365 + 366) * 24 - len(df), ' rows')


# %% [markdown]
# Okazuje się, że w zbiorze brakuje rekordów dla okresów (godzin) podczas których nikt nie korzystał z
# rowerów. Aby szeregi czasowe były regularne, trzeba je uzupełnić. Przy okazji pozbędziemy się
# niepotrzebnych kolumn

# %%
full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')

df_resampled = df.reindex(full_index)

# dla kolumn przechowujących zarejestrowane liczby wypożyczeń 
# (casual, registered, cnt), wypełnij brakujące wiersze zerami,
cols_to_fill_zero = ['casual', 'registered', 'cnt']
df_resampled[cols_to_fill_zero] = df_resampled[cols_to_fill_zero].fillna(0)

# dla kolumn przechowujących sensoryczne dane pogodowe 
# (temp, atemp, hum, windspeed), zastosuj interpolację
cols_to_interpolate = ['temp', 'atemp', 'hum', 'windspeed']
df_resampled[cols_to_interpolate] = df_resampled[cols_to_interpolate].interpolate(method='time')


# dla kolumn kategoryzowanych (holiday, weekday, workingday, weathersit),
#  wypełnij brakujące wartości z poprzedniego rekordu.
cols_to_fill_prev = ['holiday', 'weekday', 'workingday', 'weathersit']
# forward fill - ffill - Fill NA/NaN values by propagating the last valid observation to next valid.
df_resampled[cols_to_fill_prev] = df_resampled[cols_to_fill_prev].fillna(method='ffill')

df = df_resampled


# %% [markdown]
# ### Sprawdzenie czy dataframe ma odpowiednia strukture

# %%
print(df.notna().sum())


# %%
print(df[['casual', 'registered', 'cnt', 'weathersit']].describe())


# %%
df.casual = df.casual / 1e3
df.registered = df.registered / 1e3
df.cnt = df.cnt / 1e3
df.weathersit = df.weathersit / 4

print(df[['casual', 'registered', 'cnt', 'weathersit']].describe())

# %%
df_2weeks = df[:24*7*2]
df_2weeks[['casual', 'registered', 'cnt', 'temp']].plot(figsize=(10,3))


# %%
print(df.dtypes)

# %%
df['dteday'] = pd.to_datetime(df['dteday'])
df_daily = df.resample('W').mean()
df_daily[['casual', 'registered', 'cnt', 'temp']].plot(figsize=(10, 3))



# %% [markdown]
# ## 2.3 Wskaźniki bazowe - uzywajac sredniego bledu bezwzglednego 

# %%
# dla doby:
df['cnt_pred_daily'] = df['cnt'].shift(24) # 24 godziny wczesniej
mae_daily = np.abs(df['cnt'].dropna() * 1000 - df['cnt_pred_daily'].dropna() * 1000).mean()
print(mae_daily)

# dla tygodnia
df['cnt_pred_weekly'] = df['cnt'].shift(24*7) # 7 dni wczesniej
mae_weekly = np.abs(df['cnt'].dropna() * 1000 - df['cnt_pred_weekly'].dropna() * 1000).mean()
print(mae_weekly)

mae_baseline = (mae_daily, mae_weekly)

import pickle

with open('mae_baseline.pkl', 'wb') as file:
    pickle.dump(mae_baseline, file)


# %% [markdown]
# ## 2.4 Predykcja przy pomocy sieci gęstej

# %% [markdown]
# proporcja: 18 miesiecy treningowy, 6 miesiecy walidacyjny

# %%
cnt_train = df['cnt']['2011-01-01 00:00' : '2012-06-30 23:00']
cnt_valid = df['cnt']['2012-07-01 00:00':]

# %%
seq_len = 1 * 24

# timeseries_dataset_from_array() - tworzy okna dancyh wejsciowych z etykietami
# seq_len  - definiuje dlugosc sekwencji wejsciowej (wielkosc okna) na 24 rekordy czyli jedna doba

# chodzi o to ze na podstawie 24 godizn model ma przewidziec kolejna wartosc jaka bedzie w 25 godzinie jakby

train_ds = tf.keras.utils.timeseries_dataset_from_array(
    cnt_train.to_numpy(),               # dane wejsciowe jako tablica numpy
    targets=cnt_train[seq_len:],        # wartosci docelowe ktore model ma przewidziec
    sequence_length=seq_len,            # kazda probka bedzie miala dlugosc 24
    batch_size=32,                      # dane beda pogrupowane po 32 paczki
    shuffle=True,                       # przetasowanie
    seed=42                             
)

valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    cnt_valid.to_numpy(),
    targets=cnt_valid[seq_len:],
    sequence_length=seq_len,
    batch_size=32
)



# %%
model = tf.keras.Sequential([
    tf.keras.Input(shape=(seq_len,)),
    tf.keras.layers.Dense(1)
])

# %%
optimizer = tf.keras.optimizers.SGD(momentum=0.9, learning_rate=0.1)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])

# %%
history = model.fit(train_ds, epochs=20, validation_data=valid_ds)

# %%
validation_mae = history.history['val_mae'][-1]
scaled_mae = validation_mae * 1000
print(scaled_mae)

# 56.22904747724533 jest lepsze niz bazowy 65.0561797752809


# %%
model.save('model_linear.keras')

mae_linear = (scaled_mae)
with open(' mae_linear.pkl', 'wb') as file:
    pickle.dump(mae_linear, file)



# %% [markdown]
# ## 2.5 Prosta sieć rekurencyjna

# %% [markdown]
# ### Utwórz prostą sieć rekurencyjną, zawierającą jedną warstwę z jednym neuronem

# %%
model = tf.keras.Sequential([
    tf.keras.Input(shape=[None,1]),
    tf.keras.layers.LSTM(1)
])



# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer,
              metrics=['mae'])


# %%
history = model.fit(train_ds, epochs=20, validation_data=valid_ds)

# %%
validation_mae = history.history['val_mae'][-1]
scaled_mae = validation_mae * 1000
print(scaled_mae)
# 106.5203994512558 wiec slabo



# %%
mae_rnn1 = (scaled_mae)

with open('mae_rnn1.pkl', 'wb') as file:
    pickle.dump(mae_rnn1, file)

# %%
model = tf.keras.Sequential([
    tf.keras.Input(shape=[None, 1]),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)
])



# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer,
              metrics=['mae'])


# %%
history = model.fit(train_ds, epochs=20, validation_data=valid_ds)


# %%
validation_mae = history.history['val_mae'][-1]
scaled_mae = validation_mae * 1000
print(scaled_mae)
#  44.81298476457596 wiec znacznie lepiej


# %%
model.save('model_rnn32.keras')

mae_rnn32 = (scaled_mae)

with open('mae_rnn32.pkl', 'wb') as file:
    pickle.dump(mae_rnn32, file)

# %% [markdown]
# ## 2.6 Głęboka RNN
# 

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=[None, 1]),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(1)
])


# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer,
              metrics=['mae'])



# %%
history = model.fit(train_ds, epochs=20, validation_data=valid_ds)


# %%
validation_mae = history.history['val_mae'][-1]
scaled_mae = validation_mae * 1000
print(scaled_mae)

# 36.5825854241848 wiec to jescze lepiej

# %%
model.save('model_rnn_deep.keras')

mae_rnn_deep = (scaled_mae)
with open('mae_rnn_deep.pkl', 'wb') as file:
    pickle.dump(mae_rnn_deep, file)

# %% [markdown]
# ## 2.7 Model wielowymiarowy

# %%
feature_cols = ['cnt', 'weathersit', 'atemp', 'workingday']

X_train_mv_df = df[feature_cols]['2011-01-01 00:00':'2012-06-30 23:00']
X_valid_mv_df = df[feature_cols]['2012-07-01 00:00':]

y_train_mv_series = df['cnt']['2011-01-01 00:00':'2012-06-30 23:00']
y_valid_mv_series = df['cnt']['2012-07-01 00:00':]

seq_len = 1 * 24

# timeseries_dataset_from_array() - tworzy okna dancyh wejsciowych z etykietami
# seq_len  - definiuje dlugosc sekwencji wejsciowej (wielkosc okna) na 24 rekordy czyli jedna doba

# chodzi o to ze na podstawie 24 godizn model ma przewidziec kolejna wartosc jaka bedzie w 25 godzinie jakby

train_ds = tf.keras.utils.timeseries_dataset_from_array(
    X_train_mv_df.to_numpy(),               # dane wejsciowe jako tablica numpy
    targets=y_train_mv_series.to_numpy()[seq_len:],        # wartosci docelowe ktore model ma przewidziec
    sequence_length=seq_len,            # kazda probka bedzie miala dlugosc 24
    batch_size=32,                      # dane beda pogrupowane po 32 paczki
    shuffle=True,                       # przetasowanie
    seed=42                             
)

valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    X_valid_mv_df.to_numpy(),
    targets=y_valid_mv_series.to_numpy()[seq_len:],
    sequence_length=seq_len,
    batch_size=32
)

# %%
model = tf.keras.Sequential([
    tf.keras.Input(shape=[None, 4]),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)
])

# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer,
              metrics=['mae'])



# %%
history = model.fit(train_ds, epochs=20, validation_data=valid_ds)


# %%
validation_mae = history.history['val_mae'][-1]
scaled_mae = validation_mae * 1000
print(scaled_mae)

# 39.15490210056305 - jest dobrze, troche gorzej niz z gleboka z poprzendiego zadania

# %%
model.save('model_rnn_mv.keras')

mae_rnn_mv = (scaled_mae)
with open('mae_rnn_mv.pkl', 'wb') as file:
    pickle.dump(mae_rnn_mv, file)


