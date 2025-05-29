# %% [markdown]
# # Strojenie hiperparametrów

# %% [markdown]
# ## Zadanie 1

# %% [markdown]
# ### Pobranie danych

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, 
                                                              housing.target,
                                                              random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)


# %%
scaler = StandardScaler() # skalowanie aby wartosci mialy podobna skale
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)



# %% [markdown]
# ## Zadanie 2
# 

# %% [markdown]
# ### Przeszukiwanie przestrzeni hiperparametrów przy pomocy scikit-learn
# 

# %% [markdown]
# Celem ćwiczenia jest przejrzenie przestrzeni parametrów w następujących zakresach:
# 1. krok uczenia: [3 ⋅ 10−4, 3 ⋅ 10−2],
# 2. liczba warstw ukrytych: od 0 do 3,
# 3. liczba neuronów na warstwę: od 1 do 100,
# 4. algorytm optymalizacji: adam, sgd lub nesterov.

# %%
import numpy as np
from scipy.stats import reciprocal

param_distribs = {
    "model__n_hidden" : [0,1,2,3],
    "model__n_neurons" : np.arange(1, 101),
    "model__learning_rate" : reciprocal(3e-4, 3e-2),   # losowa wartosc z tego zakresu - rozklad odwrotnu
    "model__optimizer" : ["adam", "sgd", "nesterov"]
}



# %%

def build_model(n_hidden, n_neurons, optimizer, learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=X_train.shape[1:]))
    for layer in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    optimizer_instance = None
    optimizer_str_lower = optimizer.lower() 
    if optimizer_str_lower == 'adam':
        optimizer_instance = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_str_lower == 'sgd':
        optimizer_instance = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_str_lower == 'momentum':
        optimizer_instance = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, 
            momentum=0.9
        )
    elif optimizer_str_lower == 'nesterov':
        optimizer_instance = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, 
            nesterov=True,
            momentum=0.9)
    model.compile(loss="mse", optimizer=optimizer_instance)
    return model


# %%
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV



learning_rates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

h = []

for learning_rate in learning_rates:
    model = build_model(n_hidden=2, n_neurons=15, optimizer="adam", learning_rate=learning_rate)
    history = model.fit(X_train, y_train, epochs=40,
                        validation_data=(X_valid, y_valid)) # walidacyjny jest by ocenic postepy w trakcie uczenia
    h.append(history)


# %%
import matplotlib.pyplot as plt

for i, h_i in enumerate(h):
    plt.plot(h_i.history['loss'], label=f"Training loss 1e-{i+1}")
plt.legend()

# %%
import scikeras
from scikeras.wrappers import KerasRegressor

es = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=1.0, verbose=1)  # verbose 1 to ze beda komunikaty ze sie zatrzymal

keras_reg_model = KerasRegressor(build_model, callbacks=[es])

# %%
from sklearn.model_selection import RandomizedSearchCV

rnd_search_cv = RandomizedSearchCV(estimator=keras_reg_model, # model do dostrojeniaerror_score=
                                   param_distributions=param_distribs, # slownik hiperparametrow
                                   n_iter=5, # liczba kombinacji hiperparametrow ktore zostana sprawdzone
                                   cv=3, 
# ccv = 3 - cross-validation na 3 czesci, tzn raz trenowany na 1 i 2 czesci a testowany na 3, potem 1 i 3 i test na 2 i potem 2 i 3 i test na 1
                                   verbose=2) # bardziej szczegolowe logi

# %%
rnd_search_cv.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), verbose=0)

# %%
import pickle

print(f"Best estimator: {rnd_search_cv.best_estimator_}")

print(f"Best score: {rnd_search_cv.best_score_}") # najlepszy sredni wynik (score)

best_params_dict = rnd_search_cv.best_params_

print(best_params_dict)


with open('rnd_search_params.pkl', 'wb') as file:
    pickle.dump(best_params_dict, file)

# %%
with open('rnd_search_scikeras.pkl', 'wb') as file:
    pickle.dump(rnd_search_cv, file)

# %% [markdown]
# ## Zadanie 2.3 Przeszukiwanie przestrzeni hiperparametrów przy pomocy Keras Tuner

# %%
import keras_tuner as kt

# %% [markdown]
# 1. krok uczenia: [3 ⋅ 10−4, 3 ⋅ 10−2],
# 2. liczba warstw ukrytych: od 0 do 3,
# 3. liczba neuronów na warstwę: od 1 do 100,
# 4. algorytm optymalizacji: adam, sgd lub nesterov.
# 

# %%
def build_model_kt(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=3, default=2)
    n_neurons = hp.Int("n_neurons", min_value=1, max_value=100)
    learning_rate = hp.Float("learning_rate", min_value=3e-4, max_value=3e-2, sampling="log")
    optimizer = hp.Choice("optimizer", values=["sgd", "adam", "nesterov"])
    if optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == "nesterov":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, nesterov=True)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation=None)) # no bo to regresja

    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mse", 'mae'])
    return model

    


# %%
random_search_tuner = kt.RandomSearch(
    build_model_kt, objective="val_mse", max_trials=10,
    overwrite=True, directory="my_california_housing",
    project_name="my_rnd_search", seed=42
)

# %%
import os

root_logdir = os.path.join(random_search_tuner.project_dir, 'tensorboard')
tb = tf.keras.callbacks.TensorBoard(root_logdir)

# %%
random_search_tuner.search(X_train, y_train, epochs=100,
                           validation_data=(X_valid, y_valid),
                           callbacks=[es, tb])

# %%
best_hyperparameters = random_search_tuner.get_best_hyperparameters(num_trials=1)
best_hps = best_hyperparameters[0]

n_hidden = best_hps.get('n_hidden')
n_neurons = best_hps.get('n_neurons')
learning_rate = best_hps.get('learning_rate')
optimizer = best_hps.get('optimizer')


print(f'Best n_hidden = {n_hidden}')
print(f'Best n_nuerons = {n_neurons}')
print(f'Best learing_rate = {learning_rate}')
print(f'Best optimizer = {optimizer}')

best_params_tuner_dict = {
    'n_hidden' : n_hidden,
    'n_neurons' : n_neurons,
    'learning_rate' : learning_rate,
    'optimizer' : optimizer
}

print(best_params_tuner_dict)



# %%
with open('kt_search_params.pkl', 'wb') as file:
    pickle.dump(best_params_tuner_dict, file)

best_model_list = random_search_tuner.get_best_models(num_models=1)
b_model = best_model_list[0]

# with open('kt_best_model.keras', 'wb') as file:
#     pickle.dump(best_model, file)

b_model.save('kt_best_model.keras')
