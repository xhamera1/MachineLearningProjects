# %% [markdown]
# # Neural Networks 

# %% [markdown]
# # Classification

# %% [markdown]
# ## Getting Data

# %% [markdown]
# Dataset Fashion MNIST

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# %%
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)


# %% [markdown]
# Scaling values

# %%
X_train = X_train/255.0
X_test = X_test/255.0


# %%
plt.figure(figsize=(2,2))
plt.imshow(X_train[42], cmap="binary")
plt.axis('off')
plt.show()

# %%
class_names = ["koszulka", "spodnie", "pulower", 
               "sukienka", "kurtka","sandał",
                "koszula", "półbut", "torba", "but"]
print(class_names[y_train[42]])
# but

# %% [markdown]
# ## Creating neural network model

# %%
import keras

model = keras.models.Sequential()

model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

# %%
model.summary()

# %%
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

# %%
import os
root_logdir = os.path.join(os.curdir, "image_logs")
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()


# %%
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

history = model.fit(X_train, y_train, epochs=20,
                    validation_split=0.1,
                    callbacks=[tensorboard_cb])


# %% [markdown]
# ### Some random prediction

# %%
image_index = np.random.randint(len(X_test)) # losowy indeks
image = np.array([X_test[image_index]]) # losowy obraz
confidences = model.predict(image) # prawdopodobienstwa
# print(confidences)
confidence = np.max(confidences[0])
prediction = np.argmax(confidences[0])
print("Prediction: ", class_names[prediction])
print("Confidence: ", confidence)
print("Truth: ", class_names[y_test[image_index]])
plt.figure(figsize=(2,2))
plt.imshow(image[0], cmap="binary")
plt.axis('off')
plt.show()


# %%
# %load_ext tensorboard
# %tensorboard --logdir=./image_logs

# %%
model.save('fashion_clf.keras')

# %% [markdown]
# # Regression

# %% [markdown]
# ### Getting data

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing()

# %%
X = housing['data']
y = housing['target']
print(housing['feature_names'])
print(housing['target_names'])

# %%
# 80% TRAIN_AND_VALID, 20% TEST
# 90% TRAIN, 10% VALID
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.1)

# %%
# warstwa normalizujaca

normalizer = keras.layers.Normalization(
    input_shape=[X_train.shape[1]])

normalizer.adapt(X_train)

# %%
model_reg = keras.models.Sequential()

model_reg.add(normalizer)
model_reg.add(keras.layers.Dense(50, activation="relu"))
model_reg.add(keras.layers.Dense(50, activation="relu"))
model_reg.add(keras.layers.Dense(50, activation="relu"))
model_reg.add(keras.layers.Dense(1))



# %%
model_reg.compile(loss="mean_squared_error",
              optimizer="adam",
              metrics=[keras.metrics.RootMeanSquaredError()])

# %%
model_reg.summary()

# %% [markdown]
# ### Early stopping

# %%
es = tf.keras.callbacks.EarlyStopping(patience=5,
                                      min_delta=0.01,
                                      verbose=1)

# %%
root_logdir = os.path.join(os.curdir, "housing_logs")
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()


# %% [markdown]
# ### Model learing, we can apply big epochs size because early stopping will stop anyway

# %%
tensorboard_cb_reg = tf.keras.callbacks.TensorBoard(run_logdir)
history = model_reg.fit(X_train, y_train, epochs=100,   
                        validation_data=(X_valid, y_valid),
                        callbacks=[tensorboard_cb_reg, es])

# %%
model_reg.save('reg_housing_1.keras')

# %%
test_loss, test_rmse = model_reg.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

# %% [markdown]
# ### 2 more regression models 

# %% [markdown]
# ### First one:

# %%
model_reg1 = keras.models.Sequential()
model_reg1.add(normalizer)
model_reg1.add(keras.layers.Dense(units=10, activation="relu"))
model_reg1.add(keras.layers.Dense(units=20, activation="relu"))
model_reg1.add(keras.layers.Dense(units=30, activation="relu"))
model_reg1.add(keras.layers.Dense(units=40, activation="relu"))
model_reg1.add(keras.layers.Dense(units=50, activation="relu"))
model_reg1.add(keras.layers.Dense(1))

# %%
model_reg1.compile(optimizer='adam', 
                   metrics=[keras.metrics.RootMeanSquaredError()],
                   loss='mean_squared_error')

# %%
history1 = model_reg1.fit(X_train, y_train,
                          epochs=100, validation_data=(X_valid, y_valid),
                          callbacks=[tensorboard_cb_reg, es])

# %%
test_loss, test_rmse = model_reg1.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

# %%
model_reg1.save('reg_housing_2.keras')

# %% [markdown]
# ### Second one:

# %%
model_reg2 = keras.models.Sequential()

model_reg2.add(normalizer)
model_reg2.add(keras.layers.Dense(128, activation='selu'))
model_reg2.add(keras.layers.Dense(64, activation='selu'))
model_reg2.add(keras.layers.Dense(32, activation='selu'))
model_reg2.add(keras.layers.Dense(1))

# %%
model_reg2.compile(optimizer='adam', 
                   metrics=[keras.metrics.RootMeanSquaredError()],
                   loss='mean_squared_error')

# %%
model_reg2.fit(X_train, y_train, epochs=100,
               validation_data=(X_valid, y_valid),
               callbacks=[tensorboard_cb_reg, es])

# %%
test_loss, test_rmse = model_reg2.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

# %%
model_reg2.save('reg_housing_3.keras')


