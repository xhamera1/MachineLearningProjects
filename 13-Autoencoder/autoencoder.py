# %% [markdown]
# # Laboratorium: Autoenkodery

# %% [markdown]
# ## Pobranie danych

# %%
import tensorflow as tf
import pickle 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
dataset = tf.keras.datasets.mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = dataset
X_train_full = X_train_full.astype(np.float32) / 255 
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]


# %% [markdown]
# ## 2.1 Autoenkoder z warstwami gęstymi

# %% [markdown]
# ### Głęboki autoenkonder zdolny do reprezentowania obrazów cyfr

# %%
print(X_train[0].shape)

# %%
encoder = tf.keras.Sequential([
    tf.keras.layers.Input([28, 28]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='selu'), # funckja selu ma wlasciwosc: samonormalizacja
    tf.keras.layers.Dense(30, activation='selu')
])



decoder = tf.keras.Sequential([
    tf.keras.layers.Input([30]),
    tf.keras.layers.Dense(100, activation='selu'),
    tf.keras.layers.Dense(28*28, activation='sigmoid'),
    tf.keras.layers.Reshape([28,28])
])

ae = tf.keras.Sequential([encoder, decoder])


# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, )
ae.compile(optimizer=optimizer, loss=tf.keras.losses.Huber() , 
           metrics=['mae']) # mae - mean absolute error



# %%
history = ae.fit(X_train, X_train, epochs=5, 
                 validation_data=(X_valid, X_valid))


# %%
ae.evaluate(X_test, X_test, return_dict=True)

# {'loss': 0.005280734039843082, 'mae': 0.03514503315091133}

# %%
def plot_reconstructions(model, images=X_test, n_images=5):
    reconstructions = np.clip(model.predict(images[:n_images]), 0, 1)
    fig = plt.figure(figsize=(n_images*1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1+image_index)
        plt.imshow(images[image_index], cmap='binary')
        plt.axis('off')
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plt.imshow(reconstructions[image_index], cmap='binary')
        plt.axis('off')

plot_reconstructions(ae)


# %%
ae.save('ae_stacked.keras')


# %% [markdown]
# ## 2.2 Autoenkoder konwolucyjny

# %%
conv_encoder = tf.keras.Sequential([
    tf.keras.layers.Input([28, 28]),
    tf.keras.layers.Reshape([28, 28, 1]),
    tf.keras.layers.Conv2D(16, kernel_size=3, padding='same', 
                           activation='selu'),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', 
                           activation='selu'),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', 
                           activation='selu'),
    tf.keras.layers.MaxPool2D(pool_size=2),
])

conv_decoder = tf.keras.models.Sequential([
    tf.keras.layers.Input([3, 3, 64]),
    tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2,
                                    padding='valid', activation='selu'),
    tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2,
                                    padding='same', activation='selu'),
    tf.keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2,
                                    padding='same', activation='sigmoid'),
    tf.keras.layers.Reshape([28, 28])
])


# %%
conv_ae = tf.keras.Sequential([conv_encoder, conv_decoder])


# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
conv_ae.compile(optimizer=optimizer, 
                loss=tf.keras.losses.Huber(), 
                metrics=['mae'])


# %%
history = conv_ae.fit(X_train, X_train, epochs=5,
                      validation_data=(X_valid, X_valid))


# %%
conv_ae.evaluate(X_test, X_test)
# [0.0016590015729889274, 0.018200533464550972]


# %%
plot_reconstructions(conv_ae)
plt.show()

# %%
conv_ae.save('ae_conv.keras')

# %% [markdown]
# ## 2.3 Wizualizacja wyników

# %%
from sklearn.manifold import TSNE

X_valid_compressed = conv_encoder.predict(X_valid)

n_samples , x, y, z = X_valid_compressed.shape
X_valid_compressed_flat = X_valid_compressed.reshape((n_samples, x*y*z))

tsne = TSNE(init="pca", learning_rate="auto", random_state=42)
X_valid_2D = tsne.fit_transform(X_valid_compressed_flat)

plt.figure(figsize=(10, 5))
plt.scatter(X_valid_2D[:, 0], X_valid_2D[:, 1], c=y_valid, s=10, cmap="tab10")
plt.show()


# %%
import matplotlib as mpl


plt.figure(figsize=(10, 5))
cmap = plt.cm.tab10
Z = X_valid_2D
Z = (Z - Z.min()) / (Z.max() - Z.min()) # normalize to the 0-1 range
plt.scatter(Z[:, 0], Z[:, 1], c=y_valid, s=10, cmap=cmap)
image_positions = np.array([[1., 1.]])
for index, position in enumerate(Z):
    dist = ((position - image_positions) ** 2).sum(axis=1)
    if dist.min() > 0.02: # if far enough from other images
        image_positions = np.r_[image_positions, [position]]
        imagebox = mpl.offsetbox.AnnotationBbox(
            mpl.offsetbox.OffsetImage(X_valid[index], cmap="binary"),
            position, bboxprops={"edgecolor": cmap(y_valid[index]), "lw": 2})
        plt.gca().add_artist(imagebox)


plt.axis("off")
plt.show()



# %% [markdown]
# ## 2.4 Odszumianie

# %% [markdown]
# ### Gausian Noise

# %%
gausian_noise_encoder = tf.keras.Sequential([
    tf.keras.layers.Input([28,28]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.GaussianNoise(stddev=0.15),
    tf.keras.layers.Dense(100, activation='selu'),
    tf.keras.layers.Dense(30, activation='selu')
])

gausian_noise_decoder = tf.keras.Sequential([
    tf.keras.layers.Input([30]),
    tf.keras.layers.Dense(100, activation='selu'),
    tf.keras.layers.Dense(28*28, activation='sigmoid'),
    tf.keras.layers.Reshape([28,28])
])

gae = tf.keras.Sequential([gausian_noise_encoder, gausian_noise_decoder])


# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
gae.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])


# %%
history = gae.fit(X_train, X_train, epochs=5, validation_data=(X_valid, X_valid))


# %%
gae.evaluate(X_test, X_test)
# [0.005728995893150568, 0.03739464282989502]


# %%
gausian_noise = tf.keras.layers.GaussianNoise(stddev=0.15)
plot_reconstructions(gae, gausian_noise(X_valid, training=True))


# %% [markdown]
# ### Dropout

# %%
dropout_encoder = tf.keras.models.Sequential([
    tf.keras.layers.Input([28,28]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(100, activation='selu'),
    tf.keras.layers.Dense(30, activation='selu'),
])

dropout_decoder = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation='selu'),
    tf.keras.layers.Dense(28*28, activation='sigmoid'),
    tf.keras.layers.Reshape([28,28])
])

dae = tf.keras.models.Sequential([dropout_encoder, dropout_decoder])


# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
dae.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer,
            metrics=['mae'])


# %%
history = dae.fit(X_train, X_train, epochs=5, 
        validation_data=(X_valid, X_valid))

# %%
dae.evaluate(X_test, X_test)
# [0.006634000223129988, 0.043030720204114914]

# %%
dropout = tf.keras.layers.Dropout(0.5)
plot_reconstructions(dae, dropout(X_valid, training=True))


# %%
dae.save('ae_denoise.keras')


