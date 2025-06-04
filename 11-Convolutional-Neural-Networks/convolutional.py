# %% [markdown]
# # Analiza obrazów przy pomocy sieci konwolucyjnych

# %% [markdown]
# ## 2.1 Ładowanie danych

# %%
import tensorflow_datasets as tfds

[test_set_raw, valid_set_raw, train_set_raw], info = tfds.load(
    "tf_flowers",
    split=['train[:10%]', 'train[10%:25%]', 'train[25%:]'], # odpowiedzi podizal zbioru
    as_supervised = True, # zwraane obiektty maja postac krotek zawieracych zarowno cechy jak i etykiety
    with_info = True # dodaje drugi element zwracanej krotki
)



# %%
class_names = info.features["label"].names
print('Class names: ',class_names)

n_classes = info.features['label'].num_classes
print('Num classes: ', n_classes)

dataset_size = info.splits['train'].num_examples
print('Dataset size: ', dataset_size)



# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import keras


# %%
plt.figure(figsize=(12,8))
index = 0

sample_images = train_set_raw.take(9)


for image, label in sample_images:
    index += 1
    plt.subplot(3,3,index)
    plt.imshow(image)
    plt.title("Class: {}".format(class_names[label]))
    plt.axis('off')

plt.show(block=False)

# %% [markdown]
# ## 2.2 Budujemy prostą sieć CNN

# %%
import tensorflow as tf

def preprocess(image, label):
    resized_image = tf.image.resize(image, [224,224])
    return resized_image, label


# %%
batch_size = 32

train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)


# map - stosuje funckje dla kazdego elementu w zestawie indywidualnie 
# shuffle - mieszka elementy w zestawie 
# batch - grupuje kolejne elementy w paczki (batches) o okreslonym rozmiarze 
# prefetch - pozwala na asynchroniczne przygotowanie danhych w tle,
#            gdy proces trenuje bierzaca to nastepna jest przygotowana 
#            1 okresla ze jedna paczke przygotwuje do przodu


# %%
plt.figure(figsize=(8,8))
sample_batch = train_set.take(1)
print(sample_batch)


for X_batch, y_batch in sample_batch:
    for index in range(12):
        plt.subplot(3, 4, index+1)
        plt.imshow(X_batch[index]/255.0)
        plt.title("Class: {}".format(class_names[y_batch[index]]))
        plt.axis("off")

plt.show()

# %% [markdown]
# ## 2.2.2 Budowa sieci

# %%
from functools import partial
import keras

# to ustawia jakby domyslne wartosci dla Conv2D
DefaultConv2D = partial(keras.layers.Conv2D,
                        kernel_size=3,
                        activation='relu',
                        padding='SAME',
                        strides=1)


model = keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255),
    DefaultConv2D(filters=32, kernel_size=7),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=64, kernel_size=5),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128, kernel_size=3),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(units=n_classes, activation='softmax')
])


# %%
optimizer = keras.optimizers.SGD(learning_rate=0.001)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        optimizer=optimizer,
                        metrics=['accuracy'])


# %%


history = model.fit(train_set, validation_data=valid_set,
                    epochs=10)

# %%
acc_train = model.evaluate(train_set, return_dict=True)['accuracy']
acc_test = model.evaluate(test_set, return_dict=True)['accuracy']
acc_valid = model.evaluate(valid_set, return_dict=True)['accuracy']

print('Train accuracy : ', acc_train)
print('Test accuracy : ', acc_test)
print('Valid accuracy : ', acc_valid)

result1 = (acc_train, acc_valid, acc_test)



# %%
with open('simple_cnn_acc.pkl', 'wb') as file:
    pickle.dump(result1, file)

model.save('simple_cnn_flowers.keras')

# %% [markdown]
# ## 2.3 Uczenie transferowe

# %%
def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = tf.keras.applications.xception.preprocess_input(resized_image)
    return final_image, label



# %%
batch_size = 32
train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)


# %%
plt.figure(figsize=(8,8))
sample_batch = train_set.take(1)
for X_batch, y_batch in sample_batch:
    for index in range(12):
        plt.subplot(3,4, index+1)
        plt.imshow(X_batch[index] / 2 + 0.5)
        plt.title("Class: {}".format(class_names[y_batch[index]]))
        plt.axis('off')

plt.show()

# %%
base_model = tf.keras.applications.xception.Xception(
    weights="imagenet",
    include_top=False
)

# weights - zapewnia inicjalizajce wag sieci wynikami uczenia ImageNet
# include_top - znaczy ze siec nie posiadac gornych warstw sepcyficznych
#               dla danego problemu, sami musimy je dodac

# %%
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation='softmax')(avg)
model = keras.models.Model(inputs=base_model.input, outputs=output)

# %% [markdown]
# ## Wstepne przyuczanie

# %%
# blokujemy narazie warstwy modleu bazowego

for layer in base_model.layers:
    layer.trainable = False
    

# %%
optimizer = keras.optimizers.SGD(learning_rate=0.2, momentum=0.9,
                                 decay=0.01)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=['accuracy'])

# %%
history = model.fit(train_set, validation_data=valid_set,
                    epochs=5)

# %% [markdown]
# ## Uczenie zasadnicze

# %%
for layer in base_model.layers:
    layer.trainable = True

optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9,
                                 nesterov=True, decay=0.0001)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


# %%
history = model.fit(train_set,
                    validation_data=valid_set,
                    epochs=5)

# %%
acc_train = model.evaluate(train_set, return_dict=True)['accuracy']
acc_test = model.evaluate(test_set, return_dict=True)['accuracy']
acc_valid = model.evaluate(valid_set, return_dict=True)['accuracy']

print('Train accuracy : ', acc_train)
print('Test accuracy : ', acc_test)
print('Valid accuracy : ', acc_valid)

result2 = (acc_train, acc_valid, acc_test)

# %%
with open('xception_acc.pkl', 'wb') as file:
    pickle.dump(result2, file)

model.save('xception_flowers.keras')


