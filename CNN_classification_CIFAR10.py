import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], "GPU")
        logical = tf.config.list_logical_devices("GPU")
        print(f"Using GPU: {logical[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("WARNING: No GPU found by TensorFlow - running on CPU")

import pandas as pd
import keras as keras 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
from keras.regularizers import l2
import zipfile
import numpy as np


_data=np.load("cifar10_data.npz")
(x_train,y_train),(x_test,y_test)=(_data["x_train"],_data["y_train"]),(_data["x_test"],_data["y_test"])

train_datagen=ImageDataGenerator(
    rescale=1/255,
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.05
)

test_datagen=ImageDataGenerator(
    rescale=1/255
)

train_set=train_datagen.flow(
    x_train,
    y_train,
    batch_size=64
)

test_set=test_datagen.flow(
    x_test,
    y_test,
    batch_size=64,
    shuffle=False
)

tracker=ReduceLROnPlateau(
    monitor="val_loss",
    patience=3,
    factor=0.5,
    min_lr=0.0001
)

early=EarlyStopping(
    patience=7,
    monitor="val_loss",
    restore_best_weights=True
)

cnn=keras.Sequential()

cnn.add(keras.layers.Conv2D(kernel_size=3,filters=32,padding="same",activation=None,kernel_regularizer=l2(1e-4),use_bias=False,input_shape=(32,32,3)))
cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.Activation("relu"))


cnn.add(keras.layers.Conv2D(kernel_size=3,filters=32,padding="same",activation=None,kernel_regularizer=l2(1e-4),use_bias=False))
cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.Activation("relu"))

cnn.add(keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(keras.layers.Dropout(0.1))


cnn.add(keras.layers.Conv2D(kernel_size=3,filters=64,padding="same",activation=None,kernel_regularizer=l2(1e-4),use_bias=False))
cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.Activation("relu"))

cnn.add(keras.layers.Conv2D(kernel_size=3,filters=64,padding="same",activation=None,kernel_regularizer=l2(1e-4),use_bias=False))
cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.Activation("relu"))



cnn.add(keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(keras.layers.Dropout(0.1))


cnn.add(keras.layers.Conv2D(kernel_size=3,filters=128,padding="same",activation = None,kernel_regularizer=l2(1e-4),use_bias=False))
cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.Activation("relu"))

cnn.add(keras.layers.Conv2D(kernel_size=3,filters=128,padding="same",activation = None,kernel_regularizer=l2(1e-4),use_bias=False))
cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.Activation("relu"))


cnn.add(keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(keras.layers.Dropout(0.1))

cnn.add(keras.layers.GlobalAveragePooling2D())

cnn.add(keras.layers.Dense(units=128,activation="relu"))
cnn.add(keras.layers.Dropout(0.5))


cnn.add(keras.layers.Dense(units=10,activation="softmax"))

cnn.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
cnn.fit(x = train_set,validation_data=test_set,epochs = 40,callbacks = [early,tracker])


from pathlib import Path

model_path = Path(__file__).resolve().parent / "cifar10_cnn.keras"
cnn.save(str(model_path))
print(f"Model saved to {model_path}")
