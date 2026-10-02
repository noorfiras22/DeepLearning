import keras as keras 
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
from keras.regularizers import l2

(x_train , y_train),(x_test,y_test)=mnist.load_data()

x_train=np.expand_dims(x_train,axis=-1)
x_test=np.expand_dims(x_test,axis=-1)

train_datagen=ImageDataGenerator(
    rescale=1/255,
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


inputs=keras.Input(
    shape=(28,28,1)
)

x=keras.layers.Conv2D(
   filters=32,
   kernel_size=3,
   padding="same",
   activation=None,
   kernel_regularizer=l2(1e-4),
   use_bias=False
)(inputs)

x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation("relu")(x)

shortcut=x

x=keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    padding="same",
    activation=None,
    kernel_regularizer=l2(1e-4),
    use_bias=False
)(x)

x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation("relu")(x)

x=keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    padding="same",
    activation=None,
    kernel_regularizer=l2(1e-4),
    use_bias=False
)(x)

x=keras.layers.BatchNormalization()(x)

x=keras.layers.Add()([x,shortcut])

x=keras.layers.Activation("relu")(x)


x=keras.layers.MaxPool2D(
    pool_size=2,
    strides=2
)(x)

x=keras.layers.Dropout(0.1)(x)

x=keras.layers.Conv2D(
    filters=64,
    kernel_size=3,
    padding="same",
    activation=None,
    kernel_regularizer=l2(1e-4),
    use_bias=False
)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation("relu")(x)

x=keras.layers.Conv2D(
    filters=64,
    kernel_size=3,
    padding="same",
    activation=None,
    kernel_regularizer=l2(1e-4),
    use_bias=False
)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation("relu")(x)

x=keras.layers.MaxPool2D(
    pool_size=2,
    strides=2
)(x)

x=keras.layers.GlobalAveragePooling2D()(x)

x=keras.layers.Dense(
    units=128,
    activation="relu"
)(x)

x=keras.layers.Dropout(0.5)(x)

outputs=keras.layers.Dense(
    units=10,
    activation="softmax"
)(x)

cnn=keras.Model(
    inputs=inputs,
    outputs=outputs
)

cnn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-5),
    loss="sparse_categorical_crossentropy",metrics=["accuracy"]
)

cnn.fit(
    x=train_set,
    validation_data=test_set,
    epochs=40,
    callbacks=[early,tracker]
)

test_loss,test_accuracy=cnn.evaluate(
    test_set
)
print("Final Test Accuracy:", test_accuracy)
print("Final Test Accuracy (%):", test_accuracy * 100)


cnn.save("mnist_residual_cnn.keras")

print("Model saved successfully!")
