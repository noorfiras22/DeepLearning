import pandas as pd
import numpy as np
import keras as keras
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix

dt = pd.read_csv("Churn_Modelling.csv")
x=dt.iloc[:,3:-1].values
y=dt.iloc[:,-1].values

lb=LabelEncoder()
x[:,2]=lb.fit_transform(x[:,2])

c1=ColumnTransformer([("Noor",OneHotEncoder(),[1])],remainder="passthrough")
x=c1.fit_transform(x)


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

sc=StandardScaler()

x_train_scaled=sc.fit_transform(x_train)
x_test_scaled=sc.transform(x_test)


teacher = ReduceLROnPlateau(
    monitor = "val_loss",
    patience = 3,
    factor = 0.5,
    min_lr = 1e-6
)

early = EarlyStopping(
    monitor = "val_loss",
    patience = 7,
    restore_best_weights= True
)

ann = keras.Sequential()

ann.add(keras.layers.Dense(activation="relu",units=10))
ann.add(keras.layers.Dense(activation="relu",units=10))
ann.add(keras.layers.Dense(activation="sigmoid",units=1))

ann.compile(optimizer="adam" , loss="binary_crossentropy",metrics=["accuracy"])
ann.fit(x_train_scaled,y_train,batch_size=12,epochs=100,validation_split=0.2,callbacks=[teacher,early])

y_pred=ann.predict(x_test_scaled)
y_pred_labels=(y_pred>0.5).astype("int")
print(confusion_matrix(y_test,y_pred_labels))]