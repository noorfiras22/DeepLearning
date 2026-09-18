import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras as keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


dt = pd.read_csv("C:/Users/2o24/Downloads/38833FF26BA1D.UnigramPreview_g9c9v27vpyspw!App/ann_regression_house_prices_10000.csv")
x=dt.iloc[:,:-1].values
y=dt.iloc[:,-1].values


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=0)

x_sc=StandardScaler()
y_sc=StandardScaler()

x_train_scaled=x_sc.fit_transform(x_train)
x_test_scaled=x_sc.transform(x_test)
y_train_scaled=y_sc.fit_transform(y_train.reshape(-1,1))

ann=keras.models.Sequential()
ann.add(keras.layers.Dense(activation="relu",units=6))
ann.add(keras.layers.Dense(activation="relu",units=6))
ann.add(keras.layers.Dense(units=1))

ann.compile(optimizer="adam",loss="mean_squared_error")

ann.fit(x_train_scaled,y_train_scaled,epochs=100,batch_size=50)

y_pred=ann.predict(x_test_scaled)

n=np.column_stack((y_test,y_sc.inverse_transform(y_pred)))
print(n)



print("MSE:",mean_squared_error(y_test,y_sc.inverse_transform(y_pred)))
