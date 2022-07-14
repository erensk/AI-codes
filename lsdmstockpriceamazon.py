# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 20:10:47 2022

@author: erenk
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

dataset_train = pd.read_csv('AMZN.csv')
training_set = dataset_train.iloc[:, 1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(60, 3272):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

dataset_test = pd.read_csv('AMZN_TEST.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 71):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
print(' R2 score')
print(r2_score(real_stock_price, predicted_stock_price))
print(' MAE score')
print(mae(real_stock_price,predicted_stock_price))
print(' MApE score')
print(mae(real_stock_price, predicted_stock_price)*100)
print('MSE score')
print(mean_squared_error(real_stock_price, predicted_stock_price))
print("--------------------")


plt.plot(real_stock_price, color = 'red', label = 'amazon Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted amazon Stock Price')
plt.title('AMAZON Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('AMAZON Stock Price')
plt.legend()
plt.show()
