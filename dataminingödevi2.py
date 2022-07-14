# -*- coding: utf-8 -*-
"""
Created on Thu May 26 23:11:37 2022

@author: erenk
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder,  StandardScaler, OrdinalEncoder,MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score, mean_squared_error
import pylab as pl

data=pd.read_excel('Date_Fruit_Datasets.xlsx')
sns.countplot(data['Class'],label="Count")
plt.show()
data.drop('Class' ,axis=1).hist(bins=30, figsize=(9,9))
pl.suptitle("Histogram for each numeric input variable")
plt.savefig('fruits_hist')
plt.show()

ordinal_encoder = OrdinalEncoder()
data[['Class']] = ordinal_encoder.fit_transform(data[['Class']]).astype('float64')

X=data.iloc[:,:34]
Y=data[['Class']]
x=X.values
y=Y.values
sc1=StandardScaler()
sc2=StandardScaler()
data_scaled = pd.DataFrame(sc1.fit_transform(x))
#sc2=StandardScaler()
#result_scaled = pd.DataFrame(sc2.fit_transform(result))#scaled data uymuyor classificationa

data_normal = pd.DataFrame(preprocessing.normalize(x))
minmax=MinMaxScaler()
data_minmax=pd.DataFrame(minmax.fit_transform(x))


x_train_sc, x_test_sc, y_train_sc, y_test_sc = train_test_split(data_scaled.values, y, test_size = 0.2)
x_train_nml, x_test_nml, y_train_nml, y_test_nml = train_test_split(data_normal.values, y, test_size = 0.2)
x_train_mm, x_test_mm, y_train_mm, y_test_mm = train_test_split(data_minmax.values, y, test_size = 0.2)
print("-----------------")



from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(x_train_sc,y_train_sc)
y_pred_svcs_sc = dtc.predict(x_test_sc)

dtc.fit(x_train_nml,y_train_nml)
y_pred_svcs_nml = dtc.predict(x_test_nml)

cm_svcs_sc = confusion_matrix(y_test_sc,y_pred_svcs_sc)
cm_svcs_nml= confusion_matrix(y_test_nml,y_pred_svcs_nml)
print("standardised consuffion  matrix is")
print(cm_svcs_sc)
print("normalised confussion matrix is")
print(cm_svcs_nml)

print('Standardised  Regression R2 score')
print(r2_score(y_test_sc, y_pred_svcs_sc))
print('Normalised  Regression R2 score')
print(r2_score(y_test_nml, y_pred_svcs_nml))

print('Standardised  Regression MAE score')
print(mae(y_test_sc, y_pred_svcs_sc))
print('Normalised  Regression MAE score')
print(mae(y_test_nml, y_pred_svcs_nml))

print('Standardised  Regression MApE score')
print(mae(y_test_sc, y_pred_svcs_sc)*100)
print('Normalised svcs Regression MApE score')
print(mae(y_test_nml, y_pred_svcs_nml)*100)

print('Standardised  Regression MSE score')
print(mean_squared_error(y_test_sc, y_pred_svcs_sc))
print('Normalised  Regression MSE score')
print(mean_squared_error(y_test_nml, y_pred_svcs_nml))
print("--------------------")
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'gini')

dtc.fit(x_train_sc,y_train_sc)
y_pred_svcs_sc = dtc.predict(x_test_sc)

dtc.fit(x_train_nml,y_train_nml)
y_pred_svcs_nml = dtc.predict(x_test_nml)

cm_svcs_sc = confusion_matrix(y_test_sc,y_pred_svcs_sc)
cm_svcs_nml= confusion_matrix(y_test_nml,y_pred_svcs_nml)
print("standardised consuffion  matrix is")
print(cm_svcs_sc)
print("normalised confussion matrix is")
print(cm_svcs_nml)

print('Standardised R2 score')
print(r2_score(y_test_sc, y_pred_svcs_sc))
print('Normalised svcs Regression R2 score')
print(r2_score(y_test_nml, y_pred_svcs_nml))

print('Standardised  Regression MAE score')
print(mae(y_test_sc, y_pred_svcs_sc))
print('Normalised  Regression MAE score')
print(mae(y_test_nml, y_pred_svcs_nml))

print('Standardised  Regression MApE score')
print(mae(y_test_sc, y_pred_svcs_sc)*100)
print('Normalised  Regression MApE score')
print(mae(y_test_nml, y_pred_svcs_nml)*100)

print('Standardised  Regression MSE score')
print(mean_squared_error(y_test_sc, y_pred_svcs_sc))
print('Normalised  Regression MSE score')
print(mean_squared_error(y_test_nml, y_pred_svcs_nml))
print("--------------------")
