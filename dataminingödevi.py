# -*- coding: utf-8 -*-
"""
Created on Sun May 15 22:45:22 2022

@author: erenk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn import metrics
from sklearn.preprocessing import   StandardScaler,MinMaxScaler
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.preprocessing import  OrdinalEncoder
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
np.random.seed(123)
data = pd.read_csv('Mall_Customers.csv')

ordinal_encoder = OrdinalEncoder()
data[['Gender']] = ordinal_encoder.fit_transform(data[['Gender']]).astype('float64')
X=data.drop(['CustomerID'],axis=1)
X = X.values

from sklearn.cluster import KMeans
kmeans = KMeans ( n_clusters = 4, init = 'k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)
sonuclar = []
for i in range(1,11):
    kmeans = KMeans (n_clusters = i, init='k-means++', random_state= 123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
    cluster_labels = kmeans.labels_
range_n_clusters = [1,2,3,4,5,6,7,8,9,10,11]
plt.plot(range(1,11),sonuclar)
plt.show()
print('---------------')
kmeans = KMeans (n_clusters = 3, init='k-means++', random_state= 123)
Y_tahmin= kmeans.fit_predict(X)
print(Y_tahmin)
plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100, c='red')
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100, c='blue')
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100, c='green')
plt.title('KMeans')
plt.show()
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, Y_tahmin))
print(kmeans.labels_)
print('------------------')
kmeans = KMeans (n_clusters = 6, init='k-means++', random_state= 123)
Y_tahmin= kmeans.fit_predict(X)
print(Y_tahmin)
plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100, c='red')
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100, c='blue')
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100, c='green')
plt.scatter(X[Y_tahmin==3,0],X[Y_tahmin==3,1],s=100, c='yellow')
plt.scatter(X[Y_tahmin==4,0],X[Y_tahmin==4,1],s=100, c='black')
plt.scatter(X[Y_tahmin==5,0],X[Y_tahmin==5,1],s=100, c='purple')
plt.title('KMeans')
plt.show()
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, Y_tahmin))
print(kmeans.labels_)
score = metrics.accuracy_score(Y_tahmin,kmeans.predict(X))
print('Accuracy:{0:f}'.format(score))
print('------------------')

X=data.drop(['Gender','CustomerID'],axis=1)
X=X.values
from sklearn.cluster import KMeans
kmeans = KMeans ( n_clusters = 4, init = 'k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)
sonuclar = []
for i in range(1,11):
    kmeans = KMeans (n_clusters = i, init='k-means++', random_state= 123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
    cluster_labels = kmeans.labels_
range_n_clusters = [1,2,3,4,5,6,7,8,9,10,11]
plt.plot(range(1,11),sonuclar)
plt.show()
print('---------------')
kmeans = KMeans (n_clusters = 4, init='k-means++', random_state= 123)
Y_tahmin= kmeans.fit_predict(X)
print(Y_tahmin)
plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100, c='red')
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100, c='blue')
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100, c='green')
plt.scatter(X[Y_tahmin==3,0],X[Y_tahmin==3,1],s=100, c='yellow')
plt.title('KMeans2')
plt.show()
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, Y_tahmin))
print(kmeans.labels_)
print('------------------')
kmeans = KMeans (n_clusters = 3, init='k-means++', random_state= 123)
Y_tahmin= kmeans.fit_predict(X)
print(Y_tahmin)
plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100, c='red')
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100, c='blue')
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100, c='green')
plt.title('KMeans2')
plt.show()
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, Y_tahmin))
print(kmeans.labels_)
print('------------------')
kmeans = KMeans (n_clusters = 2, init='k-means++', random_state= 123)
Y_tahmin= kmeans.fit_predict(X)
print(Y_tahmin)
plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100, c='red')
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100, c='blue')

plt.title('KMeans2')
plt.show()
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, Y_tahmin))
print(kmeans.labels_)
print('------------------')
kmeans = KMeans (n_clusters = 6, init='k-means++', random_state= 123)
Y_tahmin= kmeans.fit_predict(X)
print(Y_tahmin)
plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100, c='red')
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100, c='blue')
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100, c='green')
plt.scatter(X[Y_tahmin==3,0],X[Y_tahmin==3,1],s=100, c='yellow')
plt.scatter(X[Y_tahmin==4,0],X[Y_tahmin==4,1],s=100, c='black')
plt.scatter(X[Y_tahmin==5,0],X[Y_tahmin==5,1],s=100, c='purple')
plt.title('KMeans2')
plt.show()
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, Y_tahmin))
print(kmeans.labels_)
score = metrics.accuracy_score(Y_tahmin,kmeans.predict(X))
print('Accuracy:{0:f}'.format(score))
print('------------------')
