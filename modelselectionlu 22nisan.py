# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 22:45:22 2022

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


corr = data.corr()
sns.heatmap(corr)
#selecting the ones with less than 0.9 correlation
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = data.columns[columns]
data = data[selected_columns]
z=data.iloc[:,:34].values

#discarding the ones with more than 0.5 pvalues
selected_columns = selected_columns[0:].values
import statsmodels.api as sm
def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)

    regressor_OLS.summary()
    return x, columns
SL = 0.05
data_modeled, selected_columns = backwardElimination(data.iloc[:,:34].values, data[['Class']].values, SL, selected_columns)

result = pd.DataFrame()
result['Class'] = data[['Class']]

data = pd.DataFrame(data = data_modeled, columns = selected_columns)

#standardisation
sc1=StandardScaler()
sc2=StandardScaler()
data_scaled = pd.DataFrame(sc1.fit_transform(data[['MeanRR','EntropyRR']]))
#sc2=StandardScaler()
#result_scaled = pd.DataFrame(sc2.fit_transform(result))#scaled data uymuyor classificationa

data_normal = pd.DataFrame(preprocessing.normalize(data[['MeanRR','EntropyRR']]))
minmax=MinMaxScaler()#bu seç
data_minmax=pd.DataFrame(minmax.fit_transform(data[['MeanRR','EntropyRR']]))


x_train_sc, x_test_sc, y_train_sc, y_test_sc = train_test_split(data_scaled.values, result.values, test_size = 0.2)
x_train_nml, x_test_nml, y_train_nml, y_test_nml = train_test_split(data_normal.values, result.values, test_size = 0.2)
x_train_mm, x_test_mm, y_train_mm, y_test_mm = train_test_split(data_minmax.values, result.values, test_size = 0.2)
print("-----------------")

#Logistic Regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)

logr.fit(x_train_sc,y_train_sc)
y_pred_sc = logr.predict(x_test_sc)

logr.fit(x_train_nml,y_train_nml)
y_pred_nml = logr.predict(x_test_nml)

cm_sc = confusion_matrix(y_test_sc,y_pred_sc)
cm_nml= confusion_matrix(y_test_nml,y_pred_nml)
print("standardised confussion Logistic regression matrix is")
print(cm_sc)
print("normalised confussion matrix is")
print(cm_nml)

print('Standardised Logistic Regression R2 score')
print(r2_score(y_test_sc, y_pred_sc))
print('Normalised Logistic Regression R2 score')
print(r2_score(y_test_nml, y_pred_nml))

print('Standardised Logistic Regression MAE score')
print(mae(y_test_sc, y_pred_sc))
print('Normalised Logistic Regression MAE score')
print(mae(y_test_nml, y_pred_nml))

print('Standardised Logistic Regression MApE score')
print(mae(y_test_sc, y_pred_sc)*100)
print('Normalised Logistic Regression MApE score')
print(mae(y_test_nml, y_pred_nml)*100)

print('Standardised Logistic Regression MSE score')
print(mean_squared_error(y_test_sc, y_pred_sc))
print('Normalised Logistic Regression MSE score')
print(mean_squared_error(y_test_nml, y_pred_nml))

print("Coefficients: \n", logr.coef_)

print("--------------------")
#knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2, metric='minkowski')

knn.fit(x_train_sc,y_train_sc)
y_pred_knn_sc = knn.predict(x_test_sc)

knn.fit(x_train_nml,y_train_nml)
y_pred_knn_nml = knn.predict(x_test_nml)

cm_knn_sc = confusion_matrix(y_test_sc,y_pred_knn_sc)
cm_knn_nml= confusion_matrix(y_test_nml,y_pred_knn_nml)
print("standardised consuffion matrix is")
print(cm_knn_sc)
print("normalised confussion matrix is")
print(cm_knn_nml)

print('Standardised knn Regression R2 score')
print(r2_score(y_test_sc, y_pred_knn_sc))
print('Normalised knn Regression R2 score')
print(r2_score(y_test_nml, y_pred_knn_nml))

print('Standardised knn Regression MAE score')
print(mae(y_test_sc, y_pred_knn_sc))
print('Normalised knn Regression MAE score')
print(mae(y_test_nml, y_pred_knn_nml))

print('Standardised knn Regression MApE score')
print(mae(y_test_sc, y_pred_knn_sc)*100)
print('Normalised knn Regression MApE score')
print(mae(y_test_nml, y_pred_knn_nml)*100)

print('Standardised knn Regression MSE score')
print(mean_squared_error(y_test_sc, y_pred_knn_sc))
print('Normalised knn Regression MSE score')
print(mean_squared_error(y_test_nml, y_pred_knn_nml))

print("--------------------")
#knn eucledean

knne = KNeighborsClassifier(n_neighbors=2, metric='euclidean')

knne.fit(x_train_sc,y_train_sc)
y_pred_knne_sc = knne.predict(x_test_sc)

knne.fit(x_train_nml,y_train_nml)
y_pred_knne_nml = knne.predict(x_test_nml)

cm_knne_sc = confusion_matrix(y_test_sc,y_pred_knne_sc)
cm_knne_nml= confusion_matrix(y_test_nml,y_pred_knne_nml)
print("standardised consuffion euclidean matrix is")
print(cm_knne_sc)
print("normalised confussion matrix is")
print(cm_knne_nml)

print('Standardised knne Regression R2 score')
print(r2_score(y_test_sc, y_pred_knne_sc))
print('Normalised knne Regression R2 score')
print(r2_score(y_test_nml, y_pred_knne_nml))

print('Standardised knne Regression MAE score')
print(mae(y_test_sc, y_pred_knne_sc))
print('Normalised knne Regression MAE score')
print(mae(y_test_nml, y_pred_knne_nml))

print('Standardised knne Regression MApE score')
print(mae(y_test_sc, y_pred_knne_sc)*100)
print('Normalised knne Regression MApE score')
print(mae(y_test_nml, y_pred_knne_nml)*100)

print('Standardised knne Regression MSE score')
print(mean_squared_error(y_test_sc, y_pred_knne_sc))
print('Normalised knne Regression MSE score')
print(mean_squared_error(y_test_nml, y_pred_knne_nml))
print("--------------------")

#knn manhattan

knnm = KNeighborsClassifier(n_neighbors=2, metric='manhattan')

knnm.fit(x_train_sc,y_train_sc)
y_pred_knnm_sc = knnm.predict(x_test_sc)

knnm.fit(x_train_nml,y_train_nml)
y_pred_knnm_nml = knnm.predict(x_test_nml)

cm_knnm_sc = confusion_matrix(y_test_sc,y_pred_knnm_sc)
cm_knnm_nml= confusion_matrix(y_test_nml,y_pred_knnm_nml)
print("standardised consuffion manhattan matrix is")
print(cm_knnm_sc)
print("normalised confussion matrix is")
print(cm_knnm_nml)

print('Standardised knnm Regression R2 score')
print(r2_score(y_test_sc, y_pred_knne_sc))
print('Normalised knnm Regression R2 score')
print(r2_score(y_test_nml, y_pred_knnm_nml))

print('Standardised knnm Regression MAE score')
print(mae(y_test_sc, y_pred_knnm_sc))
print('Normalised knnm Regression MAE score')
print(mae(y_test_nml, y_pred_knnm_nml))

print('Standardised knnm Regression MApE score')
print(mae(y_test_sc, y_pred_knnm_sc)*100)
print('Normalised knnm Regression MApE score')
print(mae(y_test_nml, y_pred_knnm_nml)*100)

print('Standardised knnm Regression MSE score')
print(mean_squared_error(y_test_sc, y_pred_knnm_sc))
print('Normalised knnm Regression MSE score')
print(mean_squared_error(y_test_nml, y_pred_knnm_nml))
print("--------------------")

#svc classifier
from sklearn.svm import SVC
svc = SVC(kernel='poly')

svc.fit(x_train_sc,y_train_sc)
y_pred_svc_sc = svc.predict(x_test_sc)

svc.fit(x_train_nml,y_train_nml)
y_pred_svc_nml = svc.predict(x_test_nml)

cm_svc_sc = confusion_matrix(y_test_sc,y_pred_svc_sc)
cm_svc_nml= confusion_matrix(y_test_nml,y_pred_svc_nml)
print("standardised consuffion svc matrix is")
print(cm_svc_sc)
print("normalised confussion matrix is")
print(cm_svc_nml)

print('Standardised svc Regression R2 score')
print(r2_score(y_test_sc, y_pred_svc_sc))
print('Normalised svc Regression R2 score')
print(r2_score(y_test_nml, y_pred_svc_nml))

print('Standardised svc Regression MAE score')
print(mae(y_test_sc, y_pred_svc_sc))
print('Normalised svc Regression MAE score')
print(mae(y_test_nml, y_pred_svc_nml))

print('Standardised svc Regression MpAE score')
print(mae(y_test_sc, y_pred_svc_sc)*100)
print('Normalised svc Regression MApE score')
print(mae(y_test_nml, y_pred_svc_nml)*100)

print('Standardised svc Regression MSE score')
print(mean_squared_error(y_test_sc, y_pred_svc_sc))
print('Normalised svc Regression MSE score')
print(mean_squared_error(y_test_nml, y_pred_svc_nml))
print("--------------------")

#svc with linear kernel
svc = SVC(kernel='linear')

svc.fit(x_train_sc,y_train_sc)
y_pred_svcl_sc = svc.predict(x_test_sc)

svc.fit(x_train_nml,y_train_nml)
y_pred_svcl_nml = svc.predict(x_test_nml)

cm_svcl_sc = confusion_matrix(y_test_sc,y_pred_svcl_sc)
cm_svcl_nml= confusion_matrix(y_test_nml,y_pred_svcl_nml)
print("standardised consuffion svcl matrix is")
print(cm_svcl_sc)
print("normalised confussion matrix is")
print(cm_svcl_nml)

print('Standardised svcl Regression R2 score')
print(r2_score(y_test_sc, y_pred_svcl_sc))
print('Normalised svcl Regression R2 score')
print(r2_score(y_test_nml, y_pred_svcl_nml))

print('Standardised svcl Regression MAE score')
print(mae(y_test_sc, y_pred_svcl_sc))
print('Normalised svcl Regression MAE score')
print(mae(y_test_nml, y_pred_svcl_nml))

print('Standardised svcl Regression MApE score')
print(mae(y_test_sc, y_pred_svcl_sc)*100)
print('Normalised svcl Regression MApE score')
print(mae(y_test_nml, y_pred_svcl_nml)*100)

print('Standardised svcl Regression MSE score')
print(mean_squared_error(y_test_sc, y_pred_svcl_sc))
print('Normalised svcl Regression MSE score')
print(mean_squared_error(y_test_nml, y_pred_svcl_nml))

print("Coefficients: \n", svc.coef_)

print("--------------------")

#svc with rbf kernel
svc = SVC(kernel='rbf')

svc.fit(x_train_sc,y_train_sc)
y_pred_svcr_sc = svc.predict(x_test_sc)

svc.fit(x_train_nml,y_train_nml)
y_pred_svcr_nml = svc.predict(x_test_nml)

cm_svcr_sc = confusion_matrix(y_test_sc,y_pred_svcr_sc)
cm_svcr_nml= confusion_matrix(y_test_nml,y_pred_svcr_nml)
print("standardised consuffion svcr matrix is")
print(cm_svcr_sc)
print("normalised confussion matrix is")
print(cm_svcr_nml)

print('Standardised svcr Regression R2 score')
print(r2_score(y_test_sc, y_pred_svcr_sc))
print('Normalised svcr Regression R2 score')
print(r2_score(y_test_nml, y_pred_svcr_nml))

print('Standardised svcr Regression MAE score')
print(mae(y_test_sc, y_pred_svcr_sc))
print('Normalised svcr Regression MAE score')
print(mae(y_test_nml, y_pred_svcr_nml))

print('Standardised svcr Regression MApE score')
print(mae(y_test_sc, y_pred_svcr_sc)*100)
print('Normalised svcr Regression MApE score')
print(mae(y_test_nml, y_pred_svcr_nml)*100)

print('Standardised svcr Regression MSE score')
print(mean_squared_error(y_test_sc, y_pred_svcr_sc))
print('Normalised svcr Regression MSE score')
print(mean_squared_error(y_test_nml, y_pred_svcr_nml))

print("--------------------")

#svc with sigmoid kernel
svc = SVC(kernel='sigmoid')

svc.fit(x_train_sc,y_train_sc)
y_pred_svcs_sc = svc.predict(x_test_sc)

svc.fit(x_train_nml,y_train_nml)
y_pred_svcs_nml = svc.predict(x_test_nml)

cm_svcs_sc = confusion_matrix(y_test_sc,y_pred_svcs_sc)
cm_svcs_nml= confusion_matrix(y_test_nml,y_pred_svcs_nml)
print("standardised consuffion svcs matrix is")
print(cm_svcs_sc)
print("normalised confussion matrix is")
print(cm_svcs_nml)

print('Standardised svcs Regression R2 score')
print(r2_score(y_test_sc, y_pred_svcs_sc))
print('Normalised svcs Regression R2 score')
print(r2_score(y_test_nml, y_pred_svcs_nml))

print('Standardised svcs Regression MAE score')
print(mae(y_test_sc, y_pred_svcs_sc))
print('Normalised svcs Regression MAE score')
print(mae(y_test_nml, y_pred_svcs_nml))

print('Standardised svcs Regression MApE score')
print(mae(y_test_sc, y_pred_svcs_sc)*100)
print('Normalised svcs Regression MApE score')
print(mae(y_test_nml, y_pred_svcs_nml)*100)

print('Standardised svcs Regression MSE score')
print(mean_squared_error(y_test_sc, y_pred_svcs_sc))
print('Normalised svcs Regression MSE score')
print(mean_squared_error(y_test_nml, y_pred_svcs_nml))
print("--------------------")

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

gnb.fit(x_train_sc,y_train_sc)
y_pred_svcs_sc = gnb.predict(x_test_sc)

gnb.fit(x_train_nml,y_train_nml)
y_pred_svcs_nml = gnb.predict(x_test_nml)

cm_svcs_sc = confusion_matrix(y_test_sc,y_pred_svcs_sc)
cm_svcs_nml= confusion_matrix(y_test_nml,y_pred_svcs_nml)
print("standardised consuffion svcs matrix is")
print(cm_svcs_sc)
print("normalised confussion matrix is")
print(cm_svcs_nml)

print('Standardised svcs Regression R2 score')
print(r2_score(y_test_sc, y_pred_svcs_sc))
print('Normalised svcs Regression R2 score')
print(r2_score(y_test_nml, y_pred_svcs_nml))

print('Standardised svcs Regression MAE score')
print(mae(y_test_sc, y_pred_svcs_sc))
print('Normalised svcs Regression MAE score')
print(mae(y_test_nml, y_pred_svcs_nml))

print('Standardised svcs Regression MApE score')
print(mae(y_test_sc, y_pred_svcs_sc)*100)
print('Normalised svcs Regression MApE score')
print(mae(y_test_nml, y_pred_svcs_nml)*100)

print('Standardised svcs Regression MSE score')
print(mean_squared_error(y_test_sc, y_pred_svcs_sc))
print('Normalised svcs Regression MSE score')
print(mean_squared_error(y_test_nml, y_pred_svcs_nml))
print("--------------------")

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
#clf.fit(x_train_sc,y_train_sc)
#y_pred_svcs_sc = clf.predict(x_test_sc)

clf.fit(x_train_mm,y_train_mm)
y_pred_svcs_nml = clf.predict(x_test_mm)

cm_svcs_sc = confusion_matrix(y_test_sc,y_pred_svcs_sc)
cm_svcs_nml= confusion_matrix(y_test_nml,y_pred_svcs_nml)
print("standardised consuffion svcs matrix is")
#print(cm_svcs_sc)
print("normalised confussion matrix is")
print(cm_svcs_nml)

print('Standardised svcs Regression R2 score')
#print(r2_score(y_test_sc, y_pred_svcs_sc))
print('Normalised svcs Regression R2 score')
print(r2_score(y_test_nml, y_pred_svcs_nml))

print('Standardised svcs Regression MAE score')
#print(mae(y_test_sc, y_pred_svcs_sc))
print('Normalised svcs Regression MAE score')
print(mae(y_test_nml, y_pred_svcs_nml))

print('Standardised svcs Regression MApE score')
#print(mae(y_test_sc, y_pred_svcs_sc)*100)
print('Normalised svcs Regression MApE score')
print(mae(y_test_nml, y_pred_svcs_nml)*100)

print('Standardised svcs Regression MSE score')
#print(mean_squared_error(y_test_sc, y_pred_svcs_sc))
print('Normalised svcs Regression MSE score')
print(mean_squared_error(y_test_nml, y_pred_svcs_nml))
print("--------------------")
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(x_train_sc,y_train_sc)
y_pred_svcs_sc = dtc.predict(x_test_sc)

dtc.fit(x_train_nml,y_train_nml)
y_pred_svcs_nml = dtc.predict(x_test_nml)

cm_svcs_sc = confusion_matrix(y_test_sc,y_pred_svcs_sc)
cm_svcs_nml= confusion_matrix(y_test_nml,y_pred_svcs_nml)
print("standardised consuffion svcs matrix is")
print(cm_svcs_sc)
print("normalised confussion matrix is")
print(cm_svcs_nml)

print('Standardised svcs Regression R2 score')
print(r2_score(y_test_sc, y_pred_svcs_sc))
print('Normalised svcs Regression R2 score')
print(r2_score(y_test_nml, y_pred_svcs_nml))

print('Standardised svcs Regression MAE score')
print(mae(y_test_sc, y_pred_svcs_sc))
print('Normalised svcs Regression MAE score')
print(mae(y_test_nml, y_pred_svcs_nml))

print('Standardised svcs Regression MApE score')
print(mae(y_test_sc, y_pred_svcs_sc)*100)
print('Normalised svcs Regression MApE score')
print(mae(y_test_nml, y_pred_svcs_nml)*100)

print('Standardised svcs Regression MSE score')
print(mean_squared_error(y_test_sc, y_pred_svcs_sc))
print('Normalised svcs Regression MSE score')
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
print("standardised consuffion svcs matrix is")
print(cm_svcs_sc)
print("normalised confussion matrix is")
print(cm_svcs_nml)

print('Standardised svcs Regression R2 score')
print(r2_score(y_test_sc, y_pred_svcs_sc))
print('Normalised svcs Regression R2 score')
print(r2_score(y_test_nml, y_pred_svcs_nml))

print('Standardised svcs Regression MAE score')
print(mae(y_test_sc, y_pred_svcs_sc))
print('Normalised svcs Regression MAE score')
print(mae(y_test_nml, y_pred_svcs_nml))

print('Standardised svcs Regression MApE score')
print(mae(y_test_sc, y_pred_svcs_sc)*100)
print('Normalised svcs Regression MApE score')
print(mae(y_test_nml, y_pred_svcs_nml)*100)

print('Standardised svcs Regression MSE score')
print(mean_squared_error(y_test_sc, y_pred_svcs_sc))
print('Normalised svcs Regression MSE score')
print(mean_squared_error(y_test_nml, y_pred_svcs_nml))
print("--------------------")

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')


rfc.fit(x_train_sc,y_train_sc)
y_pred_svcs_sc = rfc.predict(x_test_sc)

rfc.fit(x_train_nml,y_train_nml)
y_pred_svcs_nml = rfc.predict(x_test_nml)
cm_svcs_sc = confusion_matrix(y_test_sc,y_pred_svcs_sc)
cm_svcs_nml= confusion_matrix(y_test_nml,y_pred_svcs_nml)
print("standardised consuffion svcs matrix is")
print(cm_svcs_sc)
print("normalised confussion matrix is")
print(cm_svcs_nml)

print('Standardised svcs Regression R2 score')
print(r2_score(y_test_sc, y_pred_svcs_sc))
print('Normalised svcs Regression R2 score')
print(r2_score(y_test_nml, y_pred_svcs_nml))

print('Standardised svcs Regression MAE score')
print(mae(y_test_sc, y_pred_svcs_sc))
print('Normalised svcs Regression MAE score')
print(mae(y_test_nml, y_pred_svcs_nml))

print('Standardised svcs Regression MApE score')
print(mae(y_test_sc, y_pred_svcs_sc)*100)
print('Normalised svcs Regression MApE score')
print(mae(y_test_nml, y_pred_svcs_nml)*100)

print('Standardised svcs Regression MSE score')
print(mean_squared_error(y_test_sc, y_pred_svcs_sc))
print('Normalised svcs Regression MSE score')
print(mean_squared_error(y_test_nml, y_pred_svcs_nml))
print("--------------------")
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, criterion = 'gini')


rfc.fit(x_train_sc,y_train_sc)
y_pred_svcs_sc = rfc.predict(x_test_sc)

rfc.fit(x_train_nml,y_train_nml)
y_pred_svcs_nml = rfc.predict(x_test_nml)
cm_svcs_sc = confusion_matrix(y_test_sc,y_pred_svcs_sc)
cm_svcs_nml= confusion_matrix(y_test_nml,y_pred_svcs_nml)
print("standardised consuffion svcs matrix is")
print(cm_svcs_sc)
print("normalised confussion matrix is")
print(cm_svcs_nml)

print('Standardised svcs Regression R2 score')
print(r2_score(y_test_sc, y_pred_svcs_sc))
print('Normalised svcs Regression R2 score')
print(r2_score(y_test_nml, y_pred_svcs_nml))

print('Standardised svcs Regression MAE score')
print(mae(y_test_sc, y_pred_svcs_sc))
print('Normalised svcs Regression MAE score')
print(mae(y_test_nml, y_pred_svcs_nml))

print('Standardised svcs Regression MApE score')
print(mae(y_test_sc, y_pred_svcs_sc)*100)
print('Normalised svcs Regression MApE score')
print(mae(y_test_nml, y_pred_svcs_nml)*100)

print('Standardised svcs Regression MSE score')
print(mean_squared_error(y_test_sc, y_pred_svcs_sc))
print('Normalised svcs Regression MSE score')
print(mean_squared_error(y_test_nml, y_pred_svcs_nml))
print("--------------------")
