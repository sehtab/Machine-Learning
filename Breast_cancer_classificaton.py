#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 03:29:56 2020

@author: altair
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys
print(cancer)
print(cancer.keys())
print(cancer['DESCR'])
print(cancer['target'])
print(cancer['target_names'])
print(cancer['feature_names'])
print(cancer['data'].shape)

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'],['target']))
print(df_cancer.head())

sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])
#print(df_cancer.tail())
sns.countplot(df_cancer['target'])
#print(df_cancer.head())
sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)
plt.figure(figsize = (20, 10))
sns.heatmap(df_cancer.corr(), annot = True)

X = df_cancer.drop(['target'], axis = 1)
y = df_cancer['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
svc_model = SVC()
print(svc_model.fit(X_train,y_train))
print(X_train)

y_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot = True)

print(classification_report(y_test, y_predict))

min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scaled = (X_train - min_train)/range_train
svc_model.fit(X_train_scaled, y_train)

sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train)

sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)

min_test = X_test.min()
range_train = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_train
svc_model.fit(X_train_scaled, y_train)

y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot = True)
print(classification_report(y_test, y_predict))

param_grid = {'C':[0.1, 1, 10, 100], 'gamma':[1,0.1,0.01,0.001], 'kernel':['rbf']}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 4)
grid.fit(X_train_scaled, y_train)
grid.best_params_
grid_predictions = grid.predict(X_test_scaled)
cm = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, grid_predictions))
