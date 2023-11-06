# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:36:21 2023

@author: aditi
"""

import pandas as pd
import numpy as np
import pickle

#df = pd.read_excel(r"C:\Users\aditi\Downloads\airqualityindex.xlsx")
df = pd.read_csv(r"D:/Users/aditi/air_quality_Model_Deployment/airqualityindex.csv")
#df = df.drop(['web-scraper-order', 'web-scraper-start-url', 'Country-href', 'State-href', 'City-href', 'Status', 'Temp', 'Humid'], axis=1)
df = df.drop(['web-scraper-order', 'web-scraper-start-url', 'Country-href', 'State-href', 'City-href', 'Status'], axis=1)


df.loc[df['PM25'] == 0, 'PM25'] = float("nan")
df.loc[df['PM10'] == 0, 'PM25'] = float("nan")


df['PM25'] = df['PM25'].fillna(df.groupby('City')['PM25'].transform('mean'))
df['PM10'] = df['PM10'].fillna(df.groupby('City')['PM10'].transform('mean'))

df = df.drop(['Country', 'State', 'City', 'LOCATIONS','AQI-US'], axis=1)

df = df[pd.notnull(df['PM25'])]
df = df[pd.notnull(df['PM10'])]

from sklearn.model_selection import train_test_split
#x = pd.DataFrame({'pm2.5' : df['PM25'], 'pm10' : df['PM10']})
x = pd.DataFrame({'pm2.5' : df['PM25'], 'pm10' : df['PM10'],'Temp' : df['Temp'], 'Humid' : df['Humid']})

y = df['AQI-IN']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, shuffle=False, test_size = 0.3)

from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import *
%matplotlib inline

rmse_val = [] #to store rmse values for different k
k_val = []
for K in range(50):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(xtrain, ytrain)  #fit the model
    pred=model.predict(xtest) #make prediction on test set
    error = sqrt(mean_squared_error(ytest, pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
   
    
from sklearn.model_selection import GridSearchCV

# params = {'n_neighbors': l}
params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]}

knn = neighbors.KNeighborsRegressor()

model = GridSearchCV(knn, params, cv=3)
model.fit(xtrain, ytrain)

knn = neighbors.KNeighborsRegressor(n_neighbors = model.best_params_['n_neighbors'])

knn.fit(xtrain, ytrain)
predict = knn.predict(xtest)

import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(ytest, predict), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(ytest, predict), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(ytest, predict), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(ytest, predict), 2)) 
print("R squared score =", round(sm.r2_score(ytest, predict), 2))


pickle.dump(knn, open("model.pkl", "wb"))
    
    
    
    
    
    
    
    
    
    
    
    