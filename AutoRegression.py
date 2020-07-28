# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:53:33 2020

@author: prana
"""

#https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# create own auto-regression model (using Decision trees)


data = pd.read_csv("train.csv",header=0,index_col=0)
test_data = pd.read_csv("test.csv")

sat1_x = data.iloc[958:3066,[0,2,8]]

x = np.array(sat1_x['epoch'])
y = np.array(sat1_x['x'] - sat1_x['x_sim'])

plt.plot(x,y)

# cheack for the correlation between the variables to determine the order of the 

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(sat1_x['x'] - sat1_x['x_sim'])

from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt
# load dataset
X = (sat1_x['x']-sat1_x['x_sim']).values
train, test = X[1:len(X)-100], X[len(X)-100:]
# train autoregression
model = AutoReg(train, lags=100)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
