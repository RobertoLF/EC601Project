#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:12:31 2021

@author: rluisfue
"""
import requests 
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA

#Function that returns a dictionary containing: 
#1.'Price' = The hourly high price for the crypto specified over the past 90 days.
#2.'Dates' = A list of the dates, one for each datapoint was taken
#3.'xAxis' = x-scale for plotting the price history. 
def getPrice(symbol):
    apiKeyFile = open(r"/Users/rluisfue/spyder-py3/EC601/CryptoCompareAPIKey.txt")
    apiKey = apiKeyFile.read()
    limit = '216'
    dataPoints = 0
    firstTime = True
    high = []
    time = []
    xAxis = []
    count = 0
    for x in range(0,10):
        global toTs
        count+=1
        if firstTime:
            URL = '''https://min-api.cryptocompare.com/data/v2/histohour?fsym=%s&tsym=USD&limit=%s&api_key={%s}'''%(symbol,limit,apiKey)
            firstTime = False
        else:
            URL = '''https://min-api.cryptocompare.com/data/v2/histohour?fsym=%s&tsym=USD&limit=%s&toTs=%s&api_key={%s}'''%(symbol,limit,toTs,apiKey)
        response = requests.get(URL)
        inspect = response.json()
        data = inspect['Data']['Data']
        toTs = str(data[0]['time'])
        xAxis.append(datetime.utcfromtimestamp(data[0]['time']).strftime('%Y-%m-%d'))
        data.reverse()
        if count == 1:
            xAxis.append(datetime.utcfromtimestamp(data[0]['time']).strftime('%Y-%m-%d'))
        for x in data:
            dataPoints+=1
            high.append(x['high'])
            time.append(datetime.utcfromtimestamp(x['time']).strftime('%Y-%m-%d'))

    high.reverse()
    result = {'Dates':time,'Price':high,'xAxis':xAxis}
    return result

data = getPrice('QNT')
dates = data['Dates']
xAxis = data['xAxis']

#decompose price history time series data.
timeSeriesDecomposed = seasonal_decompose(data['Price'],model='multiplicative',period=24)
timeSeriesDecomposed.plot()

#obtain different compnents of time series.
npresid = timeSeriesDecomposed.resid
nptrend = timeSeriesDecomposed.trend
npseasonal = timeSeriesDecomposed.seasonal
residual = npresid.tolist()
trend = nptrend.tolist()
seasonal = npseasonal.tolist()

#remove NANs produced by seasonal_decompose()
index = 0
cleanResidual = []
cleanTrend = []
cleanSeasonal = []
cleanDates = []
for x in residual:
    if ~np.isnan(x):
        cleanResidual.append(residual[index])
        cleanTrend.append(trend[index])
        cleanSeasonal.append(seasonal[index])
        cleanDates.append(dates[index])
    index+=1

#plot partial autocorrelation
pacf = plot_pacf(cleanResidual,lags=20,method='ywm')

#fit AR(10) model and print model fit summary.
model_fit = ARIMA(cleanResidual,order=(10,0,0)).fit()
print(model_fit.summary())

