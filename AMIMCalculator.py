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

def cleanResidual(residual):
    #remove NANs produced by seasonal_decompose()
    index = 0
    cleanResidual = []
    for x in residual:
        if ~np.isnan(x):
            cleanResidual.append(residual[index])
        index+=1
    return cleanResidual
    
BTCdata = getPrice('BTC')
ETHdata = getPrice('ETH')
BNBdata = getPrice('BNB')
ADAdata = getPrice('ADA')
SOLdata = getPrice('SOL')

#decompose price history time series data.
btcDecomposed = seasonal_decompose(BTCdata['Price'],model='multiplicative',period=24)
ethDecomposed = seasonal_decompose(ETHdata['Price'],model='multiplicative',period=24)
bnbDecomposed = seasonal_decompose(BNBdata['Price'],model='multiplicative',period=24)
adaDecomposed = seasonal_decompose(ADAdata['Price'],model='multiplicative',period=24)
solDecomposed = seasonal_decompose(SOLdata['Price'],model='multiplicative',period=24)

#obtain stationary residual series.
BTCresidual = cleanResidual(btcDecomposed.resid.tolist())
ETHresidual = cleanResidual(ethDecomposed.resid.tolist())
BNBresidual = cleanResidual(bnbDecomposed.resid.tolist())
ADAresidual = cleanResidual(adaDecomposed.resid.tolist())
SOLresidual = cleanResidual(solDecomposed.resid.tolist())

#fit AR(10) model.
BTCmodel = ARIMA(BTCresidual,order=(10,0,0)).fit()
ETHmodel = ARIMA(ETHresidual,order=(10,0,0)).fit()
BNBmodel = ARIMA(BNBresidual,order=(10,0,0)).fit()
ADAmodel = ARIMA(ADAresidual,order=(10,0,0)).fit()
SOLmodel = ARIMA(SOLresidual,order=(10,0,0)).fit()

#Retreive autocorrelation coeffecients and format into matrix
BTCcoeffecients = np.delete(getattr(BTCmodel,'polynomial_ar'),[0])
ETHcoeffecients = np.delete(getattr(ETHmodel,'polynomial_ar'),[0])
BNBcoeffecients = np.delete(getattr(BNBmodel,'polynomial_ar'),[0])
ADAcoeffecients = np.delete(getattr(ADAmodel,'polynomial_ar'),[0])
SOLcoeffecients = np.delete(getattr(SOLmodel,'polynomial_ar'),[0])

coefList = []
coefList.extend(BTCcoeffecients)
coefList.extend(ETHcoeffecients)
coefList.extend(BNBcoeffecients)
coefList.extend(ADAcoeffecients)
coefList.extend(SOLcoeffecients)

coefArray = np.array(coefList)

coefMatrix = np.reshape(coefArray,(5,10))

#Normalize autocorrelation coeffecients 
covMatrix = np.cov(coefMatrix)
lInverse = np.linalg.inv(np.linalg.cholesky(covMatrix))
standardizedCoeff = np.dot(lInverse,coefMatrix)

#Calculate magnitude of market inefficiency (MIM) for each crypto 
BTCcoefSum = 0
ETHcoefSum = 0
BNBcoefSum = 0
ADAcoefSum = 0
SOLcoefSum = 0

for x in range(0,10):
    BTCcoefSum = BTCcoefSum + abs(standardizedCoeff[0,x])
    ETHcoefSum = ETHcoefSum + abs(standardizedCoeff[1,x])
    BNBcoefSum = BNBcoefSum + abs(standardizedCoeff[2,x])
    ADAcoefSum = ADAcoefSum + abs(standardizedCoeff[3,x])    
    SOLcoefSum = SOLcoefSum + abs(standardizedCoeff[4,x])

btcMIM = BTCcoefSum / (1 + BTCcoefSum)
ethMIM = ETHcoefSum / (1 + ETHcoefSum)
bnbMIM = BNBcoefSum / (1 + BNBcoefSum)
adaMIM = ADAcoefSum / (1 + ADAcoefSum)
solMIM = SOLcoefSum / (1 + SOLcoefSum)

#Calculate the adjusted magnitude of market inefficiency (AMIM) for each crypto 
CI = 0.9184596

btcAMIM = (btcMIM - CI) / (1 - CI)
ethAMIM = (ethMIM - CI) / (1 - CI)
bnbAMIM = (bnbMIM - CI) / (1 - CI)
adaAMIM = (adaMIM - CI) / (1 - CI)
solAMIM = (solMIM - CI) / (1 - CI)

print(btcAMIM)
print(ethAMIM)
print(bnbAMIM)
print(adaAMIM)
print(solAMIM)