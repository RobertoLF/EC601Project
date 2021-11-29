#%% Imports

import requests
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


#%% CryptoCompare API

# define API key
apiKey = open(r"C:\Users\bobby\Documents\Education\Boston University\EC601\Main Project\cryptocompare_key.txt").read()


# urls for various purposes
'''
https://min-api.cryptocompare.com/data/price             Single symbol price
https://min-api.cryptocompare.com/data/pricemulti        Multiple symbol price
https://min-api.cryptocompare.com/data/pricemultifull    Multiple symbol full data
https://min-api.cryptocompare.com/data/v2/histoday       Daily historical price
https://min-api.cryptocompare.com/data/v2/histohour      Hourly historical price
https://min-api.cryptocompare.com/data/v2/histominute    Minute historical price

'''

url = "https://min-api.cryptocompare.com/data/histohour"

coin = "SOL"

# Define API key, fysm = symbol(s), tsym = currency type, limit = # of time units to pull data for 
payload = {
    "api_key": apiKey,
    "fsym": coin,
    "tsym": "USD",
    "limit": 2000 
}

# Creates dictionary of all results
result = requests.get(url, params=payload).json()


#%% Data in Pandas Dataframe

df = pd.DataFrame(result['Data'])
#df = df.set_index('time')
#df.index = pd.to_datetime(df.index, unit='s')
df['time'] = pd.to_datetime(df['time'],unit='s')
df.drop(["conversionType", "conversionSymbol"], axis = 'columns', inplace = True)
target_col = 'close'
df.drop(columns=['open', 'high', 'low', 'volumefrom','volumeto'], inplace=True)
print(df.head())

#%% Create close data series, and create train and test splits 80-20
close_data = df['close'].values
close_data = close_data.reshape((-1,1))

split_percent = 0.80
split = int(split_percent*len(close_data))

close_train = close_data[:split]
close_test_1 = close_data[split:]
close_test_2 = close_data[split-15:]

date_train = df['time'][:split]
date_test = df['time'][split:]


#%%
look_back = 15

train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)     
test_generator = TimeseriesGenerator(close_test_2, close_test_2, length=look_back, batch_size=1)

#%% Build Model

model = Sequential()
model.add(
    LSTM(10,
        activation='relu',
        input_shape=(look_back,1))
)
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

num_epochs = 25
model.fit_generator(train_generator, epochs=num_epochs, verbose=1)


#%% Modeling fitting on Test data

# set plot resolution
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

prediction = model.predict(test_generator)
close_train = close_train.reshape((-1))
close_test_1 = close_test_1.reshape((-1))
prediction = prediction.reshape((-1))

MAE = mean_absolute_error(prediction, close_test_1)
MAPE = mean_absolute_percentage_error(prediction, close_test_1)*100

# Calculating MAE manually to verify answers match
#difference = np.absolute(prediction-close_test_1)
#error_avg = difference.mean()

# Model prediction plotting entire test set range
plt.plot(date_test, close_test_1, label="test",color = 'blue')
plt.plot(date_test, prediction, label="prediction",color = 'red')
plt.ylabel("{} Price (USD)".format(coin))
plt.title("Test Set Predictions (Entire Range)")
plt.xticks(rotation = 45)
plt.text(0.4,0.95, 'MAE = {:.2f}'.format(MAE), transform=plt.gca().transAxes,fontsize=9)
plt.text(0.4,0.90, 'MAPE = {:.3f}%'.format(MAPE), transform=plt.gca().transAxes,fontsize=9)
plt.legend()
plt.show()


# Model prediction plotting only portion of range
plt.plot(date_test, close_test_1, label="test",color = 'blue')
plt.plot(date_test, prediction, label="prediction",color = 'red')
plt.xlim(pd.Timestamp('2021-11-15 10:00:00'), pd.Timestamp('2021-11-19 10:00:00'))
plt.ylabel("{} Price (USD)".format(coin))
plt.title("Test Set Predictions (Zoomed in)")
plt.xticks(rotation = 45)
plt.text(0.4,0.95, 'MAE = {:.2f}'.format(MAE), transform=plt.gca().transAxes,fontsize=9)
plt.text(0.4,0.90, 'MAPE = {:.3f}%'.format(MAPE), transform=plt.gca().transAxes,fontsize=9)
plt.legend()
plt.show()

# Mean absolute error (artichmetic average of absolute errors)
# Error will be in $USD



# #%% Forecasting

# hours_back = 24
# window = 24
# new_data_close = close_data[:-window]
# new_data_time = df['time'][:-window]
# close_data_forecast = new_data_close[:-hours_back]
# actual = new_data_close[-(hours_back+1):]
# close_data_forecast = close_data_forecast.reshape((-1))

# def predict(num_prediction, model):
#     prediction_list = close_data_forecast[-look_back:]
    
#     for _ in range(num_prediction):
#         x = prediction_list[-look_back:]
#         x = x.reshape((1, look_back, 1))
#         out = model.predict(x)[0][0]
#         prediction_list = np.append(prediction_list, out)
#     prediction_list = prediction_list[look_back-1:]
        
#     return prediction_list
    
# def predict_dates(num_prediction):
#     last_date = new_data_time.values[-hours_back]#df['time'].values[-hours_back]
#     print(last_date)
#     prediction_dates = pd.date_range(last_date, periods=num_prediction+1, freq = 'H').tolist()
#     return prediction_dates

# num_prediction = 24
# forecast = predict(num_prediction, model)
# forecast_dates = predict_dates(num_prediction)



# # Plotting dataframe
# plt.plot(new_data_time[-72:],new_data_close[-72:],label='actual',color = 'blue')

# plt.plot(forecast_dates,forecast,label="forecast",color = 'red')
# #plt.plot(forecast_dates,actual,label="actual")
# plt.ylabel("Price (USD)")
# plt.title("Three Day Window")
# plt.xticks(rotation = 45)
# plt.legend()
# plt.show()





# plt.plot(forecast_dates,forecast,label="forecast",color='red')
# plt.plot(forecast_dates,actual,label="actual",color='blue')
# plt.ylabel("Price (USD)")
# plt.title("One Day Window")
# plt.xticks(rotation = 45)
# plt.legend()
# plt.show()