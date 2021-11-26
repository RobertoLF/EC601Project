#%%
# import libraries
import requests
import pandas as pd
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM

#%% Get results from cryptocompare API

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

url = "https://min-api.cryptocompare.com/data/histoday"


# Define API key, fysm = symbol(s), tsym = currency type, limit = # of time units to pull data for 
payload = {
    "api_key": apiKey,
    "fsym": "BTC",
    "tsym": "USD",
    "limit": 180   
}

# Creates dictionary of all results
result = requests.get(url, params=payload).json()

#%% # Create dataframe of data for manipulation purposes

df = pd.DataFrame(result['Data'])
df = df.set_index('time')
df.index = pd.to_datetime(df.index, unit='s')
df.drop(["conversionType", "conversionSymbol"], axis = 'columns', inplace = True)
target_col = 'close'
print(df.head())

#%% Plotting

# Plotting dataframe
df.plot(y='close')
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("Historic Price")
#plt.xticks(rotation = 45)
plt.show()


#%% Model Training and Testing

# Train and test splits (80%-20%)
def train_test_split(df, test_size=0.2):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data

train, test = train_test_split(df, test_size=0.2)

# Plot data with train and test splits on graph
def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16);

line_plot(train[target_col], test[target_col], 'training', 'test', title='')


# Normalize data for better machine learning results
def normalise_zero_base(df):
    return df / df.iloc[0] - 1

def normalise_min_max(df):
    return (df - df.min()) / (df.max() - df.min())


# ??
def extract_window_data(df, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)


# Prepare data to be fed into neural network
def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):
    train_data, test_data = train_test_split(df, test_size=test_size)
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    y_train = train_data[target_col][window_len:].values
    y_test = test_data[target_col][window_len:].values
    if zero_base:
        y_train = y_train / train_data[target_col][:-window_len].values - 1
        y_test = y_test / test_data[target_col][:-window_len].values - 1

    return train_data, test_data, X_train, X_test, y_train, y_test


def build_lstm_model(input_data, output_size, neurons=100, activ_func='linear',
                     dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

#%%
np.random.seed(42)
window_len = 5
test_size = 0.2
zero_base = True
lstm_neurons = 100
epochs = 20
batch_size = 32
loss = 'mse'
dropout = 0.2
optimizer = 'adam'


train, test, X_train, X_test, y_train, y_test = prepare_data(
    df, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)


model = build_lstm_model(
    X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
    optimizer=optimizer)
history = model.fit(
    X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

targets = test[target_col][window_len:]
preds = model.predict(X_test).squeeze()
mean_absolute_error(preds, y_test)


preds = test[target_col].values[:-window_len] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)
line_plot(targets, preds, 'actual', 'prediction', lw=3)