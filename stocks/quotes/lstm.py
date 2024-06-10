import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

def get_future_prices(tickerSymbol:str):


    # Get data on the ticker
    tickerData = yf.Ticker(tickerSymbol)

    # Get the historical prices for this ticker
    tickerDf = tickerData.history(period='1d', start='2015-01-01')

    # Calculate the EMA with a window of 20 days
    ema = tickerDf['Close'].ewm(span=20, adjust=False).mean()

    # Merge the EMA with the original DataFrame
    tickerDf = pd.concat([tickerDf, ema.rename('EMA')], axis=1)

    # Define the feature and target variables
    data = tickerDf[['Close', 'EMA']]
    target = tickerDf['Close']

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)



    model = load_model(f'{tickerSymbol.upper()}_lstm_model.h5')

    n_steps = 365

    # Predict future prices
    future_prices = []
    last_n_steps = data_scaled[-n_steps:]
    for i in range(90):  # predict 30 days in the future
        future_price = model.predict(last_n_steps.reshape(1, n_steps, 2))
        future_prices.append(future_price[0, 0])
        last_n_steps = np.append(last_n_steps[1:], [[future_price[0, 0], future_price[0, 0]]], axis=0)

    # # Denormalize the predicted prices
    # future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))
    # Define the scaler object with the same feature range as before
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit the scaler to the original target variable
    scaler.fit(tickerDf['Close'].values.reshape(-1, 1))

    # Denormalize the predicted prices using the scaler
    future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))
    # Print the predicted prices
    return future_prices



    


