from django.shortcuts import render,redirect
from .models import Stock
from django.contrib import messages
from .forms import StockForm
# from lstm import get_future_prices


def home(request):
    import yfinance as yf
    import pandas as pd

    if request.method == 'POST':
        ticker = request.POST['ticker']
        stock = yf.Ticker(ticker)
        key_info = ['shortName','symbol','previousClose ','marketCap','fiftyTwoWeekLow','fiftyTwoWeekHigh','52WeekChange','sector']
        info = {key: value for key, value in stock.info.items() if key in key_info}
        TIME_PERDIOD = '5y'
        stockData = stock.history(TIME_PERDIOD) 
        stockData.reset_index(inplace=True)
        stockData['Date'] = stockData['Date'].dt.strftime("%Y-%m-%d")
        stockData = stockData.drop(['Dividends', "Stock Splits"], axis=1)

        dates = [str(date) for date in list(stockData['Date'])]
        
        history = {
            'date': dates,
            'open':list(stockData['Open']),
            'high':list(stockData['High']),
            'low':list(stockData['Low']),
            'close':list(stockData['Close']),
            'volume':list(stockData['Volume']),
        }
        # print((history['open']))
        return render(request, 'home.html',{'stock':info,'history':history})
    else:
        return render(request, 'home.html',{'ticker':'Enter a Ticker Symbol Above'})

    # TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN","META"]
    # TIME_PERDIOD = '2y'
    # stock = yf.Ticker(TICKERS[0]) # APPLE
    # stockData = stock.history(TIME_PERDIOD)

    # stockData = stockData.drop(['Dividends', "Stock Splits"], axis=1)
    

def about(request):
    return render(request, 'about.html', {})

def add_stock(request):
    import yfinance as yf
    import pandas as pd



    stock_model_tickers = ['AAPL','MSFT','GOOG','META','AMZN']
    sectors = ['Technology']
    industries = ['Consumer Electronics','Internet Content & Information','Internet Retail']
    if request.method == 'POST':
        form = StockForm(request.POST or None)

        if form.is_valid():
            form.save()
            messages.success(request, ("Stock has Been Added"))
            return redirect('add_stock')
    else:
        

        ticker = Stock.objects.all()

        # print(ticker)
        output = dict()
        future_prices = dict()
        dates = list()
        str_future_prices=dict()

        loaded_prices = load_stock_predictions('tech')
        print(f'Supported Stocks = {loaded_prices.keys()}')
        for ticker_item in ticker:
            # check if predictions are availible for ticker
            ticker_string = str(ticker_item).upper()
            if ticker_string in loaded_prices.keys():
                print(f'Loading {ticker_string}')
                future_prices[ticker_string] = loaded_prices[ticker_string]
                str_future_prices[ticker_string] = [str(i) for i in future_prices[ticker_string]]
                
            elif ticker_string not in loaded_prices.keys() and ticker_string in stock_model_tickers:
                print(f'Modeling {ticker_string}')
                future_prices[ticker_string] = [round(i,3) for i in get_future_prices(ticker_string).flatten()][0:70]
                str_future_prices[ticker_string] = [str(i) for i in future_prices[ticker_string]]    
            
                

            info = yf.Ticker(ticker_string)
            if info.info['industry'] in industries or info.info['sector'] in sectors:
                output[ticker_item.id] = info.info
        # save predictions make        
        save_stock_predictions('tech',str_future_prices)

        dates = [str(i)[0:10] for i in pd.bdate_range(start='05/08/2023', end='09/04/2023')][0:70]

        return render(request, 'add_stock.html', {'ticker':ticker,'output':output,'future_prices':future_prices,'dates':dates})


def motor_stock(request):
    import yfinance as yf 
    import pandas as pd

    stock_model_tickers = ["TSLA","TM", "MBGYY","F","GM","HMC"]
    # stock_model_tickers = ["HMC"]
    industries = ['Auto Manufacturers']
    
    if request.method == 'POST':
        form = StockForm(request.POST or None)

        if form.is_valid():
            form.save()
            messages.success(request, ("Stock has been added"))
            return redirect('motor_stock')
    else:
        ticker = Stock.objects.all()

        # print(ticker)
        output = dict()
        future_prices = dict()
        dates = list()
        str_future_prices=dict()

        loaded_prices = load_stock_predictions('motor')
        print(f'Supported Stocks = {loaded_prices.keys()}')
        for ticker_item in ticker:
            # check if predictions are availible for ticker
            ticker_string = str(ticker_item).upper()
            print(f'Supported Stocks = {loaded_prices.keys()}')
            if ticker_string in loaded_prices.keys():
                future_prices[ticker_string] = loaded_prices[ticker_string]
                print(f'Loading {ticker_string}')
                str_future_prices[ticker_string] = [str(i) for i in future_prices[ticker_string]]
            elif ticker_string not in loaded_prices.keys() and ticker_string in stock_model_tickers:
                print(f'Modeling {ticker_string}')
                future_prices[ticker_string] = [round(i,3) for i in get_future_prices(ticker_string).flatten()][0:70]
                
                str_future_prices[ticker_string] = [str(i) for i in future_prices[ticker_string]]
                

            info = yf.Ticker(ticker_string)
            if info.info['industry'] in industries:
                output[ticker_item.id] = info.info
        # save predictions make        
        save_stock_predictions('motor',str_future_prices)

        dates = [str(i)[0:10] for i in pd.bdate_range(start='05/08/2023', end='09/04/2023')][0:70]
			

        return render(request, 'motor_stock.html', {'ticker':ticker,'output':output,'future_prices':future_prices,'dates':dates})


def pharma_stock(request):
    import yfinance as yf 

    import pandas as pd
    
    stock_model_tickers = ["AZN","RHHBY","NVS","BAYRY","SNY","PFE"]
    industries = ['Drug Manufacturers—General']
        
    if request.method == 'POST':
        form = StockForm(request.POST or None)

        if form.is_valid():
            form.save()
            messages.success(request, ("Stock has been added"))
            return redirect('pharma_stock')
    else:
        ticker = Stock.objects.all()

        # print(ticker)
        output = dict()
        future_prices = dict()
        dates = list()
        str_future_prices=dict()

        loaded_prices = load_stock_predictions('pharma')
        print(f'Supported Stocks = {loaded_prices.keys()}')
        for ticker_item in ticker:
            # check if predictions are availible for ticker

            ticker_string = str(ticker_item).upper()
            
            if ticker_string in loaded_prices.keys():
                future_prices[ticker_string] = loaded_prices[ticker_string]
                print(f'Loading {ticker_string}')
                str_future_prices[ticker_string] = [str(i) for i in future_prices[ticker_string]]
            elif ticker_string not in loaded_prices.keys() and ticker_string in stock_model_tickers:
                print(f'Modeling {ticker_string}')
                future_prices[ticker_string] = [round(i,3) for i in get_future_prices(ticker_string).flatten()][0:70]
            
                str_future_prices[ticker_string] = [str(i) for i in future_prices[ticker_string]]
                

            info = yf.Ticker(ticker_string)
            if info.info['industry'] in industries:
                output[ticker_item.id] = info.info
        # save predictions make        
        save_stock_predictions('pharma',str_future_prices)

        dates = [str(i)[0:10] for i in pd.bdate_range(start='05/08/2023', end='09/04/2023')][0:70]

        return render(request, 'pharma_stock.html', {'ticker':ticker,'output':output,'future_prices':future_prices,'dates':dates})


def delete(request, stock_id, page):
    item = Stock.objects.get(pk=stock_id)
    item.delete()
    messages.success(request,(f'{item} Stock removed From Portfolio'))
    if page =='motor':
        return redirect(motor_stock)
    elif page == 'pharma':
        return redirect(pharma_stock)
    elif page == 'technology':
        return redirect(add_stock)
    # return redirect(add_stock)

def stock_model(request):
    return render(request,'stock_model.html',{})

def load_stock_predictions(category):
    import json

    PATH = 'C:\\Users\\shahi\\Documents\\dev\\stock-prediction-app\\stocks\\quotes\\stock_models'
    with open(f'{PATH}\\{category}_stocks_predictions.json', 'r') as f:
        # Load the JSON data into a dictionary
        data = json.load(f)
    for key in data:
        decimal = [float(i) for i in data[key]]
        data[key] = decimal
    # Now the data is a dictionary containing the contents of the JSON file

    return data

def save_stock_predictions(category,data):
    import json
    
    PATH = 'C:\\Users\\shahi\\Documents\\dev\\stock-prediction-app\\stocks\\quotes\\stock_models'
    

    # Open the JSON file for writing
    with open(f'{PATH}\\{category}_stocks_predictions.json', 'w') as f:
        # Write the dictionary to the JSON file
        json.dump(data, f)
    print("Saved Stock models")

def get_future_prices(tickerSymbol:str):

    import yfinance as yf
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import load_model
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


    PATH = 'C:\\Users\\shahi\\Documents\\dev\\stock-prediction-app\\stocks\\quotes\\upgraded_stock_models'
    model = load_model(f'{PATH}\\{tickerSymbol.upper()}_model.h5')

    n_steps = 1000

    # Predict future prices
    future_prices = []
    last_n_steps = data_scaled[-n_steps:]
    for i in range(90):  # predict 30 days in the future
        future_price = model.predict(last_n_steps.reshape(1, n_steps, 2),verbose=None)
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