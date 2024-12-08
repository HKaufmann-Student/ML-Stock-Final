import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from collections import Counter
import ta

def add_technical_indicators(df):
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['Stoch_RSI'] = ta.momentum.StochRSIIndicator(close=df['Close'], window=14).stochrsi()
    df['TSI'] = ta.momentum.TSIIndicator(close=df['Close'], window_slow=25, window_fast=13).tsi()

    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    cci = ta.trend.CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=20).cci()
    df['CCI'] = cci

    dpo = ta.trend.DPOIndicator(close=df['Close'], window=20).dpo()
    df['DPO'] = dpo

    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()

    atr = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ATR'] = atr.average_true_range()

    ulcer = ta.volatility.UlcerIndex(close=df['Close'], window=14).ulcer_index()
    df['Ulcer_Index'] = ulcer

    obv = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    df['OBV'] = obv

    cmf = ta.volume.ChaikinMoneyFlowIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=20).chaikin_money_flow()
    df['CMF'] = cmf

    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_10'] = df['Close'].ewm(span=10).mean()

    return df

def add_performance_labels(df, n=5, k=3):
    df['Performance'] = (df['Close'].shift(-n) - df['Close']) / df['Close']
    df['Top_Performer'] = (df.groupby('Date')['Performance'].rank(ascending=False, method='first') <= k).astype(int)
    return df

def add_time_features(df):
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    return df

def merge_with_macro(df, macro_df):
    df = pd.merge(df, macro_df, on='Date', how='left')
    df.sort_values('Date', inplace=True)
    df.ffill(axis=0, inplace=True)
    df.bfill(axis=0, inplace=True)
    return df

def prepare_dataset(df, tickers, window_size=60):
    label_encoder = LabelEncoder()
    df['Ticker_encoded'] = label_encoder.fit_transform(df['Ticker'])

    features = [
        'Close', 'SMA_10', 'EMA_10', 'RSI', 'Stoch_RSI', 'TSI', 'MACD', 'MACD_Signal',
        'CCI', 'DPO', 'BB_High', 'BB_Low', 'ATR', 'Ulcer_Index', 'OBV', 'CMF',
        'SPY_Close', 'VIX_Close', 'Year', 'Month', 'DayOfWeek', 'Ticker_encoded'
    ]

    # Drop rows with missing values in features or target
    df = df.dropna(subset=features + ['Top_Performer']).copy()

    # 1. Fit the scaler globally on the entire dataset
    scaler = MinMaxScaler()
    scaler.fit(df[features])

    X_list, y_list = [], []

    # 2. For each ticker, transform and create sequences
    for ticker in tickers:
        ticker_encoded = label_encoder.transform([ticker])[0]
        ticker_data = df[df['Ticker_encoded'] == ticker_encoded].sort_values('Date')
        scaled_data = scaler.transform(ticker_data[features])
        y_values = ticker_data['Top_Performer'].values

        for i in range(window_size, len(scaled_data)):
            X_list.append(scaled_data[i - window_size:i])
            y_list.append(y_values[i])

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y, scaler, label_encoder, features