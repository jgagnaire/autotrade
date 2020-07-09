from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import os
import requests
import ta
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

def get_last_close_timestamp():
    # If today is a weekend day, return today's timestamp
    current_time = datetime.today()
    weekno = current_time.weekday()
    if weekno >= 5:
        return int(current_time.timestamp())

    # If market is currently opened, return yesterday's post-close timestamp
    today_market_close = current_time.replace(hour=18, minute=0, second=0, microsecond=0)
    if current_time < today_market_close:
        yesterday_market_close = (current_time - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
        return int(yesterday_market_close.timestamp())

    # Otherwise return today's post-close timestamp
    return int(today_market_close.timestamp())

def download_stock_data(symbol, start=None, end=None):
    filepath = os.path.join(os.getcwd(), 'historical-data_{}.csv'.format(symbol))
    if not start:
        start = 636278400 # oldest CAC40 quotation date - March 1st 1990 8am
    if not end:
        end = get_last_close_timestamp()
    url = 'http://query1.finance.yahoo.com/v7/finance/download/{}?period1={:d}&period2={:d}&interval=1d&events=history'.format(symbol, int(start), int(end))
    response = requests.get(url)
    with open(filepath, 'wb') as f:
        f.write(response.content)
    try:
        return pd.read_csv(filepath, index_col=0, parse_dates=True, infer_datetime_format=True)
    except:
        print('Error fetching URL {}:'.format(url))
        print(response.content)
        raise

def get_stock_feature_dataset(symbol, start=None, end=None):
    # Download stock historical data in a DataFrame
    df = download_stock_data(symbol, start, end)

    # Add technical analysis indicators as features
    df = ta.add_all_ta_features(df, open='Open', high='High', low='Low', close='Close', volume='Volume')

    # Add historical data for major indices as features
    if not start:
        start = datetime.timestamp(df.index[0])
    if not end:
        end = datetime.timestamp(df.index[-1])
    cac40_df = download_stock_data('^FCHI', start, end)
    sbf120_df = download_stock_data('^SBF120', start, end)
    # Drop 'Volume' and 'Adj Close' features, meaningless regarding market indices
    drop_index_cols = ['Volume', 'Adj Close']
    if not cac40_df.empty:
        cac40_df.drop(drop_index_cols, axis=1, inplace=True)
        # Prefix columns with index name
        prefixed_cac_cols = []
        for cac_col in cac40_df.columns:
            prefixed_cac_cols.append('cac40_' + cac_col)
        cac40_df.columns = prefixed_cac_cols
        # Add them to the main dataset
        df = pd.concat([df, cac40_df], axis=1)
    # Do the same with SBF120
    if not sbf120_df.empty:
        sbf120_df.drop(drop_index_cols, axis=1, inplace=True)
        prefixed_sbf_cols = []
        for sbf_col in sbf120_df.columns:
            prefixed_sbf_cols.append('sbf120_' + sbf_col)
        sbf120_df.columns = prefixed_sbf_cols
        df = pd.concat([df, sbf120_df], axis=1)

    # Fill inf values with NaNs, and then NaNs with interpolated values
    df = df.astype(float)
    df.replace(np.inf, np.nan, inplace=True)
    df.replace(-np.inf, np.nan, inplace=True)
    df.interpolate(axis=0, limit_direction='both', inplace=True)
    return df

def make_labels_dataset(X_df, increase=0.05, label_name='increase_tomorrow'):
    '''
        increase: float between 0 and 1, equivalent to the desired % increase when multiplied by 100
        label_name: name for the column containing labels
    '''

    # Build the target dataset: label 1 if stock price increased by 5% or more in the following days, 0 otherwise
    y_df = pd.DataFrame(index=X_df.index, columns=[label_name])
    for i in range(len(X_df) - 1):
        increase_threshold = X_df['Adj Close'].iloc[i] + increase * X_df['Adj Close'].iloc[i]
        y_df.iloc[i] = 1 if X_df['Adj Close'].iloc[i+1] > increase_threshold else 0

    # Drop last row, for which there is no label
    X_df.drop(X_df.tail(1).index, inplace=True)
    y_df.drop(y_df.tail(1).index, inplace=True)
    return X_df, y_df

# This function returns the Root Mean Squared Error, normalized by Standard-Deviation
def stdev_root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) / np.std(y_true)

# This one returns the Mean Absolute Percentage Error, normalized by the true values
# and expressed as a percentage
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def split_dataset(X_df, y_df, train_size=0.9, do_shuffle=False):
    if do_shuffle:
        X_df, y_df = shuffle(X_df, y_df)
    train_data_size = int(len(X_df) * train_size)
    train_X, test_X = X_df.iloc[0:train_data_size].astype(float), X_df.iloc[train_data_size:-1].astype(float)
    train_y, test_y = y_df.iloc[0:train_data_size].astype(float), y_df.iloc[train_data_size:-1].astype(float)
    return train_X, train_y, test_X, test_y