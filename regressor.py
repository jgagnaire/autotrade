import os
import requests
from datetime import datetime, timedelta
import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

# Retrieve and preprocess market data for a given stock
def get_stock_dataset(symbol, num_days_to_predict=7, make_predict_set=False):
    # Download stock historical data in a DataFrame
    df = download_stock_data(symbol)

    # Add technical analysis indicators as features
    df = ta.add_all_ta_features(df, open='Open', high='High', low='Low', close='Close', volume='Volume')

    # Add historical data for major indices as features
    start_timestamp = datetime.timestamp(df.index[0])
    end_timestamp = datetime.timestamp(df.index[-1])
    cac40_df = download_stock_data('^FCHI', start_timestamp, end_timestamp)
    sbf120_df = download_stock_data('^SBF120', start_timestamp, end_timestamp)
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
    df.replace(np.inf, np.nan, inplace=True)
    df.replace(-np.inf, np.nan, inplace=True)
    df.interpolate(axis=0, limit_direction='both', inplace=True)

    # Build target variable dataset
    adjclose_df = pd.DataFrame()
    adjclose_cols = []
    for i in range(1, num_days_to_predict + 1):
        colname = 'AdjClose_D+' + str(i)
        adjclose_df[colname] = df['Adj Close'].shift(periods=-i)
        adjclose_cols.append(colname)
    adjclose_df.columns = adjclose_cols
    # Drop rows with shifted NaN values
    adjclose_df.drop(adjclose_df.tail(i).index, inplace=True)
    predict_set = df.iloc[[-1]].copy() if make_predict_set else None
    df.drop(df.tail(i).index, inplace=True)

    return df, adjclose_df, predict_set

# This function returns the Root Mean Squared Error, normalized by Standard-Deviation
def stdev_root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) / np.std(y_true)

# This one returns the Mean Absolute Percentage Error, normalized by the true values
# and expressed as a percentage
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_metrics_list(y_true, y_pred, decimals=3):
    mse = round(mean_squared_error(y_true, y_pred), decimals)
    sd_rmse = round(stdev_root_mean_squared_error(y_true, y_pred), decimals)
    mae = round(mean_absolute_error(y_true, y_pred), decimals)
    mape = round(mean_absolute_percentage_error(y_true, y_pred), decimals)
    return [mse, sd_rmse, mae, mape]

# Function that trains n models, each predicting adjusted close for day + 1 to n
def train_eval(model_list, train_X, train_y, test_X, test_y, num_days_to_predict=7):
    preds_list = []
    index_names = [str('d+{}'.format(i)) for i in range(1, num_days_to_predict + 1)]
    metrics_df = pd.DataFrame(columns=['MSE', 'SD-RMSE', 'MAE', 'MAPE'], index=index_names)
    for i in range(num_days_to_predict):
        model_list[i].fit(train_X, train_y[:,i])
        preds_list.append(model_list[i].predict(test_X))
        metrics_df.loc['d+{}'.format(i + 1)] = get_metrics_list(test_y[:,i], preds_list[i])
    print(metrics_df)
    return preds_list

def train_predict(model_list, X, y, X_pred, y_scaler, num_days_to_predict=7):
    column_names = [str('d+{}'.format(i)) for i in range(1, num_days_to_predict + 1)]
    predict_df = pd.DataFrame(columns=column_names)
    for i in range(num_days_to_predict):
        model_list[i].fit(X, y[:,i])
        pred_array = np.zeros((len(X_pred), num_days_to_predict))
        pred_array[0] = model_list[i].predict(X_pred)
        unscaled_pred = y_scaler.inverse_transform(pred_array)
        print('Prediction for D+{}: {:.4f}'.format(i + 1, unscaled_pred[0][0]))

# MAIN

# TODO: argument parsing for num_days_to_predict and stock symbol
num_days_to_predict = 1
symbol = 'ETL.PA'

X_df, y_df, predict_set = get_stock_dataset(symbol, num_days_to_predict, make_predict_set=True)

print('')
print('Predicting {} up to d+{} from day {}'.format(symbol, num_days_to_predict, str(predict_set.index.values)))
print('')

# Split dataset into 90-10% training-testing sets:
train_size = int(len(X_df) * 0.90)
train_X, test_X = X_df.iloc[0:train_size], X_df.iloc[train_size:-1]
train_y, test_y = y_df.iloc[0:train_size], y_df.iloc[train_size:-1]

# Normalize datasets
X_scaler = MinMaxScaler().fit(train_X.values)
y_scaler = MinMaxScaler().fit(train_y.values)

train_X_scaled = X_scaler.transform(train_X.values)
train_y_scaled = y_scaler.transform(train_y.values)

test_X_scaled = X_scaler.transform(test_X.values)
test_y_scaled = y_scaler.transform(test_y.values)

X_pred = X_scaler.transform(predict_set.values)

print('')
print('Last day of training set: {}'.format(X_df.index[train_size]))
print('')

# Train and test LinearRegression model
linear_models_list = []
for i in range(num_days_to_predict):
    linear_models_list.append(LinearRegression())
print('')
print('Performance of LinearRegression:')
linear_preds_list = train_eval(linear_models_list, train_X_scaled, train_y_scaled, test_X_scaled, test_y_scaled, num_days_to_predict)
print('')

# Predict values we're interested in
for i, model in enumerate(linear_models_list):
    pred_array = np.zeros((len(X_pred), num_days_to_predict))
    pred_array[0] = model.predict(X_pred)
    unscaled_pred = y_scaler.inverse_transform(pred_array)
    print('Prediction for D+{}: {:.4f}'.format(i + 1, unscaled_pred[0][0]))
print('')

# Train and test KernelRidge model
kr_models_list = []
for i in range(num_days_to_predict):
    kr_models_list.append(KernelRidge())
print('')
print('Performance of KernelRidge on original dataset:')
kr_preds_list = train_eval(kr_models_list, train_X_scaled, train_y_scaled, test_X_scaled, test_y_scaled, num_days_to_predict)
print('')

# Predict values we're interested in
for i, model in enumerate(kr_models_list):
    pred_array = np.zeros((len(X_pred), num_days_to_predict))
    pred_array[0] = model.predict(X_pred)
    unscaled_pred = y_scaler.inverse_transform(pred_array)
    print('Prediction for D+{}: {:.4f}'.format(i + 1, unscaled_pred[0][0]))
print('')
print('')

print('')
print('Now retraining models on complete recent data, future closing prices will be the testing set')
print('Last day of training set: {}'.format(X_df.index[len(X_df) - 1]))
print('')
num_days_train_data = [10, 20, 30, 60, 90, 150, len(X_df)]
for num_days_training in num_days_train_data:
    recent_X_df = X_df.tail(num_days_training)
    recent_y_df = y_df.tail(num_days_training)
    X_scaler = MinMaxScaler().fit(recent_X_df.values)
    y_scaler = MinMaxScaler().fit(recent_y_df.values)
    X_scaled = X_scaler.transform(recent_X_df.values)
    y_scaled = y_scaler.transform(recent_y_df.values)
    print('Training on the last {} days'.format(num_days_training))
    print('\tLinear model:')
    train_predict(linear_models_list, X_scaled, y_scaled, X_pred, y_scaler, num_days_to_predict)
    print('')
    print('\tKernel ridge model:')
    train_predict(kr_models_list, X_scaled, y_scaled, X_pred, y_scaler, num_days_to_predict)
    print('')
    print('')

print('')
print('Last known price:')
print('{}'.format(predict_set.iloc[[-1]]['Adj Close']))
print('')
