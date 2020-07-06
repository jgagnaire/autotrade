from datetime import datetime
import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import utils

def get_metrics_list(y_true, y_pred, decimals=3):
    mse = round(mean_squared_error(y_true, y_pred), decimals)
    sd_rmse = round(utils.stdev_root_mean_squared_error(y_true, y_pred), decimals)
    mae = round(mean_absolute_error(y_true, y_pred), decimals)
    mape = round(utils.mean_absolute_percentage_error(y_true, y_pred), decimals)
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

num_days_to_predict = 7
symbol = 'ETL.PA'

# Retrieve and preprocess market data for a given stock
X_df = utils.get_stock_feature_dataset(symbol)

# Build target variable dataset
y_df = pd.DataFrame()
y_cols = []
for i in range(1, num_days_to_predict + 1):
    colname = 'AdjClose_D+' + str(i)
    y_df[colname] = X_df['Adj Close'].shift(periods=-i)
    y_cols.append(colname)
y_df.columns = y_cols
# Drop rows with shifted NaN values
y_df.drop(y_df.tail(i).index, inplace=True)
predict_set = X_df.iloc[[-1]].copy()
X_df.drop(X_df.tail(i).index, inplace=True)

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
