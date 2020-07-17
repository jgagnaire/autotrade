import utils
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

def make_labels_dataset(X_df, increase=0.05, label_name='increase_tomorrow'):
    '''
        increase: float between 0 and 1, equivalent to the desired % increase when multiplied by 100
        label_name: name for the column containing labels
    '''

    # Build the target dataset: label 1 if stock price increased by 5% or more in the following days, 0 otherwise
    y_df = pd.DataFrame(index=X_df.index, columns=[label_name])
    for i in range(len(X_df) - 1):
        increase_threshold = X_df['Close'].iloc[i] + increase * X_df['Close'].iloc[i]
        y_df.iloc[i] = 1 if X_df['High'].iloc[i+1] >= increase_threshold else 0

    # Drop last row, for which there is no label
    X_df.drop(X_df.tail(1).index, inplace=True)
    y_df.drop(y_df.tail(1).index, inplace=True)

    return X_df.astype(float), y_df.astype(float)

def print_metrics(y_true, y_pred):
    print('\taccuracy: {:.2f}%'.format(accuracy_score(y_true, y_pred) * 100))
    print('\tprecision: {:.2f}%'.format(precision_score(y_true, y_pred) * 100))
    print('\trecall: {:.2f}%'.format(recall_score(y_true, y_pred) * 100))
    print('\tfbeta: {:.3f}'.format(fbeta_score(y_true, y_pred, beta=0.5)))

def train_eval(model, train_X, train_y, test_X, test_y):
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    print('Results:')
    print_metrics(test_y, pred_y)
    return model

def build_tomorrow_predict_set(symbol_list, X_scaler):
    pred_X = pd.DataFrame()
    for count, symbol in enumerate(symbol_list):
        # Download data for last full market day
        X_df = utils.get_stock_feature_dataset(symbol)
        pred_X = pred_X.append(X_df.tail(1).reset_index(drop=True), ignore_index=True)
        # print progress
        if not count % 5:
            print('Done processing {}! new pred_X shape: {}'.format(symbol, pred_X.shape))

    pred_X = pred_X.astype(float)
    pred_X.replace(np.inf, np.nan, inplace=True)
    pred_X.replace(-np.inf, np.nan, inplace=True)
    pred_X.interpolate(axis=0, limit_direction='both', inplace=True)
    # Scale all values to have the same range:
    pred_X_scaled = X_scaler.transform(pred_X.values)
    return pred_X_scaled

# MAIN

symbol=None

if symbol:  # Predict increase or not for one symbol only
    X_df = utils.get_stock_feature_dataset(symbol)
    X_df, y_df = make_labels_dataset(X_df)

    train_X, test_X, train_y, test_y = train_test_split(X_df, y_df, train_size=0.98)

    print('')
    print('\ttraining set contains {:.2f}% records labeled as 1'.format(train_y.values.sum()/train_y.shape[0] * 100))
    print('\ttesting set contains {:.2f}% records labeled as 1'.format(test_y.values.sum()/test_y.shape[0] * 100))
    print('')

    X_scaler = MinMaxScaler().fit(train_X.values)
    train_X_scaled = X_scaler.transform(train_X.values)
    test_X_scaled = X_scaler.transform(test_X.values)
    train_y = train_y.values.reshape(-1).astype(float)
    test_y = test_y.values.reshape(-1).astype(float)

    print('training in progress...')
    rf_weak_learner = RandomForestClassifier(n_estimators=1000, class_weight='balanced', max_features=30)
    ada_model_rf = AdaBoostClassifier(rf_weak_learner, n_estimators=50)
    model = train_eval(ada_model_rf, train_X_scaled, train_y, test_X_scaled, test_y)

    predict_list = [symbol]

else:  # Predict increase or not for each stock of the global training set
    # Load symbol list
    with open('symbol-list.pickle', 'rb') as f:
        predict_list = pickle.load(f)
    # Load scaler
    with open('x-scaler.pickle', 'rb') as f:
        X_scaler = pickle.load(f)
    # Load model
    with open('RF-clf.pickle', 'rb') as f:
        model = pickle.load(f)

print('Predicting for {}:'.format(predict_list))

pred_X = build_tomorrow_predict_set(predict_list, X_scaler)
pred_y = model.predict(pred_X)

if not pred_y.sum():
    print('No increase predicted tomorrow!')
else:
    print('Increase predicted tomorrow for:')
    for sym_idx, sym in enumerate(symbol_list):
        if y_true[sym_idx] == 1:
            print(sym)