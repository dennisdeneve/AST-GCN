import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from numpy import random


def create_dataset(station):
    """
    Creates a dataset from the original weather station data
    Parameters:
        station - which station's data to read in
    Returns:
        data - returns dataframe with selected features
    """

    df = pd.read_csv(station)
    features_final = ['Temperature', 'Pressure', 'WindSpeed', 'WindDir', 'Humidity', 'Rain']
    data = df[features_final]
    return data


def dataSplit(split, series):
    """
    Splits the data into train, validation, test sets for walk-forward validation.
    Parameters:
        split - points at which to split the data into train, validation and test sets
        series - weather station data
    Returns:
        train, validation, test - returns the train, validation and test sets
    """

    train = series[0:split[0]]
    validation = series[split[0]:split[1]]
    test = series[split[1]:split[2]]

    return train, validation, test


def z_score_normalize(train, validation, test):
    """
    Performs z-score normalisation on the train, validation and test sets using the train data mean and standard
    deviation.
    Parameters:
        train, validation, test - train, validation and test data sets
    Returns:
        train, validation, test - returns the scaled train, validation and test sets
    """

    train_mean = train.mean()
    train_std = train.std()

    train = (train - train_mean) / train_std
    validation = (validation - train_mean) / train_std
    test = (test - train_mean) / train_std

    return train, validation, test, train_mean, train_std


def min_max(train, validation, test):
    """
    Performs MinMax scaling on the train, validation and test sets using the train data min and max.
    Parameters:
        train, validation, test - train, validation and test data sets
    Returns:
        train, validation, test - returns the scaled train, validation and test sets
    """

    norm = MinMaxScaler().fit(train)

    train_data = norm.transform(train)
    val_data = norm.transform(validation)
    test_data = norm.transform(test)

    return train_data, val_data, test_data


def create_X_Y(ts: np.array, lag=1, n_ahead=1, target_index=0):
    """
    A method to create X and Y matrix from a time series array.
    Parameters:
        ts - time series array
        lag - length of input sequence
        n_ahead - length of output sequence(forecasting horizon)
        target_index - index to be used as output target(Temperature)
    """

    n_features = ts.shape[1]

    X, Y = [], []

    if len(ts) - lag <= 0:
        X.append(ts)
    else:
        for i in range(len(ts) - lag - n_ahead):
            Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])
            X.append(ts[i:(i + lag)])

    X, Y = np.array(X), np.array(Y)

    X = np.reshape(X, (X.shape[0], lag, n_features))
    x, y = shuffle(X, Y, random_state=0)

    return x, y


def generateRandomTCNParameters():
    """
    Generates a random configuration of hyper-parameters from the pre-defined search space.

    Returns:
        configuration_list - Returns randomly selected configuration of TCN hyper-parameters.
    """

    activation = 'relu'
    epoch = 40
    patience = 5
    batch_size = 64
    lr = 0.01
    batch_norm = False
    weight_norm = False
    layer_norm = True
    padding = 'causal'
    kernels = 2
    loss_metric = 'MSE'
    n_ahead_length = 24

    dilations = [1, 2, 4, 8, 16, 32, 64]
    filters = [32, 64, 128]
    layers = [1, 2]
    dropout = [0.1, 0.2, 0.3, 0.4]
    lag_length = [24, 48]

    filters = filters[random.randint(len(filters))]
    lag_length = lag_length[random.randint(len(lag_length))]
    dropout = dropout[random.randint(len(dropout))]
    layers = layers[random.randint(len(layers))]

    cfg = {'Activation': activation,
           'Epochs': epoch,
           'Patience': patience,
           'Loss': loss_metric,
           'Forecast Horizon': n_ahead_length,
           'Batch Size': batch_size,
           'Lag': lag_length,
           'Dropout': dropout,
           'lr': lr,
           'Batch Norm': batch_norm,
           'Weight Norm': weight_norm,
           'Layer Norm': layer_norm,
           'Padding': padding,
           'Kernels': kernels,
           'Dilations': dilations,
           'Filters': filters,
           'Layers': layers,
           }

    return cfg


def stringtoCfgTCN(params):
    """
    Creates a configuration of the optimal hyper-parameters for each weather station.

    Parameters:
        params -  String of optimal hyper-parameters.

    Returns:
        config -  List of hyper-parameters for the TCN model.
    """

    parameters = params.split(", ")
    config = []
    for i in range(len(parameters)):
        config.append(str.strip(parameters[i]))

    return config
