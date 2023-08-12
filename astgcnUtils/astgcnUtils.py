import tensorflow as tf
import scipy.sparse as sp
import numpy as np
import os
import random
import math
import numpy.linalg as la
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

def generate_execute_file_paths(base_path):#, forecast_len, station):
    target_path = f'{base_path}/Targets/target.csv'
    result_path = f'{base_path}/Predictions/result.csv'
    loss_path = f'{base_path}/Predictions/loss.csv'
    actual_vs_predicted_path = f'{base_path}/Predictions/actual_vs_predicted.csv'
    # Make sure all paths exist
    for path in [target_path, result_path, loss_path, actual_vs_predicted_path]:
        create_file_if_not_exists(path)
    return target_path, result_path, loss_path, actual_vs_predicted_path

def get_file_paths(station, horizon, model='ASTGCN'):
    return {
        "yhat": f'Results/{model}/{horizon} Hour Forecast/{station}/Predictions/result.csv',
        "target": f'Results/{model}/{horizon} Hour Forecast/{station}/Targets/target.csv',
        "metrics": f'Results/{model}/{horizon} Hour Forecast/{station}/Metrics/metrics.txt',
        "actual_vs_predicted": f'Results/{model}/{horizon} Hour Forecast/{station}/Metrics/actual_vs_predicted.txt'
    }

def create_file_if_not_exists(file_path):
    # Get the directory from the file path
    directory = os.path.dirname(file_path)
    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    # If the file does not exist, create it
    if not os.path.isfile(file_path):
        open(file_path, 'w').close()
        
def dataSplit(split, series):
    train = series[0:split[0]]
    validation = series[split[0]:split[1]]
    test = series[split[1]:split[2]]
    return train, validation, test

def min_max(train, validation, test, splits):
    norm = MinMaxScaler().fit(train.reshape(train.shape[0], -1))
    train_data = norm.transform(train.reshape(train.shape[0], -1))
    val_data = norm.transform(validation.reshape(validation.shape[0], -1))
    test_data = norm.transform(test.reshape(test.shape[0], -1))
    return train_data, val_data, test_data, splits

def create_X_Y(ts: np.array, lag=1, num_nodes=1, n_ahead=1, target_index=0):
    X, Y = [], []
    if len(ts) - lag - n_ahead + 1 <= 0:
        X.append(ts)
        Y.append(ts)
    else:
        for i in range(len(ts) - lag - n_ahead + 1):
            X.append(ts[i:(i + lag)])
            Y.append(ts[i + lag + n_ahead - 1])  
    X, Y = np.array(X), np.array(Y)

    num_samples = len(X)
    time_steps = 10 
    num_samples -= num_samples % time_steps
    X = X[:num_samples]
    Y = Y[:num_samples]
    ### Reshaping to match the ASTGCN model output architecture
    X = np.expand_dims(X, axis=2)
    Y = np.reshape(Y, (Y.shape[0], -1))  # Reshape Y to match the shape of y_pred
    return X, Y

def calculate_laplacian(adj):
    adj_normalized = normalize_adj(adj + np.eye(adj.shape[0]))
    return adj_normalized

def prepare_data_astgcn(split,attribute_data, time_steps, num_nodes, forecast_len):
    """
    Split and normalize the attribute data and return the train, validation and test data.
    """
    train_attribute, val_attribute, test_attribute = dataSplit(split, attribute_data)
    train_Attribute, val_Attribute, test_Attribute, split = min_max(train_attribute.values, val_attribute.values, test_attribute.values, split) 
    X_attribute_train, Y_attribute_train = create_X_Y(train_Attribute, time_steps, num_nodes, forecast_len)
    X_val, Y_val = create_X_Y(val_Attribute, time_steps, num_nodes, forecast_len)
    X_test, Y_test = create_X_Y(test_Attribute, time_steps, num_nodes, forecast_len)
    return X_attribute_train, Y_attribute_train

def calculate_laplacian_astgcn(adj, num_nodes):
    # Calculate the normalized Laplacian matrix
    adj = tf.convert_to_tensor(adj, dtype=tf.float32)
    adj = tf.sparse.reorder(tf.sparse.SparseTensor(indices=tf.where(adj != 0),
                                                   values=tf.gather_nd(adj, tf.where(adj != 0)),
                                                   dense_shape=adj.shape))
    adj = tf.sparse.to_dense(adj)
    adj = tf.reshape(adj, [adj.shape[0], adj.shape[0]])
    # Calculate row sums
    rowsum = tf.reduce_sum(adj, axis=1)
    rowsum = tf.maximum(rowsum, 1e-12)  # Add small epsilon to avoid division by zero
    # Calculate the degree matrix
    degree = tf.linalg.diag(1.0 / tf.sqrt(rowsum))
    # Calculate the normalized Laplacian matrix
    laplacian = tf.eye(adj.shape[0]) - tf.matmul(tf.matmul(degree, adj), degree)
    # Rest of normalization that occured in model method before
    laplacian = tf.convert_to_tensor(laplacian, dtype=tf.float32)
    laplacian = tf.reshape(laplacian, [num_nodes, num_nodes])
    return laplacian

def normalize_adj(adj):
    """
    Normalize the adjacency matrix.
    """
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = np.diag(r_inv)
    adj_normalized = adj.dot(r_mat_inv).transpose().dot(r_mat_inv)
    return adj_normalized
    
def weight_variable_glorot(input_dim, output_dim, name=""):
    """
    Create a weight variable using the Glorot initialization.
    """
    # Calculate the initialization range
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    # Create a random uniform weight matrix
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                            maxval=init_range, dtype=tf.float32)
    # Return the weight variable
    return tf.Variable(initial,name=name)  

def SMAPE(actual, predicted):
    """
    Calculates the SMAPE metric
    Parameters:
        actual - target values
        predicted - output values predicted by model
    Returns:
        smape - returns smape metric
    """
    return (np.mean(abs(predicted - actual)  / ((abs(predicted) + abs(actual)) / 2)) * 100) 

def smape_std(actual, predicted):
        """
        Calculates the standard deviation of SMAPE values
        Parameters:
            actual - target values
            predicted - output values predicted by model
        Returns:
            std - returns the standard deviation of SMAPE values
        """
        smapes = abs(predicted - actual)  / ((abs(predicted) + abs(actual)) / 2) * 100
        return (np.std(smapes)) 

def MSE(target, pred):
    """
    Calculates the MSE metric
    Parameters:
        actual - target values
        predicted - output values predicted by model
    Returns:
        mse - returns MSE metric
    """
    return mean_squared_error(target, pred, squared=True)

def RMSE(target, pred):
    """
    Calculates the RMSE metric
    Parameters:
        actual - target values
        predicted - output values predicted by model
    Returns:
        root - returns RMSE metric
    """
    root = math.sqrt(mean_squared_error(target, pred))
    return root

def MAE(target, pred):
    """
    Calculates the MAE metric
    Parameters:
        actual - target values
        predicted - output values predicted by model
    Returns:
        mae - returns MAE metric
    """
    return mean_absolute_error(target, pred)


def generateRandomParameters(config):
    # pass
    batch_size = [32,64, 128]
    epochs = [30, 40, 50, 60]

    batch = batch_size[random.randint(0,len(batch_size)-1)]
    epoch = epochs[random.randint(0,len(epochs)-1)]

    config['batch_size']['default'] = batch
    config['training_epoch']['default'] = epoch

    return [batch, epoch]

### Perturbation Analysis
def MaxMinNormalization(x,Max,Min):
    x = (x-Min)/(Max-Min)
    return x