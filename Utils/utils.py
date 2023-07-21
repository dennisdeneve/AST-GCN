import tensorflow as tf
import scipy.sparse as sp
import numpy as np
import os
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

def calculate_laplacian_astgcn(adj):
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
    
    return laplacian


def normalize_adj(adj):
    """
    Normalize the adjacency matrix.

     Args:
         adj (np.ndarray or sp.spmatrix): Input adjacency matrix.

     Returns:
         np.ndarray: Normalized adjacency matrix.
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

    Args:
        input_dim (int): Input dimension of the weight variable.
        output_dim (int): Output dimension of the weight variable.
        name (str): Name of the weight variable.

    Returns:
        tf.Variable: Weight variable.
    """
    # Calculate the initialization range
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    # Create a random uniform weight matrix
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                            maxval=init_range, dtype=tf.float32)
    # Return the weight variable
    return tf.Variable(initial,name=name)  
