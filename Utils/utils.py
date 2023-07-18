import tensorflow as tf
import scipy.sparse as sp
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def dataSplit(split, series):
    train = series[0:split[0]]
    validation = series[split[0]:split[1]]
    test = series[split[1]:split[2]]
    return train, validation, test

def min_max(train, validation, test):
    norm = MinMaxScaler().fit(train.reshape(train.shape[0], -1))
    train_data = norm.transform(train.reshape(train.shape[0], -1))
    val_data = norm.transform(validation.reshape(validation.shape[0], -1))
    test_data = norm.transform(test.reshape(test.shape[0], -1))
    return train_data, val_data, test_data

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



# def normalized_adj(adj):
#     """
#     Normalize the adjacency matrix.

#     Args:
#         adj (np.ndarray or sp.spmatrix): Input adjacency matrix.

#     Returns:
#         np.ndarray: Normalized adjacency matrix.
#     """
#     # Convert adjacency matrix to COO format
#     adj = sp.coo_matrix(adj)
#     # Calculate the sum of each row
#     rowsum = np.array(adj.sum(1))
#     # Calculate the inverse square root of row sums
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     # Handle infinity values
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     # Create a diagonal matrix with inverse square root values
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     # Perform matrix operations
#     normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
#     # Convert the result to float32
#     normalized_adj = normalized_adj.astype(np.float32)
#     return normalized_adj
    
# def sparse_to_tuple(mx):
#     """
#     Convert a sparse matrix to TensorFlow SparseTensor format.

#     Args:
#         mx (sp.spmatrix): Input sparse matrix.

#     Returns:
#         tf.SparseTensor: SparseTensor representation of the input matrix.
#     """
#     # Convert sparse matrix to COO format
#     mx = mx.tocoo()
#     # Get the coordinate representation of non-zero elements
#     coords = np.vstack((mx.row, mx.col)).transpose()
#     # Create a SparseTensor using the coordinates and data
#     L = tf.SparseTensor(coords, mx.data, mx.shape)
#     # Reorder the SparseTensor for better performance
#     return tf.sparse.reorder(L) 
    



# def calculate_laplacian(adj, lambda_max=1):
#     """
#     Calculate the normalized Laplacian matrix.

#     Args:
#         adj (np.ndarray or sp.spmatrix): Input adjacency matrix.
#         lambda_max (float): Maximum eigenvalue for normalization.

#     Returns:
#         tf.SparseTensor: Normalized Laplacian matrix.
#     """
#     # Convert adj to csr_matrix
#     adj = sp.csr_matrix(adj)

#     # Add an identity matrix and normalize the adjacency matrix
#     adj = normalized_adj(adj + sp.eye(adj.shape[0]))

#     # Convert the adjacency matrix to float32
#     adj = adj.astype(np.float32)

#     # Convert the adjacency matrix to SparseTensor format
#     return sparse_to_tuple(adj)
















    
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
