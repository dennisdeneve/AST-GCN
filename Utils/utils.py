# -*- coding: utf-8 -*-

import tensorflow as tf
import scipy.sparse as sp
#from scipy.sparse import linalg
import numpy as np

def normalized_adj(adj):
    """
    Normalize the adjacency matrix.

    Args:
        adj (np.ndarray or sp.spmatrix): Input adjacency matrix.

    Returns:
        np.ndarray: Normalized adjacency matrix.
    """
    # Convert adjacency matrix to COO format
    adj = sp.coo_matrix(adj)
    # Calculate the sum of each row
    rowsum = np.array(adj.sum(1))
    # Calculate the inverse square root of row sums
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    # Handle infinity values
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # Create a diagonal matrix with inverse square root values
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # Perform matrix operations
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    # Convert the result to float32
    normalized_adj = normalized_adj.astype(np.float32)
    return normalized_adj
    
def sparse_to_tuple(mx):
    """
    Convert a sparse matrix to TensorFlow SparseTensor format.

    Args:
        mx (sp.spmatrix): Input sparse matrix.

    Returns:
        tf.SparseTensor: SparseTensor representation of the input matrix.
    """
    # Convert sparse matrix to COO format
    mx = mx.tocoo()
    # Get the coordinate representation of non-zero elements
    coords = np.vstack((mx.row, mx.col)).transpose()
    # Create a SparseTensor using the coordinates and data
    L = tf.SparseTensor(coords, mx.data, mx.shape)
    # Reorder the SparseTensor for better performance
    return tf.sparse.reorder(L) 
    
# def calculate_laplacian(adj, lambda_max=1):  
#     """
#     Calculate the normalized Laplacian matrix.

#     Args:
#         adj (np.ndarray or sp.spmatrix): Input adjacency matrix.
#         lambda_max (float): Maximum eigenvalue for normalization.

#     Returns:
#         tf.SparseTensor: Normalized Laplacian matrix.
#     """
#      #Add an identity matrix and normalize the adjacency matrix
#     adj = normalized_adj(adj + sp.eye(adj.shape[0]))
#     # Convert the adjacency matrix to CSR format
#     adj = sp.csr_matrix(adj)
#     # Convert the adjacency matrix to float32
#     adj = adj.astype(np.float32)
#     # Convert the adjacency matrix to SparseTensor format
#     return sparse_to_tuple(adj)


def calculate_laplacian(adj, lambda_max=1):
    """
    Calculate the normalized Laplacian matrix.

    Args:
        adj (np.ndarray or sp.spmatrix): Input adjacency matrix.
        lambda_max (float): Maximum eigenvalue for normalization.

    Returns:
        tf.SparseTensor: Normalized Laplacian matrix.
    """
    # Convert adj to csr_matrix
    adj = sp.csr_matrix(adj)

    # Add an identity matrix and normalize the adjacency matrix
    adj = normalized_adj(adj + sp.eye(adj.shape[0]))

    # Convert the adjacency matrix to float32
    adj = adj.astype(np.float32)

    # Convert the adjacency matrix to SparseTensor format
    return sparse_to_tuple(adj)
















    
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
