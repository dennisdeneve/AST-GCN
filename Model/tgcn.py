# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.compat.v1.nn.rnn_cell import RNNCell
from Utils.utils import calculate_laplacian
from Model.acell import preprocess_data,load_assist_data
import numpy as np
import pandas as pd
import yaml
import os

'''
This method implements the TGCN model with tgcn.py 
It takes in three arguments: _X, _weights, and _biases.
_X is a placeholder for the input data, which is a tensor of shape (batch_size, time_steps, num_nodes, input_dim). 
_weights and _biases are dictionaries that contain the weights and biases for the different layers of the T-GCN model.
'''
def TGCN(_X, _weights, _biases):
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the relative file path to the config file
    config_file_path = os.path.join(current_dir, '..', 'config.yaml')
    # Open the config file
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    gru_units =  config['gru_units']['default']
    data_name = config['dataset']['default']
    pre_len =  config['pre_len']['default']
    
    ########## load data #########
    if data_name == 'sz':
        data, adj = load_assist_data('sz')
    
    num_nodes = data.shape[1]
    
    ### defines a TGCN cell using the tgcnCell class and creates a multi-layer RNN cell 
    cell_1 = tgcnCell(gru_units, adj, num_nodes=num_nodes)
    cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    
    #It then unstacks the input tensor _X along the time_steps axis and feeds the resulting list
    # of tensors into the RNN cell using the tf.compat.v1.nn.static_rnn method.
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.compat.v1.nn.static_rnn(cell, _X, dtype=tf.float32)
    
    # then the method reshapes each tensor in the output list and concatenates them along 
    # the time_steps axis to obtain a tensor of shape (batch_sizetime_steps, num_nodesgru_units). 
    # It then multiplies this tensor with the output weight matrix and adds the output bias vector to obtain the final output tensor.
    m = []
    for i in outputs:
        o = tf.reshape(i,shape=[-1,num_nodes,gru_units])
        o = tf.reshape(o,shape=[-1,gru_units])
        m.append(o)
        
    #Finally the method reshapes the output tensor into the original shape (batch_size, num_nodes, pre_len) 
    # and transposes the second and third dimensions to obtain the output tensor in the format (batch_size, pre_len, num_nodes). 
    # It then reshapes the tensor to have shape (batch_size*pre_len, num_nodes) and returns the output tensor, 
    # the intermediate output tensors m, and the final RNN states.
    last_output = m[-1]
    output = tf.matmul(last_output, _weights['out']) + _biases['out']
    output = tf.reshape(output,shape=[-1,num_nodes,pre_len])
    output = tf.transpose(output, perm=[0,2,1])
    output = tf.reshape(output, shape=[-1,num_nodes])
    return output, m, states
    
    
class tgcnCell(RNNCell):
    """Temporal Graph Convolutional Network """

    def call(self, inputs, **kwargs):
        pass

    '''
    Constructor method for tgcnCell which initializes the object and sets its properties,
    num_units: The number of units (neurons) in the TGCN cell.
    adj: The adjacency matrix of the graph, which describes the relationships between nodes in the graph.
    num_nodes: The number of nodes in the graph.
    '''
    def __init__(self, num_units, adj, num_nodes, input_size=None,
                 act=tf.nn.tanh, reuse=None):

        super(tgcnCell, self).__init__(_reuse=reuse)
        self._act = act
        self._nodes = num_nodes
        self._units = num_units
        self._adj = []
        self._adj.append(calculate_laplacian(adj))

    '''
    Accessor method that returns the state size (nodes * units)
    '''
    @property
    def state_size(self):
        return self._nodes * self._units

    '''
    Accessor method that returns the output size (Units)
    '''
    @property
    def output_size(self):
        return self._units

    '''
    This method implements a variant of the Temporal Graph Convolutional Network (TGCN) cell in TensorFlow.
    The function takes three arguments:
    - inputs and state are tensors that represent the input and hidden state of the TGCN cell, respectively. 
    - optional scope argument is used to define a variable scope for the TGCN cell.
    
    Overall, this function defines the behavior of a single TGCN cell, which can be used as a building block for a TGCN model.
    '''
    def __call__(self, inputs, state, scope=None):
        #The function first creates a variable scope using the tf.compat.v1.variable_scope function
        # with the provided scope argument or a default value of "tgcn".
        # Within this scope, the function defines two sub-scopes named "gates" and "candidate".
        with tf.compat.v1.variable_scope(scope or "tgcn"):
            with tf.compat.v1.variable_scope("gates"):  
                # Within the "gates" sub-scope, the function computes the update and reset gates
                # of the TGCN cell using a sigmoid activation function applied to a linear transformation
                # of the concatenation of inputs and state. The self._gc function is used to perform 
                # the linear transformation, which is a graph convolutional operation. The bias argument 
                # is set to 1.0 to initialize the bias vector of the linear transformation to a constant value of 1.0.
                value = tf.compat.v1.nn.sigmoid(
                    self._gc(inputs, state, 2 * self._units, bias=1.0, scope=scope))
                # The tf.compat.v1.split function is used to split the concatenated output of the self._gc function
                # into two tensors r and u, each of size self._units. r represents the reset gate,
                # while u represents the update gate.
                r, u = tf.compat.v1.split(value=value, num_or_size_splits=2, axis=1)
                
                
            with tf.compat.v1.variable_scope("candidate"):
                #Within the "candidate" sub-scope, the function computes a candidate activation value c
                # using a nonlinear activation function (self._act) applied to another linear transformation 
                # of inputs and r * state. This operation is also a graph convolutional operation.
                r_state = r * state
                c = self._act(self._gc(inputs, r_state, self._units, scope=scope))
            # The new hidden state new_h is then computed as a weighted sum of the candidate activation value c 
            # and the current hidden state state, where the weights are determined by the update gate u. 
            # Finally, the function returns the new hidden state new_h as well as a copy of new_h.
            new_h = u * state + (1 - u) * c
        return new_h, new_h

    '''
    This method implements the graph convolutional network (GCN) architecture for graph-based data.
    The _gc function takes in three arguments: 
    - inputs and state are both TensorFlow tensors, while output_size and bias are scalars.
    '''
    def _gc(self, inputs, state, output_size, bias=0.0, scope=None):
        ##  The function first reshapes inputs and state to have the same dimensions and then 
        # concatenates them along the last dimension.
        ## inputs:(-1,num_nodes)
        inputs = tf.expand_dims(inputs, 2)
#        print('inputs_shape:',inputs.shape)
        ## state:(batch,num_node,gru_units)
        state = tf.reshape(state, (-1, self._nodes, self._units))
#        print('state_shape:',state.shape)
        ## concat
        x_s = tf.concat([inputs, state], axis=2)
#        print('x_s_shape:',x_s.shape)
        
         #####  commented out code in the function which appears to be an attempt to include additional knowledge graphs in -
         
#        kgembedding = np.array(pd.read_csv(r'/DHH/sz_gcn/sz_data/sz_poi_transR_embedding20.csv',header=None))
#        kgeMatrix = np.repeat(kgembedding[np.newaxis, :, :], self._units, axis=0)
#        kgeMatrix = tf.reshape(tf.constant(kgeMatrix, dtype=tf.float32), (self._units, -1))
#        kgMatrix = tf.reshape(kgeMatrix,(-1,self._nodes, 20))
#        ## inputs:(-1,num_nodes)
#        inputs = tf.expand_dims(inputs, 2)
#        ## state:(batch,num_node,gru_units)
#        state = tf.reshape(state, (-1, self._nodes, self._units))
#        ## concat
#        print('kgMatrix_shape:',kgMatrix.shape)
#        print('inputs_shape:',inputs.shape)
#        print('state_shape:',state.shape)
#        kg_x = tf.concat([inputs, kgMatrix],axis = 2)
#        print('kg_x_shape:',kg_x.shape)
#        x_s = tf.concat([kg_x, state], axis=2)
        # input_size = x_s.get_shape()[2].value
        input_size = x_s.get_shape()[2]

        
        ####It then applies a series of matrix multiplications to compute the output of the GCN.
        ## (num_node,input_size,-1)
        x0 = tf.transpose(x_s, perm=[1, 2, 0])  
        x0 = tf.reshape(x0, shape=[self._nodes, -1])
        
        ## The tf.compat.v1.sparse_tensor_dense_matmul function is used to perform a sparse matrix multiplication 
        # between each adjacency matrix m in self._adj and the input tensor x0. 

        scope = tf.compat.v1.get_variable_scope()
        with tf.compat.v1.variable_scope(scope):
            for m in self._adj:
                x1 = tf.compat.v1.sparse_tensor_dense_matmul(m, x0)
#                print(x1)
            #  # The resulting output is reshaped and then multiplied with a learnable weight matrix and a bias term, 
            # and the output is reshaped and returned.
            x = tf.compat.v1.reshape(x1, shape=[self._nodes, input_size,-1])
            x = tf.compat.v1.transpose(x,perm=[2,0,1])
            x = tf.compat.v1.reshape(x, shape=[-1, input_size])
            weights = tf.compat.v1.get_variable(
                # 'weights', [input_size, output_size], initializer=tf.compat.v1.contrib.layers.xavier_initializer())
                'weights', [input_size, output_size], initializer=tf.initializers.glorot_uniform())

            
            x = tf.compat.v1.matmul(x, weights)  # (batch_size * self._nodes, output_size)
            biases = tf.compat.v1.get_variable(
                "biases", [output_size], initializer=tf.compat.v1.constant_initializer(bias, dtype=tf.compat.v1.float32))
                       
            x = tf.compat.v1.nn.bias_add(x, biases)
            x = tf.compat.v1.reshape(x, shape=[-1, self._nodes, output_size])
            x = tf.compat.v1.reshape(x, shape=[-1, self._nodes * output_size])
        return x
