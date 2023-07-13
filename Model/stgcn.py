import tensorflow as tf
from tensorflow.compat.v1.nn.rnn_cell import RNNCell
from Utils.utils import calculate_laplacian
from Model.acell import load_assist_data
from tensorflow.keras.layers import Input, Dense, LSTM, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def stgcnModel(time_steps, num_nodes, adjacency_matrix, save_File,
               X_train, Y_train, X_val, Y_val
               ):
    """
    Build and train the temperature model.
    Returns:
    model (Sequential): The trained temperature model.
    history (History): The training history.
    """
        
     # Step 4: Define the ST-GCN model architecture
    inputs = Input(shape=(time_steps, 1, num_nodes * 60))  # Update the input shape
    x = tf.keras.layers.TimeDistributed(stgcnCell(64, adjacency_matrix, num_nodes))(inputs)
    x = Reshape((-1, 10 * 64))(x)  # Reshape into 3D tensor
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    outputs = Dense(60, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
                
    # Step 5: Compile and train the T-GCN model
    model.compile(optimizer='adam', 
                  loss='mean_squared_error')
                
    # Define callbacks for early stopping and model checkpointing
    early_stop = EarlyStopping(monitor='val_loss', 
                               mode='min', 
                               patience=5)
    checkpoint = ModelCheckpoint(filepath=save_File, 
                                save_weights_only=False, 
                                monitor='val_loss', 
                                verbose=1,
                                save_best_only=True,
                                mode='min', 
                                save_freq='epoch')
    callback = [early_stop, checkpoint]            

    # print("Final X Train shape ",X_train.shape)
    # print("Final Y Train shape ",Y_train.shape)
    # print("Final X Val shape ",X_val.shape)
    # print("Final Y Val shape ",Y_val.shape)
                
    ########## Training the model
    history = model.fit(X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    batch_size=196,
                    epochs=1,
                    verbose=1,
                    callbacks=callback)
    
    return model, history
                
                
# Sptial dynamics considered
# In the ST-GCN model, each graph convolutional cell (GcnCell) represents a single time step. 
# The GcnCell layer applies graph convolution operation to capture spatial dependencies among 
# nodes in the graph. This operation takes into account the adjacency matrix (adj) to propagate 
# information between neighboring nodes.

# Temporal dynamics considered
# # By using a recurrent neural network (RNN) layer (GRU) within the GcnCell, the model captures 
# the temporal dynamics of the data. The GRU layer receives the output of the graph convolutional
# operation from the previous time step as input and processes it along with the current input 
# to generate the output sequence.

# Therefore, the ST-GCN model combines both spatial and temporal information,
# allowing it to learn and model the dynamics of the graph-structured data over time. 
# The ST-GCN model combines both spatial and temporal information by integrating 
# graph convolutional networks and recurrent neural networks (GRU   ). 

#################### Method from model, stgcn ####################
def stgcnCell(units, adj, num_nodes):
    class GcnCell(tf.keras.layers.Layer):
        def __init__(self, units, adj):
            super(GcnCell, self).__init__()
            self.units = units
            self.adj = adj

        def build(self, input_shape):
            self.layer = tf.keras.layers.GRU(self.units, return_sequences=True)
            self.layer.build(input_shape)

        def call(self, inputs):
            adj_normalized_tiled = tf.expand_dims(self.adj, axis=0)
            adj_normalized_tiled = tf.tile(adj_normalized_tiled, [tf.shape(inputs)[0], 1, 1])
            return self.layer(inputs)

        def compute_output_shape(self, input_shape):
            return input_shape[:-1] + (self.units,)

        def get_config(self):
            config = super().get_config().copy()
            config.update({
                'units': self.units,
                'adj': self.adj.numpy().tolist(),  # convert tensor to list for saving
            })
            return config

    adj_normalized = calculate_laplacian(adj)
    adj_normalized = tf.convert_to_tensor(adj_normalized, dtype=tf.float32)

    adj_normalized = tf.sparse.reorder(tf.sparse.SparseTensor(indices=tf.where(adj_normalized != 0),
                                                              values=tf.gather_nd(adj_normalized, tf.where(adj_normalized != 0)),
                                                              dense_shape=adj_normalized.shape))
    adj_normalized = tf.sparse.to_dense(adj_normalized)
    adj_normalized = tf.reshape(adj_normalized, [num_nodes, num_nodes])
    return GcnCell(units, adj_normalized)


################### RNNs -> GRU ####################
# Yes, a GRU (Gated Recurrent Unit) is a type of recurrent neural network (RNN). 
# GRU is a variant of the traditional RNN architecture that addresses some of 
# the limitations of standard RNNs, such as the vanishing gradient problem and 
# difficulty in capturing long-term dependencies.

# Similar to RNNs, GRUs are designed to process sequential data, where the current 
# output depends on both the current input and the previous hidden state. 
# They are recurrent in nature because they have a hidden state that is updated at 
# each time step and used as input for the next time step.

# However, GRUs differ from traditional RNNs in their internal structure. GRUs introduce 
# the concept of "gates" that control the flow of information through the network. These gates, 
# namely the update gate and reset gate, determine how much of the previous hidden state to 
# incorporate and how much of the new input to remember.
# By using these gates, GRUs can selectively update and remember relevant information over 
# long sequences, making them better suited for capturing long-term dependencies. They have been shown
# to be effective in various tasks involving sequential dat and time series analysis.


############################## OLD MODEL IMPLEMENTATIONS OF TGCN ##############################################

'''
This method implements the TGCN model with tgcn.py 
It takes in three arguments: _X, _weights, _biases and config.
_X is a placeholder for the input data, which is a tensor of shape (batch_size, time_steps, num_nodes, input_dim). 
_weights and _biases are dictionaries that contain the weights and biases for the different layers of the T-GCN model.
config is the yaml configuration file used for settings.
'''
def TGCN(_X, _weights, _biases, config):
    gru_units =  config['gru_units']['default']
    data_name = config['dataset']['default']
    pre_len =  config['pre_len']['default']
    data, adj = load_assist_data('data/ADDO ELEPHANT PARK.csv','data/adj_mx.csv')
    num_nodes = data.shape[1]
    
    ### defines a TGCN cell using the tgcnCell class and creates a multi-layer RNN cell 
    cell_1 = tgcnCellOld(gru_units, adj, num_nodes=num_nodes)
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
    
class tgcnCellOld(RNNCell):
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

        super(tgcnCellOld, self).__init__(_reuse=reuse)
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
#       print('x_s_shape:',x_s.shape)
        
        input_size = x_s.get_shape()[2]

        ####It then applies a series of matrix multiplications to compute the output of the GCN.
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