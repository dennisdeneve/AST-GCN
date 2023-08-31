import numpy as np
import tensorflow as tf
from Utils.astgcnUtils import calculate_laplacian_astgcn, prepare_data_astgcn
from tensorflow.keras.layers import Input, Dense, LSTM, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
tf.config.run_functions_eagerly(False)
import warnings
# Filter out specific runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class AstGcn:
    """
    This class handles the creation, compilation, and training 
    of an ASTGCN model for temperature forecasting.
    """
    def __init__(self, time_steps, num_nodes, adjacency_matrix, attribute_data, save_File, forecast_len,
                 X_train, Y_train, X_val, Y_val, split, batch_size,epochs, gru_units, lstm_units):
        """
        Initialize AstGcn with the provided parameters.
        Parameters:
        - time_steps: Number of time steps in each input sequence
        - num_nodes: Number of nodes in the graph
        - adjacency_matrix: Adjacency matrix of the graph
        - attribute_data: Node attribute data
        - save_File: File path to save the model
        - forecast_len: Forecast length (in time steps)
        - X_train: Input sequences for training
        - Y_train: Target sequences for training
        - X_val: Input sequences for validation
        - Y_val: Target sequences for validation
        - split: Tuple for train-val-test split
        """
        self.time_steps = time_steps
        self.num_nodes = num_nodes
        self.adjacency_matrix = adjacency_matrix
        self.attribute_data = attribute_data
        self.save_File = save_File
        self.forecast_len = forecast_len
        self.batch_size = batch_size
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.split = split
        self.epochs = epochs
        self.gru_units = gru_units
        self.lstm_units = lstm_units

    def build_model(self, X_attribute_train, Y_attribute_train, adj_normalized, gru_units, lstm_units):
        """Build and return the initialized AST-GCN model."""
        inputs = Input(shape=(self.time_steps, 1, self.X_train.shape[-1]))
        x = GcnCell(gru_units, adj_normalized, X_attribute_train, Y_attribute_train)(inputs)
        # x = Reshape((-1, self.time_steps * self.num_nodes))(x)
        x = LSTM(lstm_units, activation='relu', return_sequences=False)(x)
        outputs = Dense(1800, activation='linear')(x)
        # outputs = Dense(40, activation='linear')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def compile_and_train_model(self, model):
        """Compile and train the model, using early stopping and model checkpointing."""
        model.compile(optimizer='adam', loss='mean_squared_error')
        early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=5)
        checkpoint = ModelCheckpoint(filepath=self.save_File, save_weights_only=False, 
                                     monitor='val_loss', verbose=1, save_best_only=True,
                                     mode='min', save_freq='epoch')
        callback = [early_stop, checkpoint]
        history = model.fit(self.X_train, self.Y_train, validation_data=(self.X_val, self.Y_val), 
                            batch_size=self.batch_size, epochs=self.epochs, verbose=1, callbacks=callback)
        return history

    def predict(self, model):
        """Use the model to make predictions on the validation set."""
        return model.predict(self.X_val)
    
    def astgcnModel(self):
        """Orchestrate the preparation of data, building and training of model, and prediction."""
        X_attribute_train, Y_attribute_train = prepare_data_astgcn(self.split,self.attribute_data, 
                                                                   self.time_steps, self.num_nodes, 
                                                                   self.forecast_len)
        adj_normalized = calculate_laplacian_astgcn(self.adjacency_matrix, self.num_nodes)
        # adj_normalized = calculate_laplacian_astgcn((self.create_adjacency_matrix(self.num_nodes)), self.num_nodes)
        model = self.build_model(X_attribute_train, Y_attribute_train, adj_normalized, self.gru_units, self.lstm_units)
        #model.summary()
        history = self.compile_and_train_model(model)
        y_pred = self.predict(model)
        return model, history

class GcnCell(tf.keras.layers.Layer):
    """
    Custom layer for Graph Convolutional Network (GCN) operations.
    
    This layer performs a graph convolution operation based on an adjacency matrix, followed by a GRU operation.
    It is designed to work with spatio-temporal data, where the graph convolution operation accounts 
    for spatial dependencies and the GRU for temporal dependencies.
    """
    def __init__(self, units, adj, X_attribute, Y_attribute):
        """
        Initialize the GcnCell layer.
        Parameters:
        - units: Number of units in the GRU layer.
        - adj: Normalized adjacency matrix.
        - X_attribute: Attribute data - Node attribute sequences.
        - Y_attribute: Attribute data - Target sequences for node attributes.
        """
        super(GcnCell, self).__init__()
        self.units = units
        self.adj = adj
        self.X_attribute = X_attribute
        self.Y_attribute = Y_attribute
        self.layer = tf.keras.layers.GRU(self.units, return_sequences=True)
        self.dense = tf.keras.layers.Dense(40)

    def call(self, inputs):
        """
        Forward pass of the layer.
        Parameters:
        - inputs: Input tensor.
        Returns:
        - reshaped_output: Output tensor.
        """
        inputs_with_attributes = self.dense(inputs)
        inputs_with_attributes = tf.squeeze(inputs_with_attributes, axis=2)  
        adj_normalized_tiled = tf.expand_dims(self.adj, axis=0)
        adj_normalized_tiled = tf.tile(adj_normalized_tiled, [tf.shape(inputs)[0], 1, 1])
        output = self.layer(inputs_with_attributes)
        reshaped_output = tf.reshape(output, [-1, tf.shape(inputs)[1], self.units])
        return reshaped_output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'adj': self.adj.numpy().tolist(),
            'X_attribute': self.X_attribute,
            'Y_attribute': self.Y_attribute
        })
        return config