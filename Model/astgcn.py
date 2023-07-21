import numpy as np
import tensorflow as tf
from Utils.utils import calculate_laplacian_astgcn, prepare_data_astgcn
from tensorflow.keras.layers import Input, Dense, LSTM, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from Utils.utils import create_X_Y, min_max, dataSplit
tf.config.run_functions_eagerly(False)

class AstGcn:
    """
    This class handles the creation, compilation, and training 
    of an ASTGCN model for temperature forecasting.
    """
    def __init__(self, time_steps, num_nodes, adjacency_matrix, attribute_data, save_File, forecast_len,
                 X_train, Y_train, X_val, Y_val, split):
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
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.split = split

    def prepare_data(self):
        """Prepare the data for model training."""
        return prepare_data_astgcn(self.split,self.attribute_data, self.time_steps, self.num_nodes, self.forecast_len)
    
    def build_model(self, X_attribute_train, Y_attribute_train, adj_normalized):
        """Build and return the initialized AST-GCN model."""
        inputs = Input(shape=(self.time_steps, 1, self.X_train.shape[-1]))
        x = GcnCell(63, adj_normalized, X_attribute_train, Y_attribute_train)(inputs)
        x = Reshape((-1, self.time_steps * self.num_nodes))(x)
        x = LSTM(64, activation='relu', return_sequences=False)(x)
        outputs = Dense(40, activation='linear')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def compile_and_train_model(self, model):
        """Compile and train the model, using early stopping and model checkpointing."""
        model.compile(optimizer='adam', loss='mean_squared_error')
        early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=5)
        checkpoint = ModelCheckpoint(filepath=self.save_File, save_weights_only=False, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_freq='epoch')
        callback = [early_stop, checkpoint]
        history = model.fit(self.X_train, self.Y_train, validation_data=(self.X_val, self.Y_val), batch_size=196, epochs=1, verbose=1, callbacks=callback)
        return history

    def predict(self, model):
        """Use the model to make predictions on the validation set."""
        return model.predict(self.X_val)
    
    def astgcnModel(self):
        """Orchestrate the preparation of data, building and training of model, and prediction."""
        X_attribute_train, Y_attribute_train = self.prepare_data()
        adj_normalized = calculate_laplacian_astgcn(self.adjacency_matrix, self.num_nodes)
        model = self.build_model(X_attribute_train, Y_attribute_train, adj_normalized)
        model.summary()
        history = self.compile_and_train_model(model)
        y_pred = self.predict(model)
        return model, history

    # def astgcnModel(self):
    #     """
    #     This method prepares data, constructs and trains the ASTGCN model, while monitoring 
    #     validation loss for early stopping and best model checkpointing. 
        
    #     It reshapes the validation data to match model output shape and finally returns the 
    #     trained model along with its training history.
        
    #     Returns:
    #     - model: The trained model
    #     - history: Training history
    #     """
    #     X_attribute_train, Y_attribute_train = prepare_data_astgcn(self.split,self.attribute_data, self.time_steps, self.num_nodes, self.forecast_len)
    #     adj_normalized = calculate_laplacian_astgcn(self.adjacency_matrix, self.num_nodes)
    #      # Define the AST-GCN model architecture
    #     inputs = Input(shape=(self.time_steps, 1, self.X_train.shape[-1]))
    #     x = GcnCell(63, adj_normalized, X_attribute_train, Y_attribute_train)(inputs)
    #     x = Reshape((-1, self.time_steps * self.num_nodes ))(x)
    #     x = LSTM(64, activation='relu', return_sequences=False)(x)
    #     outputs = Dense(40, activation='linear')(x)
    #     model = Model(inputs=inputs, outputs=outputs)
    #     # Compile and train the T-GCN model
    #     model.compile(optimizer='adam', loss='mean_squared_error')
    #     # Define callbacks for early stopping and model checkpointing
    #     early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    #     checkpoint = ModelCheckpoint(filepath=self.save_File, 
    #                                  save_weights_only=False, 
    #                                  monitor='val_loss', verbose=1, 
    #                                  save_best_only=True, mode='min', 
    #                                  save_freq='epoch')
    #     callback = [early_stop, checkpoint]
    #     #Print out model summary statistics
    #     model.summary()
    #     # Reshape X & Y validation data
    #     last_column_X = self.X_val[:, :, :, -1] 
    #     self.X_val = np.repeat(np.expand_dims(last_column_X, axis=-1), 40, axis=-1) 
    #     last_column_Y = self.Y_val[:, -1] 
    #     self.Y_val = np.repeat(np.expand_dims(last_column_Y, axis=-1), 40, axis=-1)
    #     ############# Training the model
    #     history = model.fit(self.X_train, self.Y_train, 
    #                         validation_data=(self.X_val, self.Y_val), 
    #                         batch_size=196, epochs=1,
    #                         verbose=1, callbacks=callback)
    #     y_pred = model.predict(self.X_val)
    #     return model, history
    


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


# # This function creates, compiles, and trains a st-gnn model for temperature forecasting.
# # It first prepares the data by splitting, normalizing, and reshaping it as necessary. 
# # The model architecture includes a custom ASTGCN cell, an LSTM layer, and a Dense layer. 
# # The model uses the Adam optimizer and mean squared error as the loss function, and 
# # implements early stopping and model checkpointing during training.
# def astgcnModel(time_steps, num_nodes, adjacency_matrix, 
#                 attribute_data, save_File, forecast_len,
#                 X_train, Y_train, X_val, Y_val, split
#                ):    
#     """
#     Build and train the ASTGCN model.
#     Parameters:
#     - time_steps: Number of time steps in each input sequence
#     - num_nodes: Number of nodes in the graph
#     - adjacency_matrix: Adjacency matrix of the graph
#     - attribute_data: Node attribute data
#     - save_File: File path to save the model
#     - forecast_len: Forecast length (in time steps)
#     - increment: The increment to use when creating sequences
#     - X_train: Input sequences for training
#     - Y_train: Target sequences for training
#     - X_val: Input sequences for validation
#     - Y_val: Target sequences for validation
#     Returns:
#     - model: The trained model
#     - history: Training history
#     """
#     ######### splitting the Attribute data
#     train_attribute, val_attribute, test_attribute = dataSplit(split, attribute_data)
#     train_Attribute, val_Attribute, test_Attribute, split = min_max(train_attribute.values, val_attribute.values, test_attribute.values, split) 
#     # Creating the X and Y for attribute forecasting (training), validation & testing
#     X_attribute_train, Y_attribute_train = create_X_Y(train_Attribute, time_steps, num_nodes, forecast_len)
#     X_val, Y_val = create_X_Y(val_Attribute, time_steps, num_nodes, forecast_len)
#     X_test, Y_test = create_X_Y(test_Attribute, time_steps, num_nodes, forecast_len)
    
#     # Normalize adjacency matrix
#     adj_normalized = calculate_laplacian_astgcn(adjacency_matrix)
#     adj_normalized = tf.convert_to_tensor(adj_normalized, dtype=tf.float32)
#     adj_normalized = tf.reshape(adj_normalized, [num_nodes, num_nodes])
    
#     # Define the AST-GCN model architecture
#     inputs = Input(shape=(time_steps, 1, X_train.shape[-1]))
#     x = astgcnCell(63, adj_normalized, X_attribute_train, Y_attribute_train, num_nodes)(inputs)
#     x = Reshape((-1, time_steps * num_nodes ))(x)
#     x = LSTM(64, activation='relu', return_sequences=False)(x)
#     outputs = Dense(40, activation='linear')(x)
#     model = Model(inputs=inputs, outputs=outputs)
#     # Compile and train the T-GCN model
#     model.compile(optimizer='adam', 
#                   loss='mean_squared_error')
#     # Define callbacks for early stopping and model checkpointing
#     early_stop = EarlyStopping(monitor='val_loss', 
#                                mode='min', 
#                                patience=5)
#     checkpoint = ModelCheckpoint(filepath=save_File, 
#                                 save_weights_only=False, 
#                                 monitor='val_loss', 
#                                 verbose=1,
#                                 save_best_only=True,
#                                 mode='min', 
#                                 save_freq='epoch')
#     callback = [early_stop, checkpoint]            
#     ##### Print out sunmary of the model
#     model.summary()
#     # Reshape X & Y validation data
#     last_column_X = X_val[:, :, :, -1] 
#     X_val = np.repeat(np.expand_dims(last_column_X, axis=-1), 40, axis=-1) 
#     last_column_Y = Y_val[:, -1] 
#     Y_val = np.repeat(np.expand_dims(last_column_Y, axis=-1), 40, axis=-1)
#     # ########## Training the model
#     history = model.fit(X_train, Y_train,
#                     validation_data=(X_val, Y_val),
#                     batch_size=196,
#                     epochs=1,
#                     verbose=1,
#                     callbacks=callback)
    
#     # Predict validation data
#     y_pred = model.predict(X_val)
#     # print("Y_pred predicted output shape:", y_pred.shape)
#     return model, history
       
# # This function creates the custom ASTGCN cell.         
# # This layer performs a graph convolution operation based on an adjacency matrix, followed by a GRU operation. 
# # It is intended to work with spatio-temporal data, with the graph convolutional operation accounting 
# # for spatial dependencies and the GRU for temporal dependencies. 
# # The function also overrides the compute_output_shape and get_config methods to support its custom operations.
# def astgcnCell(units, adj, X_attribute, Y_attribute, num_nodes):
#     """
#     Custom ASTGCN cell.
#     Parameters:
#     - units: Number of units in the GRU layer
#     - adj: Normalized adjacency matrix
#     - X_attribute: Node attribute sequences
#     - Y_attribute: Target sequences for node attributes
#     - num_nodes: Number of nodes in the graph
#     Returns:
#     - GcnCell: The custom layer
#     """
#     class GcnCell(tf.keras.layers.Layer):
#         """
#         Custom layer for the ASTGCN cell.
#         """
#         def __init__(self, units, adj, X_attribute, Y_attribute):
#             super(GcnCell, self).__init__()
#             self.units = units
#             self.adj = adj
#             self.X_attribute = X_attribute
#             self.Y_attribute = Y_attribute
#             self.layer = tf.keras.layers.GRU(self.units, return_sequences=True)
#             self.dense = tf.keras.layers.Dense(40)

#         def call(self, inputs):
#             """
#             Forward pass of the layer.
#             """
#             inputs_with_attributes = self.dense(inputs)
#             inputs_with_attributes = tf.squeeze(inputs_with_attributes, axis=2)  # Remove the extra dimension

#             adj_normalized_tiled = tf.expand_dims(self.adj, axis=0)
#             adj_normalized_tiled = tf.tile(adj_normalized_tiled, [tf.shape(inputs)[0], 1, 1])

#             output = self.layer(inputs_with_attributes)

#             reshaped_output = tf.reshape(output, [-1, tf.shape(inputs)[1], self.units])

#             return reshaped_output

#         def compute_output_shape(self, input_shape):
#             """
#             Compute output shape of the layer.
#             """
#             return input_shape[:-1] + (self.units,)

#         def get_config(self):
#             """
#             Get configuration of the layer.
#             """
#             config = super().get_config().copy()
#             config.update({
#                 'units': self.units,
#                 'adj': self.adj.numpy().tolist(),
#                 'X_attribute': self.X_attribute,
#                 'Y_attribute': self.Y_attribute
#             })
#             return config
        
#     adj_normalized = calculate_laplacian_astgcn(adj)
#     adj_normalized = tf.convert_to_tensor(adj_normalized, dtype=tf.float32)
#     adj_normalized = tf.sparse.reorder(tf.sparse.SparseTensor(indices=tf.where(adj_normalized != 0),
#                                                               values=tf.gather_nd(adj_normalized, tf.where(adj_normalized != 0)),
#                                                               dense_shape=adj_normalized.shape))
#     adj_normalized = tf.sparse.to_dense(adj_normalized)
#     adj_normalized = tf.reshape(adj_normalized, [num_nodes, num_nodes])
#     return GcnCell(units, adj_normalized, X_attribute, Y_attribute)