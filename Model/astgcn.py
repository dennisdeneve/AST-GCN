import numpy as np
import tensorflow as tf
from Utils.utils import calculate_laplacian_astgcn
from tensorflow.keras.layers import Input, Dense, LSTM, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from Utils.utils import create_X_Y, min_max, dataSplit
tf.config.experimental_run_functions_eagerly(False)

# This function creates, compiles, and trains a st-gnn model for temperature forecasting.
# It first prepares the data by splitting, normalizing, and reshaping it as necessary. 
# The model architecture includes a custom ASTGCN cell, an LSTM layer, and a Dense layer. 
# The model uses the Adam optimizer and mean squared error as the loss function, and 
# implements early stopping and model checkpointing during training.
def astgcnModel(time_steps, num_nodes, adjacency_matrix, 
                attribute_data, save_File, forecast_len,increment,
                X_train, Y_train, X_val, Y_val
               ):    
    """
    Build and train the ASTGCN model.
    Parameters:
    - time_steps: Number of time steps in each input sequence
    - num_nodes: Number of nodes in the graph
    - adjacency_matrix: Adjacency matrix of the graph
    - attribute_data: Node attribute data
    - save_File: File path to save the model
    - forecast_len: Forecast length (in time steps)
    - increment: The increment to use when creating sequences
    - X_train: Input sequences for training
    - Y_train: Target sequences for training
    - X_val: Input sequences for validation
    - Y_val: Target sequences for validation
    Returns:
    - model: The trained model
    - history: Training history
    """
   # splitting the processed time series data
    num_samples = len(attribute_data)
    train_split = int(num_samples * 0.7)  # 70% of the data for training
    val_split = train_split + int(num_samples * 0.15)  # 15% of the data for validation
    train_attribute, val_attribute, test_attribute = dataSplit([train_split, val_split, num_samples], attribute_data)
    train_Attribute, val_Attribute, test_Attribute = min_max(train_attribute.values, val_attribute.values, test_attribute.values) 
    # Creating the X and Y for forecasting (training), validation & testing
    X_attribute_train, Y_attribute_train = create_X_Y(train_Attribute, time_steps, num_nodes, forecast_len)
    X_val, Y_val = create_X_Y(val_Attribute, time_steps, num_nodes, forecast_len)
    X_test, Y_test = create_X_Y(test_Attribute, time_steps, num_nodes, forecast_len)
    
    # Normalize adjacency matrix
    adj_normalized = calculate_laplacian_astgcn(adjacency_matrix)
    adj_normalized = tf.convert_to_tensor(adj_normalized, dtype=tf.float32)
    adj_normalized = tf.reshape(adj_normalized, [num_nodes, num_nodes])
    
    # Define the AST-GCN model architecture
    inputs = Input(shape=(time_steps, 1, X_train.shape[-1]))
    x = astgcnCell(63, adj_normalized, X_attribute_train, Y_attribute_train, num_nodes)(inputs)
    x = Reshape((-1, time_steps * num_nodes ))(x)
    x = LSTM(64, activation='relu', return_sequences=False)(x)
    outputs = Dense(40, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    # Compile and train the T-GCN model
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
    tf.config.run_functions_eagerly(True)
    model.summary()
    # Reshape validation data
    last_column_X = X_val[:, :, :, -1]  # Extract the last column
    X_val = np.repeat(np.expand_dims(last_column_X, axis=-1), 40, axis=-1)  # Repeat the last column to match (10, 1, 40)
    last_column_Y = Y_val[:, -1]  # Extract the last column
    Y_val = np.repeat(np.expand_dims(last_column_Y, axis=-1), 40, axis=-1)
    # ########## Training the model
    history = model.fit(X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    batch_size=196,
                    epochs=1,
                    verbose=1,
                    callbacks=callback)
    
    # Predict validation data
    y_pred = model.predict(X_val)
    # print("Y_pred predicted output shape:", y_pred.shape)
    return model, history
       
# This function creates the custom ASTGCN cell.         
# This layer performs a graph convolution operation based on an adjacency matrix, followed by a GRU operation. 
# It is intended to work with spatio-temporal data, with the graph convolutional operation accounting 
# for spatial dependencies and the GRU for temporal dependencies. 
# The function also overrides the compute_output_shape and get_config methods to support its custom operations.
def astgcnCell(units, adj, X_attribute, Y_attribute, num_nodes):
    """
    Custom ASTGCN cell.
    Parameters:
    - units: Number of units in the GRU layer
    - adj: Normalized adjacency matrix
    - X_attribute: Node attribute sequences
    - Y_attribute: Target sequences for node attributes
    - num_nodes: Number of nodes in the graph
    Returns:
    - GcnCell: The custom layer
    """
    class GcnCell(tf.keras.layers.Layer):
        """
        Custom layer for the ASTGCN cell.
        """
        def __init__(self, units, adj, X_attribute, Y_attribute):
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
            """
            inputs_with_attributes = self.dense(inputs)
            inputs_with_attributes = tf.squeeze(inputs_with_attributes, axis=2)  # Remove the extra dimension

            adj_normalized_tiled = tf.expand_dims(self.adj, axis=0)
            adj_normalized_tiled = tf.tile(adj_normalized_tiled, [tf.shape(inputs)[0], 1, 1])

            output = self.layer(inputs_with_attributes)

            reshaped_output = tf.reshape(output, [-1, tf.shape(inputs)[1], self.units])

            return reshaped_output

        def compute_output_shape(self, input_shape):
            """
            Compute output shape of the layer.
            """
            return input_shape[:-1] + (self.units,)

        def get_config(self):
            """
            Get configuration of the layer.
            """
            config = super().get_config().copy()
            config.update({
                'units': self.units,
                'adj': self.adj.numpy().tolist(),
                'X_attribute': self.X_attribute,
                'Y_attribute': self.Y_attribute
            })
            return config
        
    adj_normalized = calculate_laplacian_astgcn(adj)
    adj_normalized = tf.convert_to_tensor(adj_normalized, dtype=tf.float32)
    adj_normalized = tf.sparse.reorder(tf.sparse.SparseTensor(indices=tf.where(adj_normalized != 0),
                                                              values=tf.gather_nd(adj_normalized, tf.where(adj_normalized != 0)),
                                                              dense_shape=adj_normalized.shape))
    adj_normalized = tf.sparse.to_dense(adj_normalized)
    adj_normalized = tf.reshape(adj_normalized, [num_nodes, num_nodes])
    return GcnCell(units, adj_normalized, X_attribute, Y_attribute)