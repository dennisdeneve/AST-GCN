import tensorflow as tf
import numpy as np
from Utils.utils import calculate_laplacian_astgcn
from tensorflow.keras.layers import Input, Dense, LSTM, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from Utils.utils import create_X_Y, min_max, dataSplit

#  This function constructs the model architecture, compiles it, and trains it. 
# The architecture includes a time-distributed ASTGCN cell followed by an LSTM layer and a Dense layer. 
# The model is compiled with Adam optimizer and Mean Squared Error loss function,
# which is commonly used for regression problems. 
# Early stopping and model checkpoints are also used during training.
def astgcnModel(time_steps, num_nodes, adjacency_matrix, 
                attribute_data, save_File, forecast_len,increment,
                X_train, Y_train, X_val, Y_val
               ):
    """
    Build and train the temperature model.
    Returns:
    model (Sequential): The trained temperature model.
    history (History): The training history.
    """
   # splitting the processed time series data
    num_samples = len(attribute_data)
    train_split = int(num_samples * 0.7)  # 70% of the data for training
    val_split = train_split + int(num_samples * 0.15)  # 15% of the data for validation
    train_attribute, val_attribute, test_attribute = dataSplit([train_split, val_split, num_samples], attribute_data)
    train_Attribute = train_attribute.values
    val_Attribute = val_attribute.values
    test_Attribute = test_attribute.values
    train_Attribute, val_Attribute, test_Attribute = min_max(train_Attribute, val_Attribute, test_Attribute) 
    # Creating the X and Y for forecasting (training), validation & testing
    X_attribute_train, Y_attribute_train = create_X_Y(train_Attribute, time_steps, num_nodes, forecast_len)
    X_val, Y_val = create_X_Y(val_Attribute, time_steps, num_nodes, forecast_len)
    X_test, Y_test = create_X_Y(test_Attribute, time_steps, num_nodes, forecast_len)
    
    # print("Final X Train shape ",X_train.shape)
    # print("Final Y Train shape ",Y_train.shape)
    # print("Attribute  X Train shape ",X_attribute_train.shape)
    # print("Attribute Y Train shape ",Y_attribute_train.shape)
    
    adj_normalized = calculate_laplacian_astgcn(adjacency_matrix)
    adj_normalized = tf.convert_to_tensor(adj_normalized, dtype=tf.float32)
    adj_normalized = tf.reshape(adj_normalized, [num_nodes, num_nodes])
    
    # Step 4: Define the AST-GCN model architecture
    # inputs = Input(shape=(time_steps, 1, num_nodes * 2)) 
    inputs = Input(shape=(time_steps, 1, X_train.shape[-1]))
    x = astgcnCell(63, adj_normalized, X_attribute_train, Y_attribute_train, num_nodes)(inputs)
    x = Reshape((-1, time_steps * num_nodes ))(x)
    x = LSTM(64, activation='relu', return_sequences=False)(x)
    outputs = Dense(40, activation='linear')(x)
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
    tf.config.run_functions_eagerly(True)
    model.summary()
    
    last_column_X = X_val[:, :, :, -1]  # Extract the last column
    X_val = np.repeat(np.expand_dims(last_column_X, axis=-1), 40, axis=-1)  # Repeat the last column to match (10, 1, 40)
    last_column_Y = Y_val[:, -1]  # Extract the last column
    Y_val = np.repeat(np.expand_dims(last_column_Y, axis=-1), 40, axis=-1)
    # print("Aligned Shapes:")
    # print("X_train shape:", X_train.shape)
    # print("Y_train shape:", Y_train.shape)
    # print("X_val shape:", X_val.shape)
    # print("Y_val shape:", Y_val.shape)
       
    # ########## Training the model
    history = model.fit(X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    batch_size=196,
                    epochs=1,
                    verbose=1,
                    callbacks=callback)
    
    # Obtain predicted output
    y_pred = model.predict(X_val)
    # print("Y_pred predicted output shape:", y_pred.shape)
    return model, history
       
# This function creates the custom ASTGCN cell. 
# Inside this cell, the adjacency matrix is converted into a Laplacian and 
# combined with the inputs and the attribute data to form the inputs for a GRU layer.
# The GRU layer's outputs are then reshaped and returned. 
# The compute_output_shape() and get_config() methods are also overridden to support the 
# custom operations in the layer.          
def astgcnCell(units, adj, X_attribute, Y_attribute, num_nodes):
    class GcnCell(tf.keras.layers.Layer):
        def __init__(self, units, adj, X_attribute, Y_attribute):
            super(GcnCell, self).__init__()
            self.units = units
            self.adj = adj
            self.X_attribute = X_attribute
            self.Y_attribute = Y_attribute
            self.layer = tf.keras.layers.GRU(self.units, return_sequences=True)
            self.dense = tf.keras.layers.Dense(40)

        def call(self, inputs):
            inputs_with_attributes = self.dense(inputs)
            inputs_with_attributes = tf.squeeze(inputs_with_attributes, axis=2)  # Remove the extra dimension

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

    adj_normalized = calculate_laplacian_astgcn(adj)
    adj_normalized = tf.convert_to_tensor(adj_normalized, dtype=tf.float32)

    adj_normalized = tf.sparse.reorder(tf.sparse.SparseTensor(indices=tf.where(adj_normalized != 0),
                                                              values=tf.gather_nd(adj_normalized, tf.where(adj_normalized != 0)),
                                                              dense_shape=adj_normalized.shape))
    adj_normalized = tf.sparse.to_dense(adj_normalized)
    adj_normalized = tf.reshape(adj_normalized, [num_nodes, num_nodes])
    return GcnCell(units, adj_normalized, X_attribute, Y_attribute)