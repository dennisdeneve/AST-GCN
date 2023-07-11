import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense, LSTM, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
###################### Method from tgcn utils ####################
def calculate_laplacian(adj):
    adj_normalized = normalize_adj(adj + np.eye(adj.shape[0]))
    return adj_normalized
###################### Method from tgcn utils ####################
def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = np.diag(r_inv)
    adj_normalized = adj.dot(r_mat_inv).transpose().dot(r_mat_inv)
    return adj_normalized
#################### Method from model, tgcn ####################
def tgcnCell(units, adj, num_nodes):
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
# def tgcnCell(units, adj, num_nodes):

#     class GcnCell(tf.keras.layers.Layer):
#         def __init__(self, units, adj):
#             super(GcnCell, self).__init__()
#             self.units = units
#             self.adj = adj

#         def build(self, input_shape):
#             self.layer = tf.keras.layers.GRU(self.units, return_sequences=True)
#             self.layer.build(input_shape)

#         def call(self, inputs):
#             adj_normalized_tiled = tf.expand_dims(self.adj, axis=0)
#             adj_normalized_tiled = tf.tile(adj_normalized_tiled, [tf.shape(inputs)[0], 1, 1])
#             return self.layer(inputs)

#         def compute_output_shape(self, input_shape):
#             return input_shape[:-1] + (self.units,)

#     adj_normalized = calculate_laplacian(adj)
#     adj_normalized = tf.convert_to_tensor(adj_normalized, dtype=tf.float32)

#     adj_normalized = tf.sparse.reorder(tf.sparse.SparseTensor(indices=tf.where(adj_normalized != 0),
#                                                               values=tf.gather_nd(adj_normalized, tf.where(adj_normalized != 0)),
#                                                               dense_shape=adj_normalized.shape))
#     adj_normalized = tf.sparse.to_dense(adj_normalized)
#     adj_normalized = tf.reshape(adj_normalized, [num_nodes, num_nodes])
#     return GcnCell(units, adj_normalized)
####################### Method from TCNUtils module ####################
def dataSplit(split, series):
    train = series[0:split[0]]
    validation = series[split[0]:split[1]]
    test = series[split[1]:split[2]]
    return train, validation, test
####################### Method from TCNUtils module ####################
from sklearn.preprocessing import MinMaxScaler
def min_max(train, validation, test):
    norm = MinMaxScaler().fit(train.reshape(train.shape[0], -1))
    train_data = norm.transform(train.reshape(train.shape[0], -1))
    val_data = norm.transform(validation.reshape(validation.shape[0], -1))
    test_data = norm.transform(test.reshape(test.shape[0], -1))
    return train_data, val_data, test_data
####################### tcn utils #######################
def create_X_Y(ts: np.array, lag=1, num_nodes=1, n_ahead=1, target_index=0):
    X, Y = [], []
    if len(ts) - lag - n_ahead + 1 <= 0:
        X.append(ts)
        Y.append(ts)
    else:
        for i in range(len(ts) - lag - n_ahead + 1):
            X.append(ts[i:(i + lag)])
            Y.append(ts[i + lag + n_ahead - 1])  # Append only the target value for Y
    X, Y = np.array(X), np.array(Y)

    num_samples = len(X)
    time_steps = 10
    num_samples -= num_samples % time_steps
    X = X[:num_samples]
    Y = Y[:num_samples]

    ### Reshaping to match the TGCN model output architecture
    # Y = np.repeat(Y, repeats=time_steps, axis=0)  # Repeat Y to match the number of time steps in X
    X = np.expand_dims(X, axis=2)
    Y = np.reshape(Y, (Y.shape[0], -1))  # Reshape Y to match the shape of y_pred
    return X, Y

def trainTGCN():
    increment =[100,200,300,8760, 10920, 13106, 15312, 17520, 19704, 21888, 24096, 26304,
                    28464, 30648, 32856, 35064, 37224, 39408, 41616, 43824, 45984,
                    48168, 50376, 52584, 54768, 56952, 59160, 61368, 63528, 65712,
                    67920, 70128, 72288, 74472, 76680, 78888, 81048, 83232, 85440,
                    87648, 89832, 92016, 94224, 96432, 98592, 100776, 102984, 105192,
                    107352, 109536, 111744, 113929]
    forecasting_horizons = [3, 6, 9, 12, 24]
    num_splits =1
    for forecast_len in forecasting_horizons:
        # Step 1: Load and preprocess the data
        station = 'ADDO ELEPHANT PARK'  
        station_name = station + '.csv'
        weather_data = pd.read_csv(station_name)
        # Preprocess the weather station data
        processed_data = weather_data[['Pressure', 'WindDir', 'WindSpeed', 'Humidity', 'Rain', 'Temperature']]
        processed_data = processed_data.astype(float)
        # Step 2: Adjust weather station nodes and adjacency matrix
        weather_stations = weather_data['StasName'].unique()
        num_nodes = len(weather_stations)
        
        adjacency_matrix = pd.read_csv('adj_mx.csv', index_col=0)
        adjacency_matrix = adjacency_matrix.iloc[:num_nodes, :num_nodes].values
        # This setting changes for each of the forecast_len in the above list for the horizon, thus not in config file
        n_ahead_length = forecast_len
                    
        lossDF = pd.DataFrame()
        resultsDF = pd.DataFrame()
        targetDF = pd.DataFrame()
        targetFile = 'Results/TGCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Targets/' + \
                                'target.csv'
        resultsFile = 'Results/TGCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Predictions/' + \
                                'result.csv'
        lossFile = 'Results/TGCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Predictions/' + \
                            'loss.csv'
        time_steps = 10  # Number of time steps to consider
        input_data = []
        target_data = []

        # Iterate over the processed data to create input-target pairs
        # It iterates over the processed data and creates a sliding window of length time_steps over the data.
        # For each window, it creates an input sequence (input_data) and the corresponding target value (target_data).
        for i in range(len(processed_data) - time_steps):
            input_data.append(processed_data.iloc[i:i+time_steps].values)
            target_data.append(processed_data.iloc[i+time_steps].values)

        # Convert the input and target data to NumPy arrays
        input_data = np.array(input_data)
        target_data = np.array(target_data)
        
        ## Reshape the input data to match the desired shape of the model
        input_data = input_data.transpose((0, 2, 1))  # Swap the time_steps and num_nodes dimensions
        input_data = input_data.reshape(-1, num_nodes, time_steps * 6)  # Reshape to (batch_size, num_nodes, time_steps * 6)

        # Normalize the input and target data if necessary
        scaler = StandardScaler()
        input_data = input_data.reshape(-1, num_nodes * 6)
        input_data = scaler.fit_transform(input_data)
        input_data = input_data.reshape(-1, time_steps, num_nodes, 6)

        target_data = scaler.transform(target_data)

        # Adjust the shape of the input and target data
        input_data = np.transpose(input_data, (0, 2, 1, 3))  # Swap the time_steps and num_nodes dimensions
        target_data = np.reshape(target_data, (target_data.shape[0], -1))


        for k in range(num_splits):
            print('TCN training started on split {0}/{3} at {1} station forecasting {2} hours ahead.'.format(k+1, station,
                                                                                                        forecast_len, num_splits))
            
            save_File = 'Garage/Final Models/TCN/' + station + '/' + str(forecast_len) + ' Hour Models/Best_Model_' \
                           + str(n_ahead_length) + '_walk_' + str(k) + '.h5'
                           
            # splitting the processed time series data
            split = [increment[k], increment[k + 1], increment[k + 2]]
            pre_standardize_train, pre_standardize_validation, pre_standardize_test = dataSplit(split, input_data)
            # Scaling the data
            train, validation, test = min_max(pre_standardize_train,
                                                    pre_standardize_validation,
                                                    pre_standardize_test)     
            # Creating the X and Y for forecasting
            X_train, Y_train = create_X_Y(train, time_steps, num_nodes, n_ahead_length)
             # Creating the X and Y for validation set
            X_val, Y_val = create_X_Y(validation, time_steps, num_nodes, n_ahead_length)
             # Get the X feature set for testing
            X_test, Y_test = create_X_Y(test, time_steps, num_nodes, n_ahead_length)
            print("start X Train shape ",X_train.shape)
            print("start Y Train shape ",Y_train.shape)
            print("start X Val shape ",X_val.shape)
            print("start Y Val shape ",Y_val.shape)
        
            # Step 4: Define the T-GCN model architecture
            # inputs = Input(shape=(time_steps, 1, num_nodes * 60))  # Update the input shape
            # x = tf.keras.layers.TimeDistributed(tgcnCell(64, adjacency_matrix, num_nodes))(inputs)
            # x = Reshape((-1, 10 * 64))(x)  # Reshape into 3D tensor
            # x = LSTM(64, activation='relu', return_sequences=True)(x)
            # # outputs = Dense(forecast_len * num_nodes * 60, activation='linear')(x)
            # outputs = Dense(60, activation='linear')(x)
            # model = Model(inputs=inputs, outputs=outputs)
            inputs = Input(shape=(time_steps, 1, num_nodes * 60))  # Update the input shape
            x = tf.keras.layers.TimeDistributed(tgcnCell(64, adjacency_matrix, num_nodes))(inputs)
            x = Reshape((-1, 10 * 64))(x)  # Reshape into 3D tensor
            x = LSTM(64, activation='relu', return_sequences=True)(x)
            # outputs = Dense(forecast_len * num_nodes * 60, activation='linear')(x)
            outputs = Dense(60, activation='linear')(x)
            model = Model(inputs=inputs, outputs=outputs)
            
            

            # Step 5: Compile and train the T-GCN model
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Define callbacks for early stopping and model checkpointing
            early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=5)
            checkpoint = ModelCheckpoint(filepath=save_File, save_weights_only=False, monitor='val_loss', verbose=1,
                                        save_best_only=True,
                                        mode='min', save_freq='epoch')
            callback = [early_stop, checkpoint]            


            print("Final X Train shape ",X_train.shape)
            print("Final Y Train shape ",Y_train.shape)
            print("Final X Val shape ",X_val.shape)
            print("Final Y Val shape ",Y_val.shape)
            
            ########## Training the model
            history = model.fit(X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    batch_size=196,
                    epochs=1,
                    verbose=1,
                    callbacks=callback)
            
            # validation and train loss to dataframe
            lossDF = lossDF.append([[history.history['loss']]])
            
            # Step 6: Use the trained model for predictions
            # Assuming you have new data for prediction stored in `new_data`
            new_data = pd.DataFrame({
                'Pressure': [997.5] * time_steps,
                'WindDir': [100.0] * time_steps,
                'WindSpeed': [2.0] * time_steps,
                'Humidity': [70.0] * time_steps,
                'Rain': [0.0] * time_steps,
                'Temperature': [25.5] * time_steps
            })
            # Replicate the columns 10 times
            new_data = pd.concat([new_data]*10, axis=1)

            # Ensure columns are in correct order
            new_data = new_data[sorted(new_data.columns)]
            
            new_data = new_data.astype(float)
            new_data = np.expand_dims(new_data, axis=0)  # Add batch dimension
            new_data = np.expand_dims(new_data, axis=2)  # Add node dimension
            new_data = new_data.reshape(-1, time_steps, 1, 60)
            predictions = model.predict(new_data)
            predictions = scaler.inverse_transform(predictions.reshape(-1, num_nodes * 6))
            yhat = model.predict(X_test)
            # predictions to dataframe
            resultsDF = pd.concat([resultsDF, pd.Series(yhat.reshape(-1, ))])
            
            Y_test = np.expand_dims(Y_test, axis=2)  # Add the missing dimension
            targetDF = pd.concat([targetDF, pd.Series(Y_test.reshape(-1, ))])
        
            # Print the predicted temperature for a specific weather station
            station_index = 0  # Replace with the index of the desired weather station
            station = weather_stations[station_index]
            temperature_prediction = predictions[0][station_index * 6 + 5]  
            print(f'Predicted temperature at {station}: {temperature_prediction}')
            
        print('TCN training finished on split {0}/{3} at {1} station forecasting {2} hours ahead.'.format(k+1, station,
                                                                                                        forecast_len, num_splits))   
        # Save the results to the file
        resultsDF.to_csv(resultsFile)
        lossDF.to_csv(lossFile)
        targetDF.to_csv(targetFile)
    print("TGCN training finished for all stations at all splits ! :)")