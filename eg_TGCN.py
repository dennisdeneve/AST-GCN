import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense, LSTM, Reshape
from tensorflow.keras.models import Model

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
    adj_normalized = calculate_laplacian(adj)
    adj_normalized = tf.convert_to_tensor(adj_normalized, dtype=tf.float32)

    adj_normalized = tf.sparse.reorder(tf.sparse.SparseTensor(indices=tf.where(adj_normalized != 0),
                                                              values=tf.gather_nd(adj_normalized, tf.where(adj_normalized != 0)),
                                                              dense_shape=adj_normalized.shape))

    adj_normalized = tf.sparse.to_dense(adj_normalized)

    adj_normalized = tf.reshape(adj_normalized, [num_nodes, num_nodes])

    def call(inputs):
        adj_normalized_tiled = tf.expand_dims(adj_normalized, axis=0)
        adj_normalized_tiled = tf.tile(adj_normalized_tiled, [tf.shape(inputs)[0], 1, 1])

        cell = tf.keras.layers.GRU(units, return_sequences=True)(inputs)

        return cell

    return call
####################### Method from TCNUtils module ####################
def dataSplit(split, series):
    """
    From TCN utils - Splits the data into train, validation, test sets for walk-forward validation.
    Parameters:
        split - points at which to split the data into train, validation and test sets
        series - weather station data
    Returns:
        train, validation, test - returns the train, validation and test sets
    """

    train = series[0:split[0]]
    validation = series[split[0]:split[1]]
    test = series[split[1]:split[2]]

    return train, validation, test
####################### Method from TCNUtils module ####################
from sklearn.preprocessing import MinMaxScaler
def min_max(train, validation, test):
    """
    Performs MinMax scaling on the train, validation and test sets using the train data min and max.
    Parameters:
        train, validation, test - train, validation and test data sets
    Returns:
        train, validation, test - returns the scaled train, validation and test sets
    """

    norm = MinMaxScaler().fit(train)

    train_data = norm.transform(train)
    val_data = norm.transform(validation)
    test_data = norm.transform(test)

    return train_data, val_data, test_data
####################### tcn utils #######################
from sklearn.utils import shuffle
def create_X_Y(ts: np.array, lag=1, n_ahead=1, target_index=0):
    """
    A method to create X and Y matrix from a time series array.
    Parameters:
        ts - time series array
        lag - length of input sequence
        n_ahead - length of output sequence(forecasting horizon)
        target_index - index to be used as output target(Temperature)
    """
    n_features = ts.shape[1]
    X, Y = [], []

    if len(ts) - lag <= 0:
        X.append(ts)
    else:
        for i in range(len(ts) - lag - n_ahead):
            Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])
            X.append(ts[i:(i + lag)])
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], lag, n_features))
    x, y = shuffle(X, Y, random_state=0)
    return x, y
## from tcn utils
# def create_dataset(station):
#     """
#     Creates a dataset from the original weather station data
#     Parameters:
#         station - which station's data to read in
#     Returns:
#         data - returns dataframe with selected features
#     """
#     df = pd.read_csv(station)
#     features_final = df[['Temperature', 'Pressure', 'WindSpeed', 'WindDir', 'Humidity', 'Rain']]
#     data = df[features_final]
#     return data

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
        station = 'ADDO ELEPHANT PARK'  # Replace with your actual data path
        # Load the weather station data using create_dataset method
        #weather_data = create_dataset(station)
        station_name = station + '.csv'
        weather_data = pd.read_csv(station_name)
        # Preprocess the weather station data
        processed_data = weather_data[['Pressure', 'WindDir', 'WindSpeed', 'Humidity', 'Rain', 'Temperature']]
        processed_data = processed_data.astype(float)

        # Step 2: Adjust weather station nodes and adjacency matrix
        weather_stations = weather_data['StasName'].unique()
        num_nodes = len(weather_stations)
        print("Num_nodes (len Weather stations) : ", num_nodes)
        
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

        # Step 3: Reshape and preprocess the input data
        time_steps = 10  # Number of time steps to consider
        input_data = []
        target_data = []

        # Iterate over the processed data to create input-target pairs
        for i in range(len(processed_data) - time_steps):
            input_data.append(processed_data.iloc[i:i+time_steps].values)
            target_data.append(processed_data.iloc[i+time_steps].values)

        # Convert the input and target data to NumPy arrays
        input_data = np.array(input_data)
        target_data = np.array(target_data)

        # Reshape the input data to match the desired shape of the model
        input_data = input_data.reshape(-1, time_steps, num_nodes, 6)

        # Normalize the input and target data if necessary
        scaler = StandardScaler()
        input_data = input_data.reshape(-1, num_nodes * 6)
        input_data = scaler.fit_transform(input_data)
        input_data = input_data.reshape(-1, time_steps, num_nodes, 6)
        target_data = scaler.transform(target_data)

        for k in range(num_splits):
            print('TCN training started on split {0}/{3} at {1} station forecasting {2} hours ahead.'.format(k+1, station,
                                                                                                        forecast_len, num_splits))
            # splitting the processed time series data
            split = [increment[k], increment[k + 1], increment[k + 2]]
            pre_standardize_train, pre_standardize_validation, pre_standardize_test = dataSplit(split, processed_data)

            # Scaling the data
            train, validation, test = min_max(pre_standardize_train,
                                                    pre_standardize_validation,
                                                    pre_standardize_test)
            
            # Defining input shape
            n_ft = train.shape[1]        
            # Creating the X and Y for forecasting
            X_train, Y_train = create_X_Y(train, time_steps, n_ahead_length)
            # Creating the X and Y for validation set
            X_val, Y_val = create_X_Y(validation, time_steps, n_ahead_length)
            # Get the X feature set for training
            X_test, Y_test = create_X_Y(test, time_steps, n_ahead_length)

            # Step 4: Define the T-GCN model architecture
            inputs = Input(shape=(time_steps, num_nodes, 6))
            x = Reshape((time_steps, num_nodes * 6))(inputs)
            cell_1 = tgcnCell(64, adjacency_matrix, num_nodes)(x)
            cell_1_reshaped = Reshape((time_steps, num_nodes * 64))(cell_1)
            x = LSTM(64, activation='relu', return_sequences=True)(cell_1_reshaped)
            outputs = Dense(num_nodes * 6)(x)
            outputs_reshaped = Reshape((time_steps, num_nodes, 6))(outputs)
            model = Model(inputs=inputs, outputs=outputs_reshaped)

            # Step 5: Compile and train the T-GCN model
            model.compile(optimizer='adam', loss='mse')
            history = model.fit(input_data, 
                                target_data, 
                                epochs=1, 
                                batch_size=196)
            
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

            new_data = new_data.astype(float)
            new_data = np.expand_dims(new_data, axis=0)  # Add batch dimension
            new_data = np.expand_dims(new_data, axis=2)  # Add node dimension
            new_data = new_data.reshape(-1, time_steps, 1, 6)
            predictions = model.predict(new_data)
            predictions = scaler.inverse_transform(predictions.reshape(-1, num_nodes * 6))

            # Print the predicted temperature for a specific weather station
            station_index = 0  # Replace with the index of the desired weather station
            station = weather_stations[station_index]
            temperature_prediction = predictions[0][station_index * 6 + 5]  
            print(f'Predicted temperature at {station}: {temperature_prediction}')
            
        print('tgcnTrain : TGCN training finished at ', station)     
        # Save the results to the file
        resultsDF.to_csv(resultsFile)
        resultsDF.to_csv(resultsFile)
        lossDF.to_csv(lossFile)
        targetDF.to_csv(targetFile)
        
    print("TGCN training finished for all stations at all splits ! :)")
    
def main():
    
    print("Started : ")
    trainTGCN()

if __name__ == '__main__':
    main()
