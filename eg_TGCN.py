import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense, LSTM, Reshape
from tensorflow.keras.models import Model

def calculate_laplacian(adj):
    adj_normalized = normalize_adj(adj + np.eye(adj.shape[0]))
    return adj_normalized

def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = np.diag(r_inv)
    adj_normalized = adj.dot(r_mat_inv).transpose().dot(r_mat_inv)
    return adj_normalized

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

# Step 1: Load and preprocess the data
weather_data_path = 'ADDO ELEPHANT PARK.csv'  # Replace with your actual data path

# Load the weather station data using pandas
weather_data = pd.read_csv(weather_data_path)

# Preprocess the weather station data
processed_data = weather_data[['Pressure', 'WindDir', 'WindSpeed', 'Humidity', 'Rain', 'Temperature']]
processed_data = processed_data.astype(float)

# Step 2: Adjust weather station nodes and adjacency matrix
weather_stations = weather_data['StasName'].unique()
num_nodes = len(weather_stations)
print("Num_nodes (len Weather stations) : ", num_nodes)

adjacency_matrix = pd.read_csv('adj_mx.csv', index_col=0)
adjacency_matrix = adjacency_matrix.iloc[:num_nodes, :num_nodes].values

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

model.fit(input_data, target_data, epochs=1, batch_size=196)

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

# print('Predicted temperatures:')
# for i, station in enumerate(weather_stations):
#     print(f'{station}: {predictions[0, -1, 0, i, -1]}')

# Print the predicted temperature for a specific weather station
station_index = 0  # Replace with the index of the desired weather station
station = weather_stations[station_index]
temperature_prediction = predictions[0][station_index * 6 + 5]  # Adjust the indexing
print(f'Predicted temperature at {station}: {temperature_prediction}')




# new_data = new_data.astype(float)
# new_data = new_data.values.reshape(1, time_steps, num_nodes, 6)
# new_data = scaler.transform(new_data)
# predictions = model.predict(new_data)
# predictions = scaler.inverse_transform(predictions.reshape(-1, num_nodes * 6))

# print('Predicted temperatures:')
# for i, station in enumerate(weather_stations):
#     print(f'{station}: {predictions[0][i * 6 + 5]}')


# # Step 4: Define the T-GCN model architecture
# inputs = Input(shape=(time_steps, num_nodes, 6))
# x = Reshape((time_steps, num_nodes * 6))(inputs)
# cell_1 = tgcnCell(64, adjacency_matrix, num_nodes)(x)
# x = LSTM(64, activation='relu', return_sequences=True)(cell_1)
# outputs = Dense(num_nodes * 6)(x)
# model = Model(inputs=inputs, outputs=outputs)