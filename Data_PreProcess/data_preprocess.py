import astgcnUtils.astgcnUtils as utils
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from geopy.distance import geodesic

###############################  AST-GCN pre-process methods  ###############################

def data_preprocess_AST_GCN(station):
    # Load and preprocess the weather station data & attribute data 
    station_name = 'data/Weather Station Data/' + station + '.csv'
    weather_data = pd.read_csv(station_name)
    processed_data = weather_data[['Pressure', 'Humidity', 'Rain', 'Temperature']]
    attribute_data = weather_data[['WindDir', 'WindSpeed']]  # Extract attribute data
    processed_data = processed_data.astype(float)
    # Adjust weather station nodes and adjacency matrix
    weather_stations = weather_data['StasName'].unique()
    num_nodes = len(weather_stations)
    adjacency_matrix = pd.read_csv('data/Graph Neural Network Data/Adjacency Matrix/adj_mx.csv', index_col=0)
    adjacency_matrix = adjacency_matrix.iloc[:num_nodes, :num_nodes].values
   
    # Already extracted adj matrix before hand
    # Extract station coordinates
    # stations_coords = weather_data.groupby('StasName')[['Latitude', 'Longitude']].first().values
    # adjacency_matrix = calculate_adjacency_matrix(stations_coords,1000)
    # print("Stations Coordinates:\n", stations_coords)
    
    return processed_data, attribute_data, adjacency_matrix, num_nodes

def calculate_adjacency_matrix(stations_coords, threshold=None, decay_factor=0.01):
    num_stations = len(stations_coords)
    adjacency_matrix = np.zeros((num_stations, num_stations))

    for i in range(num_stations):
        for j in range(num_stations):
            if i != j:  # exclude self-connections
                distance = geodesic(stations_coords[i], stations_coords[j]).kilometers
                decayed_distance = np.exp(-decay_factor * distance)
                
                if threshold:  # for binary connections
                    if distance < threshold:
                        adjacency_matrix[i][j] = 1
                else:  # for weighted connections
                    adjacency_matrix[i][j] = decayed_distance / (distance + 1e-5)  # added small value to avoid division by zero

    # Now, Z-score normalization for non-zero entries
    non_zero_entries = adjacency_matrix[adjacency_matrix != 0]
    mean_distance = np.mean(non_zero_entries)
    std_distance = np.std(non_zero_entries)
    adjacency_matrix[adjacency_matrix != 0] = (non_zero_entries - mean_distance) / std_distance

    return adjacency_matrix


def sliding_window_AST_GCN(processed_data, time_steps, num_nodes):
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
    input_data = input_data.reshape(-1, num_nodes, time_steps * 4)  
    # Normalize the input and target data if necessary, also reshape 
    scaler = StandardScaler()
    input_data = input_data.reshape(-1, num_nodes * 4)
    input_data = scaler.fit_transform(input_data)
    input_data = input_data.reshape(-1, time_steps, num_nodes, 4)
    target_data = scaler.transform(target_data)
    # Adjust the shape of the input and target data
    input_data = np.transpose(input_data, (0, 2, 1, 3))  # Swap the time_steps and num_nodes dimensions
    target_data = np.reshape(target_data, (target_data.shape[0], -1))
    
    return input_data, target_data, scaler



####  st-gcn version methods
def data_preprocess_ST_GCN(station):
    # Load and preprocess the weather station data
    station_name = 'data/Weather Station Data/' + station + '.csv'
    weather_data = pd.read_csv(station_name)
    processed_data = weather_data[['Pressure', 'WindDir', 'WindSpeed', 'Humidity', 'Rain', 'Temperature']]
    processed_data = processed_data.astype(float)
    #Adjust weather station nodes and adjacency matrix
    weather_stations = weather_data['StasName'].unique()
    adjacency_matrix = pd.read_csv('data/Graph Neural Network Data/Adjacency Matrix/adj_mx.csv', index_col=0)
    num_nodes = len(weather_stations)
    adjacency_matrix = adjacency_matrix.iloc[:num_nodes, :num_nodes].values
    return processed_data, adjacency_matrix, num_nodes


def sliding_window_ST_GCN(processed_data, time_steps, num_nodes):
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
    input_data = input_data.reshape(-1, num_nodes, time_steps * 6)  
    # Normalize the input and target data if necessary, also reshape 
    scaler = StandardScaler()
    input_data = input_data.reshape(-1, num_nodes * 6)
    input_data = scaler.fit_transform(input_data)
    input_data = input_data.reshape(-1, time_steps, num_nodes, 6)
    target_data = scaler.transform(target_data)
    # Adjust the shape of the input and target data
    input_data = np.transpose(input_data, (0, 2, 1, 3))  # Swap the time_steps and num_nodes dimensions
    target_data = np.reshape(target_data, (target_data.shape[0], -1))
    
    return input_data, target_data, scaler