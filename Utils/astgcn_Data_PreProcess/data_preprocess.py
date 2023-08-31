import Utils.astgcnUtils as utils
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from geopy.distance import geodesic
from datetime import datetime,timedelta

###############################  AST-GCN pre-process methods  ###############################

def data_preprocess_AST_GCN(station):
    # Load and preprocess the weather station data & attribute data 
    # station_name = 'DataNew/Weather Station Data/' + station + '.csv'
    weather_data = pd.read_csv('DataNew/Graph Neural Network Data/Graph Station Data/graph.csv')
    # weather_data = pd.read_csv(station_name)
    # processed_data = weather_data[['Pressure', 'Humidity', 'Rain', 'Temperature']]
    processed_data = weather_data.drop(['StasName', 'DateT', 'Latitude', 'Longitude', 'WindDir', 'WindSpeed'], axis=1) 
    processed_data = np.array(processed_data) 

    # processed_data = processed_data.astype(float)
    # print(f"Type of processed_data: {type(processed_data)}")
    # print(f"Shape of processed_data: {processed_data.shape}")
    processed_data_final = np.reshape(processed_data, (113929, 45, 4))
    # print("Successfully preccessed input data")
    
    attribute_data = weather_data[['WindDir', 'WindSpeed']]  # Extract attribute data
    attribute_data = np.array(attribute_data) 
    attribute_data_final = np.reshape(attribute_data, (113929, 45, 2))
   
    # Adjust weather station nodes and adjacency matrix
    weather_stations = weather_data['StasName'].unique()
    num_nodes = len(weather_stations)
    
    # adjacency_matrix = pd.read_csv('data/Graph Neural Network Data/Aπdjacency Matrix/adj_mx.csv', index_col=0)
    # adjacency_matrix = adjacency_matrix.iloc[:num_nodeSs, :num_nodes].values
  
    # Already extracted adj matrix before hand
    # Extract station coordinates
    # stations_coords = weather_data.groupby('StasName')[['Latitude', 'Longitude']].first().values
    # adjacency_matrix = calculate_adjacency_matrix(stations_coords,1000)
    # print("Stations Coordinates:\n", stations_coords)
    adjacency_matrix = random_adjacency_matrix(num_nodes)
    
    return processed_data_final, attribute_data_final, adjacency_matrix, num_nodes

### Using euclidean distances for adj mx
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


def random_adjacency_matrix(num_stations, threshold=0.5):
    # Generate random values between 0 and 1
    matrix = np.random.rand(num_stations, num_stations)
    # Set values below threshold to 0 and above threshold to 1 to ensure binary matrix
    matrix[matrix < threshold] = 0
    matrix[matrix >= threshold] = 1
    # Ensure zero diagonal (no self-connections)
    np.fill_diagonal(matrix, 0)
    return matrix

def sliding_window_AST_GCN(processed_data, time_steps, num_nodes):
    input_data = []
    target_data = []
    # Iterate over the processed data to create input-target pairs
    # It iterates over the processed data and creates a sliding window of length time_steps over the data.
    # For each window, it creates an input sequence (input_data) and the corresponding target value (target_data).
    for i in range(len(processed_data) - time_steps):
        input_data.append(processed_data[i:i+time_steps])
        target_data.append(processed_data[i+time_steps])

    input_data = np.array(input_data)
    target_data = np.array(target_data)    
    
    return input_data, target_data

def get_timestamp_at_index(hours):
        # Create a datetime object
        date_string = "2010-01-01 00:00:00"
        formatted_date = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
        # Number of hours to add
        hours_to_add = hours//45

        # Create a timedelta representing the number of hours to add
        time_delta = timedelta(hours=hours_to_add)

        # Add the timedelta to the original date
        new_date = formatted_date + time_delta
        return new_date