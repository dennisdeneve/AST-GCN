from Evaluation.metrics import MaxMinNormalization
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

###############################  NEW DATA PRE-PROCESS METHODs THAT WWORK  ###############################
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

def data_preprocess_AST_GCN(station):
    # Load and preprocess the weather station data & attribute data 
    station_name = 'data/Weather Station Data/' + station + '.csv'
    weather_data = pd.read_csv(station_name)
    processed_data = weather_data[['Pressure', 'Humidity', 'Rain', 'Temperature']]
    attribute_data = weather_data[['WindDir', 'WindSpeed']]  # Extract attribute data
    processed_data = processed_data.astype(float)
    # Adjust weather station nodes and adjacency matrix
    weather_stations = weather_data['StasName'].unique()
    adjacency_matrix = pd.read_csv('data/Graph Neural Network Data/Adjacency Matrix/adj_mx.csv', index_col=0)
    num_nodes = len(weather_stations)
    adjacency_matrix = adjacency_matrix.iloc[:num_nodes, :num_nodes].values
    # print("Processed data:", processed_data)
    # print("Attribute data:", attribute_data)
    return processed_data, attribute_data, adjacency_matrix, num_nodes

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
    
#################  Old methods to pr-process data #################
def load_assist_data(station_file, adjacency_file):
    # Read the weather station CSV file into a Pandas DataFrame
    weather_data = pd.read_csv(station_file)
    
    # Read the adjacency matrix CSV file into a Pandas DataFrame
    adjacency_matrix = pd.read_csv(adjacency_file, header=None)
    
    # Return the weather station data and adjacency matrix
    return weather_data, adjacency_matrix

# def data_preprocess(config):    
#     noise_name = config['noise_name']['default']
#     data_name = config['dataset']['default']
#     PG =  config['noise_param']['default']
    
#     ########## load data #########
#     # if data_name == 'sz':
#     data, adj = load_assist_data('ADDO ELEPHANT PARK.csv', 'adj_mx.csv')
    
#     ## Applying different types of noise filter to data
#     if noise_name == 'Gauss':
#         # Generate Gaussian noise using the np.random.normal() function with mean 0 and standard deviation PG
#         Gauss = np.random.normal(0, PG, size=data.shape)
#         noise_Gauss = MaxMinNormalization(Gauss, np.max(Gauss), np.min(Gauss))
#         data = data + noise_Gauss
#         return data 
#     elif noise_name == 'Possion':
#         # Generate Poisson noise using the np.random.poisson() function with mean PG
#         Possion = np.random.poisson(PG, size=data.shape)
#         noise_Possion = MaxMinNormalization(Possion, np.max(Possion), np.min(Possion))
#         data = data + noise_Possion
#         return data 
#     else:
#         # Return the unchanged data
#         return data 
     
def processing_data(weather_data, adjacency_matrix, time_len, train_rate, seq_len, pre_len, model_name, scheme):
    train_size = int(time_len * train_rate)
    train_data = weather_data[0:train_size]
    test_data = weather_data[train_size:time_len]            
            
    if model_name == 'tgcn':
        trainX, trainY, testX, testY = [], [], [], []
        for i in range(len(train_data) - seq_len - pre_len):
            a1 = train_data[i: i + seq_len + pre_len]
            trainX.append(a1[0:seq_len])
            trainY.append(a1[seq_len: seq_len + pre_len])
        for i in range(len(test_data) - seq_len - pre_len):
            b1 = test_data[i: i + seq_len + pre_len]
            testX.append(b1[0:seq_len])
            testY.append(b1[seq_len: seq_len + pre_len])         
            
    else:
        sz_poi = pd.read_csv('sz_poi.csv', header=None)
        sz_poi = np.transpose(sz_poi)
        sz_poi_max = np.max(np.max(sz_poi))
        sz_poi_nor = sz_poi / sz_poi_max
        sz_weather = pd.read_csv('sz_weather.csv', header=None)
        sz_weather = np.mat(sz_weather)
        sz_weather_max = np.max(np.max(sz_weather))
        sz_weather_nor = sz_weather / sz_weather_max
        sz_weather_nor_train = sz_weather_nor[0:train_size]
        sz_weather_nor_test = sz_weather_nor[train_size:time_len]

        if scheme == 1:  # add poi(dim+1)
            trainX, trainY, testX, testY = [], [], [], []
            for i in range(len(train_data) - seq_len - pre_len):
                a1 = train_data[i: i + seq_len + pre_len]
                a = np.row_stack((a1[0:seq_len], sz_poi_nor[:1]))
                trainX.append(a)
                trainY.append(a1[seq_len: seq_len + pre_len])
            for i in range(len(test_data) - seq_len - pre_len):
                b1 = test_data[i: i + seq_len + pre_len]
                b = np.row_stack((b1[0:seq_len], sz_poi_nor[:1]))
                testX.append(b)
                testY.append(b1[seq_len: seq_len + pre_len])
        if scheme == 2:  # add weather(dim+11)
            trainX, trainY, testX, testY = [], [], [], []
            for i in range(len(train_data) - seq_len - pre_len):
                a1 = train_data[i: i + seq_len + pre_len]
                a2 = sz_weather_nor_train[i: i + seq_len + pre_len]
                a = np.row_stack((a1[0:seq_len], a2[0: seq_len + pre_len]))
                trainX.append(a)
                trainY.append(a1[seq_len: seq_len + pre_len])
            for i in range(len(test_data) - seq_len - pre_len):
                b1 = test_data[i: i + seq_len + pre_len]
                b2 = sz_weather_nor_test[i: i + seq_len + pre_len]
                b = np.row_stack((b1[0:seq_len], b2[0: seq_len + pre_len]))
                testX.append(b)
                testY.append(b1[seq_len: seq_len + pre_len])
        else:  # add kg(dim+12)
            trainX, trainY, testX, testY = [], [], [], []
            for i in range(len(train_data) - seq_len - pre_len):
                a1 = train_data[i: i + seq_len + pre_len]
                a2 = sz_weather_nor_train[i: i + seq_len + pre_len]
                a = np.row_stack((a1[0:seq_len], a2[0: seq_len + pre_len], sz_poi_nor[:1]))
                trainX.append(a)
                trainY.append(a1[seq_len: seq_len + pre_len])
            for i in range(len(test_data) - seq_len - pre_len):
                b1 = test_data[i: i + seq_len + pre_len]
                b2 = sz_weather_nor_test[i: i + seq_len + pre_len]
                b = np.row_stack((b1[0:seq_len], b2[0: seq_len + pre_len], sz_poi_nor[:1]))
                testX.append(b)
                testY.append(b1[seq_len: seq_len + pre_len])

    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    
    return trainX1, trainY1, testX1, testY1




# def load_assist_data(dataset):
#     # Get the current working directory and create the data directory path
#     data_dir = os.path.join(os.getcwd(), 'data')
#     # Read the adjacency CSV file into a Pandas DataFrame
#     sz_adj = pd.read_csv(os.path.join(data_dir, f'{dataset}_adj.csv'), header=None)
#     # Convert the DataFrame to a NumPy matrix
#     adj = np.mat(sz_adj)
#      # Read the speed CSV file into another Pandas DataFrame
#     data = pd.read_csv(os.path.join(data_dir, f'{dataset}_speed.csv'))
#      # Return the data DataFrame and adj matrix
#     return data, adj

# def data_preprocess(config):    
#     noise_name = config['noise_name']['default']
#     data_name = config['dataset']['default']
#     PG =  config['noise_param']['default']
    
#     ########## load data #########
#     if data_name == 'sz':
#         data, adj = load_assist_data('sz')
    
#     ## Applying different types of noise filter to data
#     if noise_name == 'Gauss':
#         #it generates Gaussian noise using the np.random.normal() function with mean 0 and standard deviation PG. 
#         # The generated noise is normalized using MaxMinNormalization() and added to the input data.
#         Gauss = np.random.normal(0,PG,size=data.shape)
#         noise_Gauss = MaxMinNormalization(Gauss,np.max(Gauss),np.min(Gauss))
#         data = data + noise_Gauss
#         return data 
#     elif noise_name == 'Possion':
#         # it generates Poisson noise using the np.random.poisson() function with mean PG. 
#         # The generated noise is also normalized and added to the input data.
#         Possion = np.random.poisson(PG,size=data.shape)
#         noise_Possion = MaxMinNormalization(Possion,np.max(Possion),np.min(Possion))
#         data = data + noise_Possion
#         return data 
#     else:
#         # else data is unchanged
#         return data 
     
# def processing_data(data1,  time_len, train_rate, seq_len, pre_len, model_name,scheme):
#     train_size = int(time_len * train_rate)
#     train_data = data1[0:train_size]
#     test_data = data1[train_size:time_len]            
            
#     if model_name == 'tgcn':################TGCN###########################
#         trainX, trainY, testX, testY = [], [], [], []
#         for i in range(len(train_data) - seq_len - pre_len):
#             a1 = train_data[i: i + seq_len + pre_len]
#             trainX.append(a1[0 : seq_len])
#             trainY.append(a1[seq_len : seq_len + pre_len])
#         for i in range(len(test_data) - seq_len -pre_len):
#             b1 = test_data[i: i + seq_len + pre_len]
#             testX.append(b1[0 : seq_len])
#             testY.append(b1[seq_len : seq_len + pre_len])         
            
#     else:################AST-GCN###########################
#         sz_poi = pd.read_csv('sz_poi.csv',header = None)
#         sz_poi = np.transpose(sz_poi)
#         sz_poi_max = np.max(np.max(sz_poi))
#         sz_poi_nor = sz_poi/sz_poi_max
#         sz_weather = pd.read_csv('sz_weather.csv',header = None)
#         sz_weather = np.mat(sz_weather)
#         sz_weather_max = np.max(np.max(sz_weather))
#         sz_weather_nor = sz_weather/sz_weather_max
#         sz_weather_nor_train = sz_weather_nor[0:train_size]
#         sz_weather_nor_test = sz_weather_nor[train_size:time_len]

#         if scheme == 1:#add poi(dim+1)
#             trainX, trainY, testX, testY = [], [], [], []
#             for i in range(len(train_data) - seq_len - pre_len):
#                 a1 = train_data[i: i + seq_len + pre_len]
#                 a = np.row_stack((a1[0:seq_len],sz_poi_nor[:1]))
#                 trainX.append(a)
#                 trainY.append(a1[seq_len : seq_len + pre_len])
#             for i in range(len(test_data) - seq_len -pre_len):
#                 b1 = test_data[i: i + seq_len + pre_len]
#                 b = np.row_stack((b1[0:seq_len],sz_poi_nor[:1]))
#                 testX.append(b)
#                 testY.append(b1[seq_len : seq_len + pre_len])
#         if scheme == 2:#add weather(dim+11)
#             trainX, trainY, testX, testY = [], [], [], []
#             for i in range(len(train_data) - seq_len - pre_len):
#                 a1 = train_data[i: i + seq_len + pre_len]
#                 a2 = sz_weather_nor_train[i: i + seq_len + pre_len]
#                 a = np.row_stack((a1[0:seq_len],a2[0: seq_len + pre_len]))
#                 trainX.append(a)
#                 trainY.append(a1[seq_len : seq_len + pre_len])
#             for i in range(len(test_data) - seq_len -pre_len):
#                 b1 = test_data[i: i + seq_len + pre_len]
#                 b2 = sz_weather_nor_test[i: i + seq_len + pre_len]
#                 b = np.row_stack((b1[0:seq_len],b2[0: seq_len + pre_len]))
#                 testX.append(b)
#                 testY.append(b1[seq_len : seq_len + pre_len])
#         else:#add kg(dim+12)
#             trainX, trainY, testX, testY = [], [], [], []
#             for i in range(len(train_data) - seq_len - pre_len):
#                 a1 = train_data[i: i + seq_len + pre_len]
#                 a2 = sz_weather_nor_train[i: i + seq_len + pre_len]
#                 a = np.row_stack((a1[0:seq_len],a2[0: seq_len + pre_len],sz_poi_nor[:1]))
#                 trainX.append(a)
#                 trainY.append(a1[seq_len : seq_len + pre_len])
#             for i in range(len(test_data) - seq_len -pre_len):
#                 b1 = test_data[i: i + seq_len + pre_len]
#                 b2 = sz_weather_nor_test[i: i + seq_len + pre_len]
#                 b = np.row_stack((b1[0:seq_len],b2[0: seq_len + pre_len],sz_poi_nor[:1]))
#                 testX.append(b)
#                 testY.append(b1[seq_len : seq_len + pre_len])

#     trainX1 = np.array(trainX)
#     trainY1 = np.array(trainY)
#     testX1 = np.array(testX)
#     testY1 = np.array(testY)
#     # print(trainX1.shape)
#     # print(trainY1.shape)
#     # print(testX1.shape)
#     # print(testY1.shape)
    
#     return trainX1, trainY1, testX1, testY1