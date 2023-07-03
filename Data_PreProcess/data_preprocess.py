# data pre-process
from Evaluation.metrics import metrics, MaxMinNormalization
# from Model.acell import load_assist_data
import numpy as np
import pandas as pd
import os



def load_assist_data(dataset):
    # Get the current working directory and create the data directory path
    data_dir = os.path.join(os.getcwd(), 'data')
    # Read the adjacency CSV file into a Pandas DataFrame
    sz_adj = pd.read_csv(os.path.join(data_dir, f'{dataset}_adj.csv'), header=None)
    # Convert the DataFrame to a NumPy matrix
    adj = np.mat(sz_adj)
     # Read the speed CSV file into another Pandas DataFrame
    data = pd.read_csv(os.path.join(data_dir, f'{dataset}_speed.csv'))
     # Return the data DataFrame and adj matrix
    return data, adj


def data_preprocess(config):    
    noise_name = config['noise_name']['default']
    data_name = config['dataset']['default']
    PG =  config['noise_param']['default']
    
    ########## load data #########
    if data_name == 'sz':
        data, adj = load_assist_data('sz')
    #if data_name == 'sh':
    #    data, adj = load_sh_data('sh')

    ## Applying different types of noise filter to data
    if noise_name == 'Gauss':
        #If noise_name is 'Gauss', 
        # it generates Gaussian noise using the np.random.normal() function with mean 0 and standard deviation PG. 
        # The generated noise is normalized using MaxMinNormalization() and added to the input data.
        Gauss = np.random.normal(0,PG,size=data.shape)
        noise_Gauss = MaxMinNormalization(Gauss,np.max(Gauss),np.min(Gauss))
        data = data + noise_Gauss
        return data 
    elif noise_name == 'Possion':
        #If noise_name is 'Possion',
        # it generates Poisson noise using the np.random.poisson() function with mean PG. 
        # The generated noise is also normalized and added to the input data.
        Possion = np.random.poisson(PG,size=data.shape)
        noise_Possion = MaxMinNormalization(Possion,np.max(Possion),np.min(Possion))
        data = data + noise_Possion
        return data 
    else:
        # Else data is unchanged
        return data 
     

# def preprocess_data(data1,  time_len, train_rate, seq_len, pre_len, model_name,scheme):
def processing_data(data1,  time_len, train_rate, seq_len, pre_len, model_name,scheme):
    train_size = int(time_len * train_rate)
    train_data = data1[0:train_size]
    test_data = data1[train_size:time_len]            
            
    if model_name == 'tgcn':################TGCN###########################
        trainX, trainY, testX, testY = [], [], [], []
        for i in range(len(train_data) - seq_len - pre_len):
            a1 = train_data[i: i + seq_len + pre_len]
            trainX.append(a1[0 : seq_len])
            trainY.append(a1[seq_len : seq_len + pre_len])
        for i in range(len(test_data) - seq_len -pre_len):
            b1 = test_data[i: i + seq_len + pre_len]
            testX.append(b1[0 : seq_len])
            testY.append(b1[seq_len : seq_len + pre_len])         
            
    else:################AST-GCN###########################
        sz_poi = pd.read_csv('sz_poi.csv',header = None)
        sz_poi = np.transpose(sz_poi)
        sz_poi_max = np.max(np.max(sz_poi))
        sz_poi_nor = sz_poi/sz_poi_max
        sz_weather = pd.read_csv('sz_weather.csv',header = None)
        sz_weather = np.mat(sz_weather)
        sz_weather_max = np.max(np.max(sz_weather))
        sz_weather_nor = sz_weather/sz_weather_max
        sz_weather_nor_train = sz_weather_nor[0:train_size]
        sz_weather_nor_test = sz_weather_nor[train_size:time_len]

        if scheme == 1:#add poi(dim+1)
            trainX, trainY, testX, testY = [], [], [], []
            for i in range(len(train_data) - seq_len - pre_len):
                a1 = train_data[i: i + seq_len + pre_len]
                a = np.row_stack((a1[0:seq_len],sz_poi_nor[:1]))
                trainX.append(a)
                trainY.append(a1[seq_len : seq_len + pre_len])
            for i in range(len(test_data) - seq_len -pre_len):
                b1 = test_data[i: i + seq_len + pre_len]
                b = np.row_stack((b1[0:seq_len],sz_poi_nor[:1]))
                testX.append(b)
                testY.append(b1[seq_len : seq_len + pre_len])
        if scheme == 2:#add weather(dim+11)
            trainX, trainY, testX, testY = [], [], [], []
            for i in range(len(train_data) - seq_len - pre_len):
                a1 = train_data[i: i + seq_len + pre_len]
                a2 = sz_weather_nor_train[i: i + seq_len + pre_len]
                a = np.row_stack((a1[0:seq_len],a2[0: seq_len + pre_len]))
                trainX.append(a)
                trainY.append(a1[seq_len : seq_len + pre_len])
            for i in range(len(test_data) - seq_len -pre_len):
                b1 = test_data[i: i + seq_len + pre_len]
                b2 = sz_weather_nor_test[i: i + seq_len + pre_len]
                b = np.row_stack((b1[0:seq_len],b2[0: seq_len + pre_len]))
                testX.append(b)
                testY.append(b1[seq_len : seq_len + pre_len])
        else:#add kg(dim+12)
            trainX, trainY, testX, testY = [], [], [], []
            for i in range(len(train_data) - seq_len - pre_len):
                a1 = train_data[i: i + seq_len + pre_len]
                a2 = sz_weather_nor_train[i: i + seq_len + pre_len]
                a = np.row_stack((a1[0:seq_len],a2[0: seq_len + pre_len],sz_poi_nor[:1]))
                trainX.append(a)
                trainY.append(a1[seq_len : seq_len + pre_len])
            for i in range(len(test_data) - seq_len -pre_len):
                b1 = test_data[i: i + seq_len + pre_len]
                b2 = sz_weather_nor_test[i: i + seq_len + pre_len]
                b = np.row_stack((b1[0:seq_len],b2[0: seq_len + pre_len],sz_poi_nor[:1]))
                testX.append(b)
                testY.append(b1[seq_len : seq_len + pre_len])


    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)

    testY1 = np.array(testY)
    print(trainX1.shape)
    print(trainY1.shape)
    print(testX1.shape)
    print(testY1.shape)
    
    return trainX1, trainY1, testX1, testY1
