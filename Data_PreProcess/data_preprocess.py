# data pre-process
from Evaluation.metrics import metrics, MaxMinNormalization
from Model.acell import preprocess_data,load_assist_data
import numpy as np

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
