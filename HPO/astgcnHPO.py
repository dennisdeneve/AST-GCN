import numpy as np
import pandas as pd
import os
from Model.ASTGCN.astgcn import AstGcn
import astgcnUtils.astgcnUtils as utils
from Data_PreProcess.data_preprocess import data_preprocess_AST_GCN, data_preprocess_HPO_AST_GCN, sliding_window_AST_GCN
from Logs.modelLogger import modelLogger 

class astgcnHPO:
    def __init__(self, config):
        self.config = config

    def hpo(self):
        increment = self.config['increment']['default']
        # stations = self.config['stations']['default']
        num_splits = self.config['num_splits']['default']
        time_steps = self.config['time_steps']['default']
        batch_size = self.config['batch_size']['default']
        epochs = self.config['training_epoch']['default']
        horizon  = 24
        
        param_path = 'HPO/Best Parameters/ASTGCN/'
        if not os.path.exists(param_path):
            os.makedirs(param_path)
        f = open(param_path + "configurations.txt", 'w')
        log_path = 'Logs/ASTGCN/HPO/'
        os.makedirs(log_path, exist_ok=True)
        log_file = log_path + 'astgcn_all_stations.txt'
        self.model_logger = modelLogger('astgcn', 'all_stations', log_file, log_enabled=True)
       
        print('********** AST-GCN model HPO started at all stations') 
            
        # processed_data, attribute_data, adjacency_matrix, num_nodes = data_preprocess_HPO_AST_GCN()
        processed_data, attribute_data, adjacency_matrix, num_nodes = data_preprocess_AST_GCN("ADDO ELEPHANT PARK")
        
        lossData, resultsData, targetData = [], [], []
            
        folder_path = f'Results/ASTGCN/{horizon} Hour Forecast/all_stations'
        targetFile, resultsFile, lossFile, actual_vs_predicted_file = utils.generate_execute_file_paths(folder_path)
        input_data, target_data, scaler = sliding_window_AST_GCN(processed_data, time_steps, num_nodes)

        textFile = 'HPO/Best Parameters/AST-GCN/configurations.txt'
        print("File with best configs is in ", textFile)
        f = open(textFile, 'w')
        best_mse = np.inf

        num_splits = 1
        for i in range(self.config['num_configs']['default']):
            config = utils.generateRandomParameters(self.config)
            print(f"Trying configuration {i+1}/{self.config['num_configs']['default']}: {config}")
            self.model_logger.info("Generating random parameters for ASTGCN")
            self.model_logger.info(f"Trying configuration {i+1}/{self.config['num_configs']['default']}: {config}")
                
            valid_config = True
            targets = []
            preds = []
            
            for k in range(num_splits):
                print('ASTGCN HPO training started on split {0}/{2} at all stations forecasting {1} hours ahead.'.format(k + 1, horizon, num_splits))
                save_File = f'Garage/Final Models/ASTGCN/{str(horizon)}Hour Models/all_stations/Best_Model_' + \
                                f'{str(horizon)}_walk_{str(k)}.h5'
                utils.create_file_if_not_exists(save_File)
                splits = [increment[k], increment[k + 1], increment[k + 2]]
                pre_standardize_train, pre_standardize_validation, pre_standardize_test = utils.dataSplit(splits, input_data)
                train, validation, test, split = utils.min_max(pre_standardize_train, pre_standardize_validation,
                                                                pre_standardize_test, splits)
                X_train, Y_train = utils.create_X_Y(train, time_steps, num_nodes, horizon)
                X_val, Y_val = utils.create_X_Y(validation, time_steps, num_nodes, horizon)
                X_test, Y_test = utils.create_X_Y(test, time_steps, num_nodes, horizon)
                try:
                    print('This is the HPO configuration: \n',
                        'Batch Size - ', self.config['batch_size']['default'], '\n',
                        'Epochs - ', self.config['training_epoch']['default'], '\n',
                        'Hidden GRU units - ', self.config['gru_units']['default'], '\n'
                        )   
                    # Instantiation and training
                    astgcn = AstGcn(time_steps, num_nodes, adjacency_matrix,
                                                attribute_data, save_File, horizon,
                                                X_train, Y_train, X_val, Y_val, split, 
                                                self.config['batch_size']['default'], self.config['training_epoch']['default'], 
                                                self.config['gru_units']['default'])
                    model, history = astgcn.astgcnModel()
                    lossData.append([history.history['loss']])
                    yhat = model.predict(X_test)
                    Y_test = np.expand_dims(Y_test, axis=2)
                    resultsData.append(yhat.reshape(-1,))
                    targetData.append(Y_test.reshape(-1,))
                except Warning:
                    valid_config = False
                    print(f"Error encountered during training with configuration {config}. Error message: {e}")
                    break
                targets.append(np.array(targetData).flatten())
                preds.append(np.array(resultsData).flatten())
            if valid_config:
                mse = utils.MSE(np.concatenate(np.array(targets, dtype=object)),
                                np.concatenate(np.array(preds, dtype=object)))
                if mse < best_mse:
                    print(f"Current MSE {mse:.2f} is better than previous best MSE {best_mse:.2f}.")
                    best_cfg = config
                    best_mse = mse
                else:
                    print(f"Current MSE {mse:.2f} is NOT better than previous best MSE {best_mse:.2f}.")

        f.write('This is the best configuration ' + str(best_cfg) + ' with an MSE of ' + str(best_mse))
        print('This is the best configuration ' + str(best_cfg) + ' with an MSE of ' + str(best_mse))
        f.close()
        self.model_logger.info('This is the best configuration ' + str(best_cfg) + ' with an MSE of ' + str(best_mse))
        self.model_logger.info("HPO finished successfully")
        print(f'HPO finished at all stations at {horizon} hour horizon')