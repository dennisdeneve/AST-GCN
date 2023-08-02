import numpy as np
import pandas as pd
from Model.astgcn import AstGcn
import astgcnUtils.astgcnUtils as utils
from Data_PreProcess.data_preprocess import data_preprocess_AST_GCN, sliding_window_AST_GCN

class astgcnHPO:
    def __init__(self, config):
        self.config = config

    def hpo(self):
        increment = self.config['increment']['default']
        stations = self.config['stations']['default']

        num_splits = self.config['num_splits']['default']
        time_steps = self.config['time_steps']['default']
        batch_size = self.config['batch_size']['default']
        epochs = self.config['training_epoch']['default']
        forecast_len  =24
        
        for station in stations:
            print('********** AST-GCN model HPO started at ' + station)     
            processed_data, attribute_data, adjacency_matrix, num_nodes = data_preprocess_AST_GCN(station)
            lossData, resultsData, targetData = [], [], []

            folder_path = f'Results/ASTGCN/{forecast_len} Hour Forecast/{station}'
            targetFile, resultsFile, lossFile, actual_vs_predicted_file = utils.generate_execute_file_paths(folder_path)
            input_data, target_data, scaler = sliding_window_AST_GCN(processed_data, time_steps, num_nodes)

               
            textFile = 'HPO/Best Parameters/AST-GCN/configurations.txt'
            f = open(textFile, 'w')
            best_mse = np.inf

            num_splits = 1
            for i in range(self.config['num_configs']['default']):
                config = utils.generateRandomParameters(self.config)
                valid_config = True
                targets = []
                preds = []
                
                for k in range(num_splits):
                    print('ASTGCN training started on split {0}/{3} at {1} station forecasting {2} hours ahead.'.format(k + 1, station, forecast_len, num_splits))
                    save_File = f'Garage/Final Models/ASTGCN/{station}/{str(forecast_len)}Hour Models/Best_Model_' + \
                                f'{str(forecast_len)}_walk_{str(k)}.h5'
                    utils.create_file_if_not_exists(save_File)
                    splits = [increment[k], increment[k + 1], increment[k + 2]]
                    pre_standardize_train, pre_standardize_validation, pre_standardize_test = utils.dataSplit(splits, input_data)
                    train, validation, test, split = utils.min_max(pre_standardize_train, pre_standardize_validation,
                                                                    pre_standardize_test, splits)
                    X_train, Y_train = utils.create_X_Y(train, time_steps, num_nodes, forecast_len)
                    X_val, Y_val = utils.create_X_Y(validation, time_steps, num_nodes, forecast_len)
                    X_test, Y_test = utils.create_X_Y(test, time_steps, num_nodes, forecast_len)
                        
                    try:
                        print('This is the HPO configuration: \n',
                            'Batch Size - ', self.config['batch_size']['default'], '\n',
                             'Epochs - ', self.config['training_epoch']['default'])
                        
                        # Instantiation and training
                        astgcn = AstGcn(time_steps, num_nodes, adjacency_matrix,
                                            attribute_data, save_File, forecast_len,
                                            X_train, Y_train, X_val, Y_val, split, 
                                            self.config['batch_size']['default'], self.config['training_epoch']['default'])
                        model, history = astgcn.astgcnModel()
                        lossData.append([history.history['loss']])
                        # Prediction
                        new_data = pd.DataFrame({
                                'Pressure': [997.5] * time_steps,
                                'WindDir': [100.0] * time_steps,
                                'WindSpeed': [2.0] * time_steps,
                                'Humidity': [70.0] * time_steps,
                                'Rain': [0.0] * time_steps,
                                'Temperature': [25.5] * time_steps})
                        new_data = pd.concat([new_data] * 10, axis=1)
                        new_data = new_data[sorted(new_data.columns)]
                        new_data = new_data.astype(float)
                        new_data = np.expand_dims(new_data, axis=0)  # Add batch dimension
                        new_data = np.expand_dims(new_data, axis=2)  # Add node dimension
                        new_data = new_data.reshape(-1, time_steps, 1, 40)
                        predictions = model.predict(new_data)
                        predictions = scaler.inverse_transform(predictions.reshape(-1, num_nodes * 4))
                        yhat = model.predict(X_test)
                        Y_test = np.expand_dims(Y_test, axis=2)
                        resultsData.append(yhat.reshape(-1,))
                        targetData.append(Y_test.reshape(-1,))
                        # Save results
                        actual_vs_predicted_data = pd.DataFrame({
                            'Actual': Y_test.flatten(),
                            'Predicted': yhat.flatten()})
                            # actual_vs_predicted_data.to_csv(actual_vs_predicted_file, index=False)
                            # resultsDF = pd.DataFrame(np.concatenate(resultsData))
                            # lossDF = pd.DataFrame(lossData)
                            # targetDF = pd.DataFrame(np.concatenate(targetData))

                    except Warning:
                        valid_config = False
                        break

                    targets.append(np.array(targetData).flatten())
                    preds.append(np.array(resultsData).flatten())

                if valid_config:
                    mse = utils.MSE(np.concatenate(np.array(targets, dtype=object)),
                                    np.concatenate(np.array(preds, dtype=object)))
                    if mse < best_mse:
                        print("current mse {mse} is better than previous mse {best_mse}")
                        best_cfg = config
                        best_mse = mse

            f.write('This is the best configuration ' + str(best_cfg) + ' with an MSE of ' + str(best_mse))
            f.close()
                # self.model_logger.info('gwnHPO : GWN best configuration found = ' +str(best_cfg) + ' with an MSE of ' + str(best_mse))
                # self.model_logger.info('gwnHPO : GWN HPO finished at all stations :)')
                
            print('HPO finished at {station} at 24 hour horizon')