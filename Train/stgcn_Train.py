import numpy as np
import pandas as pd
from Utils.utils import create_X_Y, min_max, dataSplit, create_file_if_not_exists
from Model.stgcn import stgcnModel
from Data_PreProcess.data_preprocess import data_preprocess_ST_GCN, sliding_window_ST_GCN

def trainSTGCN(config):
    increment = config['increment']['default']
    stations = config['stations']['default']
    forecasting_horizons = config['forecasting_horizons']['default']
    num_splits =config['num_splits']['default']
    time_steps =config['time_steps']['default']
    
    for forecast_len in forecasting_horizons:
        for station in stations:
            print('********** ST-GCN model training started at ' + station) 
            
            processed_data, adjacency_matrix, num_nodes = data_preprocess_ST_GCN(station)
            lossDF = pd.DataFrame()
            resultsDF = pd.DataFrame()
            targetDF = pd.DataFrame()
            folder_path = f'Results/STGCN/{forecast_len} Hour Forecast/{station}'
            targetFile = f'{folder_path}/Targets/target.csv'
            resultsFile = f'{folder_path}/Predictions/result.csv'
            lossFile = f'{folder_path}/Predictions/loss.csv'
            create_file_if_not_exists(targetFile)
            create_file_if_not_exists(resultsFile)
            create_file_if_not_exists(lossFile)

            input_data, target_data, scaler = sliding_window_ST_GCN(processed_data, time_steps, num_nodes)

            for k in range(num_splits):
                print('STGCN training started on split {0}/{3} at {1} station forecasting {2} hours ahead.'.format(k+1, station,
                                                                                                            forecast_len, num_splits))
                
                save_File = 'Garage/Final Models/STGCN/' + station + '/' + str(forecast_len) + ' Hour Models/Best_Model_' \
                            + str(forecast_len) + '_walk_' + str(k) + '.h5'
                create_file_if_not_exists(save_File)
                # splitting the processed time series data
                split = [increment[k], increment[k + 1], increment[k + 2]]
                pre_standardize_train, pre_standardize_validation, pre_standardize_test = dataSplit(split, input_data)
                # Scaling the data
                train, validation, test = min_max(pre_standardize_train,
                                                        pre_standardize_validation,
                                                        pre_standardize_test) 
                    
                # Creating the X and Y for forecasting (training), validation & testing
                X_train, Y_train = create_X_Y(train, time_steps, num_nodes, forecast_len)
                X_val, Y_val = create_X_Y(validation, time_steps, num_nodes, forecast_len)
                X_test, Y_test = create_X_Y(test, time_steps, num_nodes, forecast_len)
                
                #### Get model from methods in stgcn.py in Model/
                model, history = stgcnModel(time_steps, num_nodes, adjacency_matrix,save_File, X_train, Y_train, X_val, Y_val)
               
                # validation and train loss to dataframe
                lossDF = lossDF.append([[history.history['loss']]])
                
                # Use the trained model for predictions
                new_data = pd.DataFrame({
                    'Pressure': [997.5] * time_steps,
                    'WindDir': [100.0] * time_steps,
                    'WindSpeed': [2.0] * time_steps,
                    'Humidity': [70.0] * time_steps,
                    'Rain': [0.0] * time_steps,
                    'Temperature': [25.5] * time_steps
                })

                new_data = pd.concat([new_data]*10, axis=1)
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
                temperature_prediction = predictions[0][station_index * 6 + 5]
                print(f'Predicted temperature at {station}: {temperature_prediction}')
                
            print('ST-GCN training finished on split {0}/{3} at {1} station forecasting {2} hours ahead.'.format(k+1, station,
                                                                                                            forecast_len, num_splits))   
            # Save the results to the file
            resultsDF.to_csv(resultsFile)
            lossDF.to_csv(lossFile)
            targetDF.to_csv(targetFile)
    print("ST-GCN training finished for all stations at all splits ! :)")