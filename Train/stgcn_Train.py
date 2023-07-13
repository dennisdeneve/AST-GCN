import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Utils.utils import create_X_Y, min_max, dataSplit
from Model.stgcn import stgcnModel
from Data_PreProcess.data_preprocess import data_preprocess_ST_GCN

def trainSTGCN(config):
    increment = config['increment']['default']
    stations = config['stations']['default']
    forecasting_horizons = config['forecasting_horizons']['default']
    num_splits =config['num_splits']['default']
    time_steps =config['time_steps']['default']
    
    for forecast_len in forecasting_horizons:
        for station in stations:
            print('********** ST-GCN model training started at ' + station)
            
            n_ahead_length = forecast_len
             
            processed_data, adjacency_matrix, num_nodes = data_preprocess_ST_GCN(station)
            
            lossDF = pd.DataFrame()
            resultsDF = pd.DataFrame()
            targetDF = pd.DataFrame()
            targetFile = 'Results/TGCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Targets/' + \
                                    'target.csv'
            resultsFile = 'Results/TGCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Predictions/' + \
                                    'result.csv'
            lossFile = 'Results/TGCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Predictions/' + \
                                'loss.csv'
            
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

            for k in range(num_splits):
                print('STGCN training started on split {0}/{3} at {1} station forecasting {2} hours ahead.'.format(k+1, station,
                                                                                                            forecast_len, num_splits))
                
                save_File = 'Garage/Final Models/STGCN/' + station + '/' + str(forecast_len) + ' Hour Models/Best_Model_' \
                            + str(n_ahead_length) + '_walk_' + str(k) + '.h5'
                            
                # splitting the processed time series data
                split = [increment[k], increment[k + 1], increment[k + 2]]
                pre_standardize_train, pre_standardize_validation, pre_standardize_test = dataSplit(split, input_data)
                # Scaling the data
                train, validation, test = min_max(pre_standardize_train,
                                                        pre_standardize_validation,
                                                        pre_standardize_test) 
                    
                # Creating the X and Y for forecasting (training), validation & testing
                X_train, Y_train = create_X_Y(train, time_steps, num_nodes, n_ahead_length)
                X_val, Y_val = create_X_Y(validation, time_steps, num_nodes, n_ahead_length)
                X_test, Y_test = create_X_Y(test, time_steps, num_nodes, n_ahead_length)
                
                #### Get model from methods in stgcn.py in Model/
                model, history = stgcnModel(time_steps, num_nodes, adjacency_matrix,save_File, X_train, Y_train, X_val, Y_val)
               
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
                # Replicate the columns 10 times
                new_data = pd.concat([new_data]*10, axis=1)

                # Ensure columns are in correct order
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
                # station = weather_stations[station_index]
                # temperature_prediction = predictions[0][station_index * 6 + 5]  
                temperature_prediction = predictions[0][station_index * 6 + 5]
                print(f'Predicted temperature at {station}: {temperature_prediction}')
                
            print('ST-GCN training finished on split {0}/{3} at {1} station forecasting {2} hours ahead.'.format(k+1, station,
                                                                                                            forecast_len, num_splits))   
            # Save the results to the file
            resultsDF.to_csv(resultsFile)
            lossDF.to_csv(lossFile)
            targetDF.to_csv(targetFile)
    print("ST-GCN training finished for all stations at all splits ! :)")