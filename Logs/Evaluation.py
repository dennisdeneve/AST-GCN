import numpy as np
import os
import pandas as pd
import Utils.metrics as metrics
# import Utils.gwnUtils as utils
import Utils.metrics as metrics
import Utils.sharedUtils as sharedUtils
from Logs.modelLogger import modelLogger
# import Utils.agcrnUtils as agcrnUtil
import Utils.astgcnUtils as astgcnUtils
     
        
def AgcrnEval(modelConfig,sharedConfig):
        stations = sharedConfig['stations']['default'] 
        for horizon in sharedConfig['horizons']['default']:
            for k in range(sharedConfig['n_split']['default']):
                fileDictionary = {
                    'predFile': './Results/AGCRN/' + str(horizon) + ' Hour Forecast/Predictions/outputs_' + str(k),
                    'targetFile': 'Results/AGCRN/' + str(horizon) + ' Hour Forecast/Targets/targets_' + str(k),
                    'trainLossFile': 'Results/AGCRN/' + str(horizon) + ' Hour Forecast/Matrices/adjacency_matrix_' + str(k) + '.csv',
                    'validationLossFile': 'Results/AGCRN/' + str(horizon) + ' Hour Forecast/Matrices/adjacency_matrix_' + str(k) + '.csv',
                    'modelFile': 'Garage/Final Models/AGCRN/' + str(horizon) + ' Hour Models/model_split_' + str(k) + ".pth",
                    'matrixFile': 'Results/AGCRN/' + str(horizon) + ' Hour Forecast/Matrices/adjacency_matrix_' + str(k) + '.csv',
                    'metricFile0': './Results/AGCRN/'+  str(horizon)+ ' Hour Forecast/Metrics/',
                    
                    'metricFile1': '/split_' + str(k) + '_metrics.txt'
                }
            
                y_pred=np.load(fileDictionary["predFile"] + ".npy")
                y_true=np.load(fileDictionary["targetFile"] + ".npy")
                
                for i in range(45):
                    station_pred = y_pred[:, :, i, 0]
                    station_true = y_true[:, :, i, 0]
                    print("Evaluating horizon:"+ str(horizon) + " split:" + str(k) + " for station:" + stations[i])
                    # print(station_pred)

                    # mae, rmse, mape, _, _ = agcrnUtil.All_Metrics(station_pred, station_true, modelConfig['mae_thresh']['default'], modelConfig['mape_thresh']['default'])

                    rmse =  metrics.rmse(station_true, station_pred)
                    mse = metrics.mse(station_true, station_pred)
                    mae = metrics.mae(station_true, station_pred)
                    smape = metrics.smape(station_true, station_pred)



                    filePath =fileDictionary['metricFile0'] +str(stations[i])
                    if not os.path.exists(filePath):
                        os.makedirs(filePath)

                    with open(filePath + fileDictionary['metricFile1'], 'w') as file:
                
                        # file.write('This is the MAE ' + str(mae) + '\n')
                        # file.write('This is the RMSE ' + str(rmse) + '\n')
                        # file.write('This is the MAPE ' + str(mape) + '\n')
                        file.write('This is the RMSE ' + str(rmse) + '\n')
                        file.write('This is the MSE ' + str(mse) + '\n')
                        file.write('This is the MAE ' + str(mae) + '\n')
                        file.write('This is the SMAPE ' + str(smape) + '\n')

def evalASTGCN(config, sharedConfig):
    stations = sharedConfig['stations']['default'] 
    best_metrics = {}  # To store the best metrics for each station
    for horizon in sharedConfig['horizons']['default']:
        for k in range(sharedConfig['n_split']['default']):
            # logger = modelLogger('ASTGCN', 'All stations','Logs/ASTGCN/Eval/' + str(horizon) + ' Hour Forecast/'+ str(station) + '.txt' , log_enabled=False)
            # logger.info("ASTGCN evaluation for single time-step started at all stations for the horizon of {horizon}")
            fileDictionary = {
                "yhat": f'Results/ASTGCN/{horizon} Hour Forecast/All Stations/Predictions/result.csv',
                "target": f'Results/ASTGCN/{horizon} Hour Forecast/All Stations/Targets/target.csv',
                "metrics": f'Results/ASTGCN/{horizon} Hour Forecast/All Stations/Metrics/',
                'metricFile1': '/split_' + str(k) + '_metrics.txt', 
                "actual_vs_predicted": f'Results/ASTGCN/{horizon} Hour Forecast/All Stations/Metrics/actual_vs_predicted.txt'
                }
            y_pred = pd.read_csv(fileDictionary["yhat"]).values
            y_true = pd.read_csv(fileDictionary["target"]).values
            
            actual_vs_predicted = pd.DataFrame({'Actual': y_true.flatten(), 'Predicted': y_pred.flatten()})
            actual_vs_predicted.to_csv(fileDictionary['actual_vs_predicted'], index=False)
        
            for i in range(sharedConfig['n_stations']['default']):
                station_pred = y_pred[i,1]
                station_true = y_true[i,1]
                
                print("Evaluating horizon : "+ str(horizon) + "|  split : " + str(k) + "|  for station : " + stations[i])
                print("Station_pred : ",station_pred)
                print("station_true : ",station_true)
                
                mse = metrics.mse(np.array([station_true]), np.array([station_pred]))
                rmse = metrics.rmse(np.array([station_true]), np.array([station_pred]))
                mae = metrics.mae(np.array([station_true]), np.array([station_pred]))
                # smape = metrics.smape(np.array([station_true]), np.array([station_pred]))
                smape = metrics.ZeroAdjustedSMAPE(np.array([station_true]), np.array([station_pred]))
  
                filePath =fileDictionary['metrics'] +str(stations[i])
                if not os.path.exists(filePath):
                    os.makedirs(filePath)

                print("This is the path : "+ filePath )
                with open(filePath + fileDictionary['metricFile1'], 'w') as file:
                    file.write('This is the RMSE ' + str(rmse) + '\n')
                    file.write('This is the MSE ' + str(mse) + '\n')
                    file.write('This is the MAE ' + str(mae) + '\n')
                    file.write('This is the SMAPE ' + str(smape) + '\n')  
                    # print('This is the SMAPE ' + str(smape) + '\n')   
                
                # Saving best metrics
                if stations[i] not in best_metrics:
                    best_metrics[stations[i]] = {'mse': mse, 'rmse': rmse, 'mae': mae, 'smape': smape}
                else:
                    if mse < best_metrics[stations[i]]['mse']:
                        best_metrics[stations[i]]['mse'] = mse
                    if rmse < best_metrics[stations[i]]['rmse']:
                        best_metrics[stations[i]]['rmse'] = rmse
                    if mae < best_metrics[stations[i]]['mae']:
                        best_metrics[stations[i]]['mae'] = mae
                    if smape < best_metrics[stations[i]]['smape']:
                        best_metrics[stations[i]]['smape'] = smape
        
        # Once all splits are done, write the best metrics to a new file
        for station in stations:
            best_metric_file_path = f'Results/ASTGCN/{horizon} Hour Forecast/All Stations/Metrics/best_metrics_{station}.txt'
            with open(best_metric_file_path, 'w') as file:
                file.write('Best RMSE: ' + str(best_metrics[station]['rmse']) + '\n')
                file.write('Best MSE: ' + str(best_metrics[station]['mse']) + '\n')
                file.write('Best MAE: ' + str(best_metrics[station]['mae']) + '\n')
                file.write('Best SMAPE: ' + str(best_metrics[station]['smape']) + '\n')