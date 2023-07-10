import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math

def SMAPE(actual, predicted):
    """
    Calculates the SMAPE metric
    Parameters:
        actual - target values
        predicted - output values predicted by model
    Returns:
        smape - returns smape metric
    """

    return np.mean(abs(predicted - actual) / ((abs(predicted) + abs(actual)) / 2)) * 100

def MSE(target, pred):
    """
    Calculates the MSE metric
    Parameters:
        actual - target values
        predicted - output values predicted by model
    Returns:
        mse - returns MSE metric
    """

    return mean_squared_error(target, pred, squared=True)



def RMSE(target, pred):
    """
    Calculates the RMSE metric
    Parameters:
        actual - target values
        predicted - output values predicted by model
    Returns:
        root - returns RMSE metric
    """

    root = math.sqrt(mse(target, pred))
    return root


def MAE(target, pred):
    """
    Calculates the MAE metric
    Parameters:
        actual - target values
        predicted - output values predicted by model
    Returns:
        mae - returns MAE metric
    """
    return mean_absolute_error(target, pred)


def evalTGCN(model):
    """
    Calculates the LSTM/TCN model's performance on the test set for each station. These metrics are written to a file
    for each station. The predictions are read from the results file for each station. The targets are pulled from
    the weather stations' data sets. The MSE, MAE, RMSE, and SMAPE metrics are calculated on all forecasting
    horizons(3, 6, 9, 12, and 24) for each individual weather station. The metrics for each station, across all
    forecasting horizons are then written to a text file.

    Parameters:
        stations - List of the weather stations.
        model - Whether these metrics are being calculated for the LSTM or TCN model
    """

    stations = ['ADDO ELEPHANT PARK']
    horizons =  [3, 6, 9, 12, 24]
    
    for station in stations:
        for horizon in horizons:
            try:
                print('TGCN evaluation started at' , str(station)+' for the horizon of ' ,str(horizon) ) 
                
                 # Set the file paths for predictions, targets, and metrics
                yhat_path = f'Results/{model}/{horizon} Hour Forecast/{station}/Predictions/result.csv'
                target_path = f'Results/{model}/{horizon} Hour Forecast/{station}/Targets/target.csv'
                metric_file = f'Results/{model}/{horizon} Hour Forecast/{station}/Metrics/metrics.txt'
                
                # Read the predictions and targets from the CSV files
                preds = pd.read_csv(yhat_path).drop(['Unnamed: 0'], axis=1)
                targets = pd.read_csv(target_path).drop(['Unnamed: 0'], axis=1)

                # Calculate the metrics
                # mse=0
                # rmse=0
                # mae=0
                # smape=0
                mse = MSE(targets.values, preds.values)
                rmse = RMSE(targets.values, preds.values)
                mae = MAE(targets.values, preds.values)
                smape = SMAPE(targets.values, preds.values)

                # Write the metrics to the metric file
                with open(metric_file, 'w') as metric:
                    metric.write('This is the MSE: {}\n'.format(mse))
                    metric.write('This is the MAE: {}\n'.format(mae))
                    metric.write('This is the RMSE: {}\n'.format(rmse))
                    metric.write('This is the SMAPE: {}\n'.format(smape))
                
                print('TGCN evaluation done at' + str(station)+' for the horizon of ' +str(horizon) ) 
                print('TGCN evaluation of ' + str(station)+' for the horizon of ' +str(horizon) +' was saved to Results/{model}/{horizon} Hour Forecast/{station}/Metrics/metrics.txt' ) 
                
                # Print the metrics for the current station and horizon length
                print('SMAPE: {0} at the {1} station forecasting {2} hours ahead.'.format(smape, station, horizon))
                print('MSE: {0} at the {1} station forecasting {2} hours ahead.'.format(mse, station, horizon))
                print('MAE: {0} at the {1} station forecasting {2} hours ahead.'.format(mae, station, horizon))
                print('RMSE: {0} at the {1} station forecasting {2} hours ahead.'.format(rmse, station, horizon))
                print('')
            except IOError as e :
                print('Error! Unable to read data or write metrics:', str(e))
                #print('Error! : Unable to read data or write metrics for station {} and horizon length {}. Please review the data or code for the metrics for errors.'.format(station, horizon))
               
    print('Finished evaluation of TGCN error metrics for all stations.')  