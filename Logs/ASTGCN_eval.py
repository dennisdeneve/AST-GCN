import pandas as pd
import astgcnUtils.astgcnUtils as utils

def evalASTGCN(config):
    stations = config['stations']['default']
    horizons = config['forecasting_horizons']['default']
    single_horizon = config['forecasting_horizon']['default']
    
    if config['single_time_step']['default']:
        print("Evaluating for single-step forecasting...")
        print("Horizon currently set to " + str(single_horizon));     
        for station in stations:
            for horizon in single_horizon:
                print(f'ASTGCN evaluation started at {station} for the horizon of {horizon}')
                paths = utils.get_file_paths(station, horizon)
                try:
                    for path in paths.values():
                        utils.create_file_if_not_exists(path)
                    # Read the predictions and targets from the CSV files
                    preds = pd.read_csv(paths['yhat']).drop(['Unnamed: 0'], axis=1)
                    targets = pd.read_csv(paths['target']).drop(['Unnamed: 0'], axis=1)
                    # Create a DataFrame of actual vs predicted values & Save it
                    actual_vs_predicted = pd.DataFrame({'Actual': targets.values.flatten(), 'Predicted': preds.values.flatten()})
                    actual_vs_predicted.to_csv(paths['actual_vs_predicted'], index=False)
                    # Calculate the metrics &  Write the metrics to the files
                    metrics = {
                        'MSE': utils.MSE(targets.values, preds.values),
                        'RMSE': utils.RMSE(targets.values, preds.values),
                        'MAE': utils.MAE(targets.values, preds.values),
                        'SMAPE': utils.SMAPE(targets.values, preds.values)
                    }
                    with open(paths['metrics'], 'w') as metric_file:
                        for name, value in metrics.items():
                            metric_file.write(f'This is the {name}: {value}\n')

                    print(f'ASTGCN evaluation done at {station} for the horizon of {horizon}')
                    print(f'And was saved to {paths["metrics"]}')
                    for name, value in metrics.items():
                        print(f'{name}: {value} at the {station} station forecasting {horizon} hours ahead.')
                except Exception as e:
                    print(f'Error! Unable to read data or write metrics: {str(e)}')
        print('Finished evaluation of TGCN error metrics for all stations.')
    
    if config['multiple_time_steps']['default']:
        print("Evaluating for multi-step forecasting...")
        print("Horizons currently set to " + str(horizons));
        for station in stations:
            for horizon in horizons:
                print(f'ASTGCN evaluation started at {station} for the horizon of {horizon}')
                paths = utils.utils.get_file_paths(station, horizon)
                try:
                    for path in paths.values():
                        utils.create_file_if_not_exists(path)
                    # Read the predictions and targets from the CSV files
                    preds = pd.read_csv(paths['yhat']).drop(['Unnamed: 0'], axis=1)
                    targets = pd.read_csv(paths['target']).drop(['Unnamed: 0'], axis=1)
                    # Create a DataFrame of actual vs predicted values & Save it
                    actual_vs_predicted = pd.DataFrame({'Actual': targets.values.flatten(), 'Predicted': preds.values.flatten()})
                    actual_vs_predicted.to_csv(paths['actual_vs_predicted'], index=False)
                    # Calculate the metrics &  Write the metrics to the files
                    metrics = {
                        'MSE': utils.MSE(targets.values, preds.values),
                        'RMSE': utils.RMSE(targets.values, preds.values),
                        'MAE': utils.MAE(targets.values, preds.values),
                        'SMAPE': utils.SMAPE(targets.values, preds.values)
                    }
                    with open(paths['metrics'], 'w') as metric_file:
                        for name, value in metrics.items():
                            metric_file.write(f'This is the {name}: {value}\n')

                    print(f'ASTGCN evaluation done at {station} for the horizon of {horizon}')
                    print(f'And was saved to {paths["metrics"]}')
                    for name, value in metrics.items():
                        print(f'{name}: {value} at the {station} station forecasting {horizon} hours ahead.')
                except Exception as e:
                    print(f'Error! Unable to read data or write metrics: {str(e)}')
        print('Finished evaluation of ASTGCN error metrics for all stations.')
    
    else:
        print("Please set a configuration setting to true for either single time step or multiple time steps forecasting for the AST-GCN model")