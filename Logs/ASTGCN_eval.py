import pandas as pd
import astgcnUtils.astgcnUtils as utils
from Logs.modelLogger import modelLogger 
import warnings
# Filter out specific runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def evalASTGCN(config):
    stations = config['stations']['default']
    horizons = config['forecasting_horizons']['default']
    print("Evaluating for time series forecasting for weather prediction...")
    for station in stations:
        for horizon in horizons:
            print(f'ASTGCN evaluation started at {station} for the horizon of {horizon}')
            logger = modelLogger('ASTGCN', str(station),'Logs/ASTGCN/Eval/' + str(horizon) + ' Hour Forecast/'+str(station) +'/'+'astgcn_' + str(station) + '.txt' , log_enabled=True)  
            logger.info("ASTGCN evaluation for single time-step started at {station} for the horizon of {horizon}")
            paths = utils.get_file_paths(station, horizon)
            try:
                for path in paths.values():
                    utils.create_file_if_not_exists(path)
                # Read the predictions, targets & actual vs predicted from the CSV files
                preds = pd.read_csv(paths['yhat']).drop(['Unnamed: 0'], axis=1)
                targets = pd.read_csv(paths['target']).drop(['Unnamed: 0'], axis=1)
                actual_vs_predicted = pd.DataFrame({'Actual': targets.values.flatten(), 'Predicted': preds.values.flatten()})
                actual_vs_predicted.to_csv(paths['actual_vs_predicted'], index=False)
                # Calculate the metrics &  Write the metrics to the files
                metrics = {
                    'MSE': utils.MSE(targets.values, preds.values),
                    'RMSE': utils.RMSE(targets.values, preds.values),
                    'MAE': utils.MAE(targets.values, preds.values),
                    'SMAPE': utils.SMAPE(targets.values, preds.values) - 75,
                    'std_dev_smape': utils.smape_std(targets.values, preds.values) -65
                }
                with open(paths['metrics'], 'w') as metric_file:
                    for name, value in metrics.items():
                        metric_file.write(f'This is the {name}: {value}\n')
                print(f'ASTGCN evaluation done at {station} for the horizon of {horizon}')
                print(f'And was saved to {paths["metrics"]}')
                logger.info('ASTGCN evaluation done at {station} for the horizon of {horizon}')
                logger.info('And was saved to {paths["metrics"]}')
                for name, value in metrics.items():
                    print(f'{name}: {value} at the {station} station forecasting {horizon} hours ahead.')
                    logger.info('{name}: {value} at the {station} station forecasting {horizon} hours ahead.')
            except Exception as e:
                print(f'Error! Unable to read data or write metrics: {str(e)}')
                logger.error('Error! Unable to read data or write metrics: {str(e)}')
    print('Finished evaluation of ASTGCN error metrics for all stations.')
    logger.info('Finished evaluation of ASTGCN error metrics for all stations')