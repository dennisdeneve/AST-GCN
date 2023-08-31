import io
import os
import numpy as np
import pandas as pd
from Model.ASTGCN.astgcn import AstGcn
import Utils.astgcnUtils as utils
import Utils.astgcn_Data_PreProcess.data_preprocess as data_preprocess
from Logs.modelLogger import modelLogger 
from contextlib import redirect_stdout
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class astgcnExecute:
    def __init__(self, sharedConfig, config):
        """Initializes an ASTGCNTrainer with a given configuration."""
        self.config = config
        self.station = None
        self.forecast_len = None
        self.increment = sharedConfig['increment']['default']
        self.stations = sharedConfig['stations']['default']
        self.forecasting_horizons = sharedConfig['horizons']['default']
        self.num_splits =sharedConfig['n_split']['default']
        self.time_steps =config['time_steps']['default']
        self.batch_size = config['batch_size']['default']
        self.epochs = config['training_epoch']['default']
        self.logger = None

    def train(self):
        """Trains the model for all forecast lengths and stations. Either set to single or multiple 
        time steps to forecast"""
        print("Executing experimentation for time series prediction for weather forecasting")
        print("Forecasting horizons currently set to " + str(self.forecasting_horizons));
        for self.forecast_len in self.forecasting_horizons:
            # for self.station in self.stations:
                # Your directory path
            log_dir = 'Logs/ASTGCN/Train/' + str(self.forecast_len) + ' Hour Forecast/' + str(self.station)
            # Check if directory exists
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.logger = modelLogger('ASTGCN', str(self.station), log_dir + str(self.station) + '.txt', log_enabled=True)
            self.train_single_station()
             
    def train_single_station(self):
        """Trains the model for a single station."""     
        self.logger.info(f'********** AST-GCN model training started')
        
        self.logger.info(f'Starting data preprocessing for all stations ')
        processed_data, attribute_data, adjacency_matrix, num_nodes = data_preprocess.data_preprocess_AST_GCN(self.station)
        self.initialize_results()   
        self.train_model(processed_data, attribute_data, adjacency_matrix, num_nodes)
            
    def split_data(self,input_data, increment,k):
        """Splits the input data into training, validation, and test sets."""
        splits = [increment[k], increment[k + 1], increment[k + 2]]
        pre_standardize_train, pre_standardize_validation, pre_standardize_test = utils.dataSplit(splits, input_data)
        return utils.min_max(pre_standardize_train, pre_standardize_validation, pre_standardize_test, splits)

    def initialize_results(self):
        """Initializes the results, loss, and target data lists."""
        self.lossData = []
        self.resultsData = []
        self.targetData = []

    def train_model(self, processed_data, attribute_data, adjacency_matrix, num_nodes):
        """Trains the model with the preprocessed data, attribute data, and adjacency matrix."""
        self.logger.debug('Starting to train the model')
        # folder_path = f'Results/ASTGCN/{self.forecast_len} Hour Forecast/{self.station}'
        folder_path = f'Results/ASTGCN/{self.forecast_len} Hour Forecast/All Stations'
        self.targetFile, self.resultsFile, self.lossFile, self.actual_vs_predicted_file = utils.generate_execute_file_paths(folder_path)
        input_data, target_data = data_preprocess.sliding_window_AST_GCN(processed_data, self.time_steps, num_nodes)
        for k in range(self.num_splits):
            self.train_single_split(k, input_data, attribute_data, adjacency_matrix, num_nodes)
        self.logger.info('Model training completed')

    

    def train_single_split(self, k, input_data, attribute_data, adjacency_matrix, num_nodes):
        """Trains the model for a single split of the data."""

        print('ASTGCN training started on split {0}/{2} at all stations forecasting {1} hours ahead.'.format(k+1, self.forecast_len, self.num_splits))
        save_File = f'Garage/Final Models/ASTGCN/All Stations/{str(self.forecast_len)}Hour Models/Best_Model_\
                    {str(self.forecast_len)}_walk_{str(k)}.h5'
        utils.create_file_if_not_exists(save_File) 
        train, validation, test, split = self.split_data(input_data, self.increment,k)
        X_train, Y_train = utils.create_X_Y(train, self.time_steps, num_nodes, self.forecast_len)
        # print("This is the train data: ", train)
        # print("This is the X train data: ", train)
        # print("This is the Y train data: ", train)
        
        X_val, Y_val = utils.create_X_Y(validation, self.time_steps, num_nodes, self.forecast_len)
        X_test, Y_test = utils.create_X_Y(test, self.time_steps, num_nodes, self.forecast_len)
        
        ## normalize the attribute data too
        attribute_data = utils.normalize_data(attribute_data)
        
        # Instantiate the AstGcn class
        astgcn = AstGcn(self.time_steps, num_nodes, adjacency_matrix, 
                                    attribute_data, save_File, self.forecast_len, 
                                    X_train, Y_train, X_val, Y_val, split, self.batch_size, self.epochs, 
                                    self.config['gru_units']['default'], self.config['lstm_neurons']['default'])
        # Train the model by calling the astgcnModel method
        model, history = astgcn.astgcnModel()

        # Log the model summary
        with io.StringIO() as buf, redirect_stdout(buf):
            model.summary()
            model_summary = buf.getvalue()
        self.logger.info(f'Model Summary:\n{model_summary}')
        
        self.lossData.append([history.history['loss']])
        # predictions = self.predict(model, num_nodes, scaler)
        yhat = model.predict(X_test)

        print(f"Actual: {Y_test.flatten()}, Predicted: {yhat.flatten()}")
        
        # Y_test = np.expand_dims(Y_test, axis=2)  
        self.resultsData.append(yhat.reshape(-1,))
        self.targetData.append(Y_test.reshape(-1,))
        self.save_data(Y_test, yhat)
        
    def predict(self, model, num_nodes, scaler):
        """Generates a prediction from the model."""
        new_data = self.create_new_data()
        new_data = pd.concat([new_data]*10, axis=1)
        new_data = new_data[sorted(new_data.columns)]
        new_data = new_data.astype(float)
        new_data = np.expand_dims(new_data, axis=0)  # Add batch dimension
        new_data = np.expand_dims(new_data, axis=2)  # Add node dimension
        new_data = new_data.reshape(-1, self.time_steps, 1, 40)
        predictions = model.predict(new_data)
        return scaler.inverse_transform(predictions.reshape(-1, num_nodes * 4))

    def create_new_data(self):
        """Creates a new DataFrame with default data for prediction."""
        return pd.DataFrame({
                    'Pressure': [997.5] * self.time_steps,
                    'WindDir': [100.0] * self.time_steps,
                    'WindSpeed': [2.0] * self.time_steps,
                    'Humidity': [70.0] * self.time_steps,
                    'Rain': [0.0] * self.time_steps,
                    'Temperature': [25.5] * self.time_steps
                })

    def save_data(self, Y_test, yhat):
        """Saves the results, loss, target data, and the actual vs predicted comparison to CSV files."""
        # Save Results, Loss, and Target
        self.logger.info(f'Saving the results of predictions to ' + str(self.resultsFile))
        # print(f'Saving the results of predictions to' + str(self.resultsFile) )
        resultsDF = pd.DataFrame(np.concatenate(self.resultsData))
        self.logger.info(f'Saving the targets of actual values to ' + str(self.targetFile) )
        # print(f'Saving the targets of actual values to ' + str(self.targetFile) )
        targetDF = pd.DataFrame(np.concatenate(self.targetData))
        self.logger.info(f'Saving the loss to ' + str(self.lossFile) )
        # print(f'Saving the loss to ' + str(self.lossFile) )
        lossDF = pd.DataFrame(self.lossData)
        
        resultsDF.to_csv(self.resultsFile)
        lossDF.to_csv(self.lossFile)
        targetDF.to_csv(self.targetFile)
        
        # Save Actual vs Predicted
        self.logger.info(f'Saving the actual vs predicted comparison to a CSV file.')
        actual_vs_predicted_data = pd.DataFrame({
            'Actual': Y_test.flatten(),
            'Predicted': yhat.flatten()
        })
        # Log all actual vs predicted values
        for index, row in actual_vs_predicted_data.iterrows():
            file_path = 'data/Weather Station Data/'+ str(self.station) +'.csv'
            date = data_preprocess.get_timestamp_at_index(index)
            self.logger.info(f'Date {date} Index {index} - Actual: {row["Actual"]}, Predicted: {row["Predicted"]}')
        # actual_vs_predicted_data.to_csv(self.actual_vs_predicted_file, index=False)  