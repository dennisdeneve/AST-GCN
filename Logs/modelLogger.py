import logging
""" 
logger class that uses Python's logging module as a flexible logging framework.
It allows you to log messages at different levels, define custom log handlers, and configure log formatting. 
"""
class modelLogger:
    def __init__(self, model_name, station, log_file,log_enabled=True):
        """
        Initializes the ModelLogger class.

        Args:
            model_name (str): Name of the model.
            station (str): Name of the station.
            log_file (str): Path to the log file.
        """
        # Could create a logger once, name becomes both model_name & station  
        # self.logger = logging.getLogger(f"{model_name}-{station}")
        # self.logger.setLevel(logging.DEBUG)
        
        self.logger = logging.getLogger(model_name)
        self.logger.setLevel(logging.DEBUG)
        
        self.logger = logging.getLogger(station)
        self.logger.setLevel(logging.DEBUG)
        
        # Allows us to switch the logging system on and off
        if log_enabled:
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
    
    def info(self, message):
        """
        Logs an info-level message.

        Args:
            message (str): Log message.
        """
        self.logger.info(message)
    
    def debug(self, message):
        """
        Logs a debug-level message.

        Args:
            message (str): Log message.
        """
        self.logger.debug(message)
    
    def warning(self, message):
        """
        Logs a warning-level message.

        Args:
            message (str): Log message.
        """
        self.logger.warning(message)
    
    def error(self, message):
        """
        Logs an error-level message.

        Args:
            message (str): Log message.
        """
        self.logger.error(message)
    
    def critical(self, message):
        """
        Logs a critical-level message.

        Args:
            message (str): Log message.
        """
        self.logger.critical(message)
    def close(self):
        """
        Closes the log file handlers.
        """
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)