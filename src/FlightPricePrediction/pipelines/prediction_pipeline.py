import pandas as pd
import numpy as np
import os
import sys
from FlightPricePrediction.logger import logging
from FlightPricePrediction.exception import customexception
from FlightPricePrediction.utils.utils import load_object  # Make sure this is implemented
from FlightPricePrediction.components.data_transformation import DataTransformation  # Assuming you have this class

class PredictionPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        
        # Load the trained model
        self.model = load_object(self.model_path)
        
        # Load the preprocessing object
        self.preprocessor = load_object(self.preprocessor_path)

    def preprocess_data(self, input_data):
        """
        Preprocess the input data using the loaded preprocessor.
        """
        try:
            # Assume input_data is a DataFrame
            logging.info("Preprocessing the input data.")
            processed_data = self.preprocessor.transform(input_data)
            return processed_data
        except Exception as e:
            logging.error("Error in preprocessing data")
            raise customexception(e, sys)

    def make_prediction(self, input_data):
        """
        Make predictions using the loaded model.
        """
        try:
            # Preprocess the input data
            processed_data = self.preprocess_data(input_data)
            # Make predictions
            predictions = self.model.predict(processed_data)
            return predictions
        except Exception as e:
            logging.error("Error in making prediction")
            raise customexception(e, sys)


class FlightCustomData:
    def __init__(self, airline, source, destination, total_stops, dep_time, arrival_time, duration, price):
        self.airline = airline
        self.source = source
        self.destination = destination
        self.total_stops = total_stops
        self.dep_time = dep_time
        self.arrival_time = arrival_time
        self.duration = duration
        self.price = price

    def get_data_as_df(self):
        # Create a dictionary with the data
        data_dict = {
            'Airline': [self.airline],
            'Source': [self.source],
            'Destination': [self.destination],
            'Total_Stops': [self.total_stops],
            'Duration': [self.duration],
            'Dep_Time': [self.dep_time],
            'Arrival_Time': [self.arrival_time],
            'Price': [self.price]  # You may choose to omit price for prediction purposes
        }
        
        # Convert to DataFrame
        return pd.DataFrame(data_dict)