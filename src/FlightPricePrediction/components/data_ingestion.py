import pandas as pd
import numpy as np
import os
import sys
from FlightPricePrediction.logger import logging
from FlightPricePrediction.exception import customexception
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path


class DataIngestionConfig:
    rawdata_path: str = os.path.join("artifacts", "raw.csv")
    traindata_path: str = os.path.join("artifacts", "train.csv")
    testdata_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            # Ensure to provide the correct path to the CSV file
            data = pd.read_csv(Path(os.path.join("notebooks/data", "Flight_csv.csv")))  
            logging.info("Successfully read the data from the dataset")
            
            # Create directories for artifacts if they don't exist
            os.makedirs(os.path.dirname(self.data_ingestion_config.rawdata_path), exist_ok=True)
            data.to_csv(self.data_ingestion_config.rawdata_path, index=False)
            logging.info("Successfully saved the raw data in the artifacts folder")
            
            logging.info("Performing train-test split")
            train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)  # Added random_state for reproducibility
            logging.info("Train-test split completed")
            
            # Save the split data
            train_data.to_csv(self.data_ingestion_config.traindata_path, index=False)
            test_data.to_csv(self.data_ingestion_config.testdata_path, index=False)
            logging.info("Successfully saved the train and test data in the artifacts folder")
            
            logging.info("Data ingestion part completed")            
            
            return (
                self.data_ingestion_config.traindata_path,
                self.data_ingestion_config.testdata_path
            )
        
        except Exception as e:
            logging.error("Exception occurred during data ingestion stage: %s", e)
            raise customexception(e, sys)
