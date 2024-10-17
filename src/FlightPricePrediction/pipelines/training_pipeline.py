from FlightPricePrediction.components.data_ingestion import DataIngestion
from FlightPricePrediction.components.data_transformation import DataTransformation
from FlightPricePrediction.components.model_trainer import ModelTrainer 
import os
import sys
import pandas as pd


obj=DataIngestion()

train_data_path,test_data_path =obj.initiate_data_ingestion()

data_transformation = DataTransformation()

train_arr , test_arr = data_transformation.initiate_data_transformation(train_data_path,test_data_path)

model_trainer = ModelTrainer()

model_trainer.initate_model_training(train_arr,test_arr)
