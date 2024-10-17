
import pickle
import sys
import os
from FlightPricePrediction.exception import customexception
from FlightPricePrediction.logger import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def save_object(file_path, obj):
    """
    Saves a Python object to a file using pickle.
    
    Parameters:
    - file_path (str): The path where the object will be saved.
    - obj (Any): The Python object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info(f"Object saved successfully at {file_path}")
    
    except Exception as e:
        logging.error(f"Failed to save object at {file_path}")
        raise customexception(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Trains and evaluates the provided models, returning their R2 scores.

    Parameters:
    - X_train (ndarray): Features for training.
    - y_train (ndarray): Target for training.
    - X_test (ndarray): Features for testing.
    - y_test (ndarray): Target for testing.
    - models (dict): Dictionary of models to evaluate.

    Returns:
    - model_report (dict): Dictionary containing model names and their R2 scores.
    """
    model_report = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2_value = r2_score(y_test, y_pred)
        model_report[model_name] = r2_value

        logging.info(f"Model: {model_name}, R2 Score: {r2_value * 100:.2f}%")
        
    return model_report

def load_object(file_path):
    """
    Loads a Python object from a file using pickle.
    
    Parameters:
    - file_path (str): The path where the object is saved.
    
    Returns:
    - obj (Any): The loaded Python object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.error(f"Failed to load object from {file_path}")
        raise customexception(e, sys)
