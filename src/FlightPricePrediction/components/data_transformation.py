# FlightPricePrediction/components/data_transformation.py

import pandas as pd
import numpy as np
import os
import sys

from FlightPricePrediction.logger import logging
from FlightPricePrediction.exception import customexception

from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from FlightPricePrediction.utils.utils import save_object

class DataTransformationConfig:
    """
    Configuration for Data Transformation.
    """
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """
    Data Transformation class to handle preprocessing of flight price prediction data.
    """

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        """
        Creates and returns a preprocessing pipeline object.
        """
        try:
            logging.info("Data Transformation initiated.")

            # Define categorical and numerical columns
            categorical_cols = ['Airline', 'Source', 'Destination', 'Total_Stops']
            numerical_cols = ['Duration', 'Dep_hour', 'Dep_minute', 'Arr_hour', 'Arr_minute', 'Journey_day', 'Journey_month']

            # Create pipelines for numerical and categorical data
            num_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="median")),
                    ("Scaler", StandardScaler())
                ]
            )

            # Use OneHotEncoder for nominal categorical variables
            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="most_frequent")),
                    ("OneHotEncoder", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            # Combine pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_cols),
                    ("cat_pipeline", cat_pipeline, categorical_cols)
                ]
            )

            logging.info("Preprocessing Pipeline Created Successfully.")

            return preprocessor

        except Exception as e:
            logging.error("Exception occurred in get_data_transformation_object method.")
            raise customexception(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        """
        Initiates the data transformation process for train and test datasets.

        Parameters:
        - train_data_path (str): Path to the training data CSV file.
        - test_data_path (str): Path to the testing data CSV file.

        Returns:
        - tuple: Transformed train and test arrays.
        """
        try:
            # Read the train and test data
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Read train and test data successfully.")

            # Feature Engineering: Extract time and date features
            logging.info("Starting Feature Engineering.")

            # Function to convert duration to minutes
            def convert_duration(duration):
                h, m = 0, 0
                if 'h' in duration:
                    h_part = duration.split('h')[0]
                    h = int(h_part)
                    duration = duration.split('h')[1]
                if 'm' in duration:
                    m_part = duration.split('m')[0]
                    m = int(m_part)
                return h * 60 + m

            # Apply to 'Duration' column
            for df in [train_df, test_df]:
                df['Duration'] = df['Duration'].apply(convert_duration)

                # Split 'Dep_Time' into 'Dep_hour' and 'Dep_minute'
                df[['Dep_hour', 'Dep_minute']] = df['Dep_Time'].str.split(':', expand=True).astype(int)

                # Handle 'Arrival_Time' which may contain date information
                df['Arrival_Time'] = df['Arrival_Time'].str.split().str[0]
                df[['Arr_hour', 'Arr_minute']] = df['Arrival_Time'].str.split(':', expand=True).astype(int)

                # Extract 'Journey_day' and 'Journey_month' from 'Date_of_Journey'
                df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y')
                df['Journey_day'] = df['Date_of_Journey'].dt.day
                df['Journey_month'] = df['Date_of_Journey'].dt.month

                # Drop original columns after feature engineering
                df.drop(['Date_of_Journey', 'Dep_Time', 'Arrival_Time'], axis=1, inplace=True)

            logging.info("Feature Engineering completed successfully.")

            # Define target and features
            target_column = 'Price'
            drop_columns = [target_column, "Route","Additional_Info"]  # Assuming 'id' is present

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)

            target_feature_train_df = train_df[target_column]
            target_feature_test_df = test_df[target_column]

            logging.info("Separated input features and target feature.")

            # Get the preprocessing object
            preprocessing_obj = self.get_data_transformation_object()

            # Fit and transform the training data, transform the testing data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applied preprocessing on train and test data.")

            # Concatenate the target variable with the transformed features
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Created the final train and test arrays.")

            # Save the preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessing object saved successfully.")

            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            logging.error("Exception occurred in initiate_data_transformation method.")
            raise customexception(e, sys)
