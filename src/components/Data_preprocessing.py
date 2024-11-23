import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from abc import ABC, abstractmethod
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
from dataclasses import dataclass
from typing import List
from sklearn.impute import SimpleImputer
import joblib
import sys

# Ensuring the script finds logging_config correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.logging_config import logging

# Config class to specify paths for the objects
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    train_array_file_path = os.path.join('artifacts', 'train_array.pkl')
    test_array_file_path = os.path.join('artifacts', 'test_array.pkl')
    feature_names_file_path = os.path.join('artifacts', 'feature_names.pkl')

class PreprocessorStrategy(ABC):
    @abstractmethod
    def preprocessor(self, df: pd.DataFrame):
        pass

class ColumnTransformation(PreprocessorStrategy):
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.target_column = 'product_wg_ton'

    def save_object(self, file_path, obj):
        """Save an object to a file."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as file:
                joblib.dump(obj, file)
            logging.info(f"Object saved to {file_path}")
        except Exception as e:
            logging.error(f"Failed to save object to {file_path}: {e}")
            raise e

    def preprocessor(self, df: pd.DataFrame):
        try:
            # Extracting categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            logging.info(f"Categorical columns: {categorical_cols.tolist()}")
            
            # Extracting numerical columns (excluding target column)
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.difference([self.target_column])
            logging.info(f"Numerical columns: {numerical_cols.tolist()}")
            
            # Validate columns
            if len(numerical_cols) == 0:
                logging.warning("No numerical columns found.")
            if len(categorical_cols) == 0:
                logging.warning("No categorical columns found.")
            
            # Pipeline for categorical features
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            # Pipeline for numerical features
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            
            # Combine pipelines for numeric and categorical columns
            transform = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_cols),
                    ("cat", cat_pipeline, categorical_cols)
                ],
                remainder='drop'  # Drops columns not specified in transformers
            )
            
            return transform
    
        except Exception as e:
            logging.error("Exception in preprocessing")
            raise e

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Clean column names
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()

            # Validate target column
            self._validate_target_column(train_df, test_df)

            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[self.target_column], errors='ignore')
            target_feature_train_df = train_df[self.target_column]

            input_feature_test_df = test_df.drop(columns=[self.target_column], errors='ignore')
            target_feature_test_df = test_df[self.target_column]

            # Create and fit preprocessing pipeline
            processing = self.preprocessor(train_df)

            # Transform features
            input_feature_train_arr = processing.fit_transform(input_feature_train_df)
            input_feature_test_arr = processing.transform(input_feature_test_df)

            # Debug: Print shapes before concatenation
            logging.info(f"Input train array shape: {input_feature_train_arr.shape}")
            logging.info(f"Target train array shape: {target_feature_train_df.shape}")
            logging.info(f"Input test array shape: {input_feature_test_arr.shape}")
            logging.info(f"Target test array shape: {target_feature_test_df.shape}")

            # Ensure target arrays are 2D and match input feature array rows
            target_feature_train_arr = target_feature_train_df.values.reshape(-1, 1)
            target_feature_test_arr = target_feature_test_df.values.reshape(-1, 1)

            # Validate array dimensions
            self._validate_array_dimensions(
                input_feature_train_arr, 
                target_feature_train_arr, 
                input_feature_test_arr, 
                target_feature_test_arr
            )

            # Combine input and target features
            train_arr = np.hstack([input_feature_train_arr, target_feature_train_arr])
            test_arr = np.hstack([input_feature_test_arr, target_feature_test_arr])

            # Save feature names
            feature_names = processing.get_feature_names_out()
            self.save_object(
                self.data_transformation_config.feature_names_file_path, 
                feature_names
            )

            # Save preprocessing artifacts
            self.save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                processing
            )
            self.save_object(
                self.data_transformation_config.train_array_file_path,
                train_arr
            )
            self.save_object(
                self.data_transformation_config.test_array_file_path,
                test_arr
            )

            return train_arr, test_arr

        except Exception as e:
            logging.error(f"Transformation error: {str(e)}")
            raise e

    def _validate_target_column(self, train_df, test_df):
        """Validate presence of target column in both datasets"""
        if self.target_column not in train_df.columns:
            raise KeyError(f"Target column '{self.target_column}' not found in training DataFrame")
        if self.target_column not in test_df.columns:
            raise KeyError(f"Target column '{self.target_column}' not found in testing DataFrame")

    def _validate_array_dimensions(self, train_input, train_target, test_input, test_target):
        """Validate array dimensions before concatenation"""
        if train_input.shape[0] != train_target.shape[0]:
            raise ValueError(f"Mismatched dimensions in training data: "
                             f"Input {train_input.shape}, Target {train_target.shape}")
        if test_input.shape[0] != test_target.shape[0]:
            raise ValueError(f"Mismatched dimensions in test data: "
                             f"Input {test_input.shape}, Target {test_target.shape}")

if __name__ == "__main__":
    data_transformation = ColumnTransformation()
    data_transformation.initiate_data_transformation(
        train_path="artifacts/train.csv",
        test_path="artifacts/test.csv" 
    )