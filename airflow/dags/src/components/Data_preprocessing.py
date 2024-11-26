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

def get_file_size(file_path):
    """Get file size in MB"""
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return round(size_mb, 2)

class PreprocessorStrategy(ABC):
    @abstractmethod
    def preprocessor(self, df: pd.DataFrame):
        pass

class ColumnTransformation(PreprocessorStrategy):
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def save_object(self, file_path, obj):
        """Save an object to a file."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as file:
                joblib.dump(obj, file)
            file_size = get_file_size(file_path)
            logging.info(f"Object saved to {file_path} (Size: {file_size} MB)")
        except Exception as e:
            logging.error(f"Failed to save object to {file_path}: {e}")
            raise e

    def preprocessor(self, df: pd.DataFrame):
        try:
            # Remove specified columns if they exist
            columns_to_drop = ['Ware_house_ID', 'WH_Manager_ID']
            df = df.drop(columns=columns_to_drop, errors='ignore')
            
            # Log total number of columns after dropping
            logging.info(f"Total number of columns after dropping specified features: {len(df.columns)}")
            
            # Extracting categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            logging.info(f"Number of categorical columns: {len(categorical_cols)}")
            logging.info(f"Categorical columns: {categorical_cols.tolist()}")
            
            # Pipeline for categorical features
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            # Extracting numerical columns (excluding target column)
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.difference(['product_wg_ton'])
            logging.info(f"Number of numerical columns: {len(numerical_cols)}")
            logging.info(f"Numerical columns: {numerical_cols.tolist()}")
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            transform = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_cols),
                    ("cat", cat_pipeline, categorical_cols)
                ]
            )
            
            return transform
        
        except Exception as e:
            logging.error("Exception occurred in preprocessing")
            raise e

    
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load datasets and log file sizes
            logging.info(f"Training data file size: {get_file_size(train_path)} MB")
            logging.info(f"Test data file size: {get_file_size(test_path)} MB")
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Strip whitespace from column names
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()

            logging.info(f"Initial number of columns in training data: {len(train_df.columns)}")
            logging.info(f"Columns in training data: {train_df.columns.tolist()}")

            target_column = 'product_wg_ton'

            # Check for target column existence
            if target_column not in train_df.columns:
                raise KeyError(f"Target column '{target_column}' not found in training DataFrame.")

            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[target_column], errors='ignore')
            target_feature_train_df = train_df[target_column]

            if target_column not in test_df.columns:
                raise KeyError(f"Target column '{target_column}' not found in testing DataFrame.")

            input_feature_test_df = test_df.drop(columns=[target_column], errors='ignore')
            target_feature_test_df = test_df[target_column]

            # Log shapes after splitting
            logging.info(f"Input features shape: {input_feature_train_df.shape}")
            logging.info(f"Target feature shape: {target_feature_train_df.shape}")

            # Create processing pipeline
            processing = self.preprocessor(train_df)

            # Apply transformations
            input_feature_train_arr = processing.fit_transform(input_feature_train_df)
            input_feature_test_arr = processing.transform(input_feature_test_df)

            # Save feature names for debugging
            feature_names = processing.get_feature_names_out()
            self.save_object(self.data_transformation_config.feature_names_file_path, feature_names)
            logging.info(f"Number of features after transformation: {len(feature_names)}")

            # Combine arrays
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            # Log final array shapes
            logging.info(f"Final transformed train array shape: {train_arr.shape}")
            logging.info(f"Final transformed test array shape: {test_arr.shape}")

            # Save objects and log their sizes
            self.save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=processing
            )

            self.save_object(
                file_path=self.data_transformation_config.train_array_file_path,
                obj=train_arr
            )

            self.save_object(
                file_path=self.data_transformation_config.test_array_file_path,
                obj=test_arr
            )

            return (
                train_arr,
                test_arr,
            )

        except Exception as e:
            logging.error("Exception occurred in the transformation step: %s", str(e))
            raise e


if __name__ == "__main__":
    data_transformation = ColumnTransformation()
    data_transformation.initiate_data_transformation(
        train_path="artifacts/train.csv",
        test_path="artifacts/test.csv"
    )