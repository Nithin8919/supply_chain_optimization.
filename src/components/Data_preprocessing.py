import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from abc import ABC, abstractmethod
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict
from sklearn.impute import SimpleImputer
import joblib
import sys
from scipy import sparse
from sklearn.feature_selection import SelectKBest, f_regression

# Ensuring the script finds logging_config correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.logging_config import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    train_array_file_path: str = os.path.join('artifacts', 'train_array.pkl')
    test_array_file_path: str = os.path.join('artifacts', 'test_array.pkl')
    feature_names_file_path: str = os.path.join('artifacts', 'feature_names.pkl')
    compression_protocol: int = 4
    max_features: int = 1000

class PreprocessorStrategy(ABC):
    """Abstract base class for preprocessing strategies"""
    @abstractmethod
    def preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """Abstract method for creating preprocessing pipeline"""
        pass

    @abstractmethod
    def initiate_data_transformation(self, train_path: str, test_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Abstract method for initiating data transformation"""
        pass

class MemoryOptimizer:
    @staticmethod
    def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
        """Reduce memory usage of a dataframe by choosing better dtypes"""
        start_mem = df.memory_usage().sum() / 1024**2
        logging.info(f'Memory usage of dataframe is {start_mem:.2f} MB')
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float32)
        
        end_mem = df.memory_usage().sum() / 1024**2
        logging.info(f'Memory usage after optimization is: {end_mem:.2f} MB')
        logging.info(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
        return df

class ColumnTransformation(PreprocessorStrategy):
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.target_column = 'product_wg_ton'
        self.memory_optimizer = MemoryOptimizer()
        
    def save_object(self, file_path: str, obj) -> None:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as file:
                joblib.dump(obj, file, compress=('gzip', 3), protocol=self.data_transformation_config.compression_protocol)
            
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            logging.info(f"Object saved to {file_path} (Size: {file_size:.2f} MB)")
        except Exception as e:
            logging.error(f"Failed to save object to {file_path}: {e}")
            raise e

    def get_categorical_cardinality(self, df: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, int]:
        """Get cardinality of categorical columns to identify high-cardinality features"""
        cardinality = {}
        for col in categorical_cols:
            cardinality[col] = df[col].nunique()
        return cardinality

    def preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        try:
            categorical_cols = df.select_dtypes(include=['object']).columns
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.difference([self.target_column])
            
            # Get cardinality of categorical columns
            cardinality = self.get_categorical_cardinality(df, categorical_cols)
            
            # Separate high and low cardinality categorical columns
            high_card_threshold = 10
            high_card_cols = [col for col, card in cardinality.items() if card > high_card_threshold]
            low_card_cols = [col for col, card in cardinality.items() if card <= high_card_threshold]
            
            logging.info(f"High cardinality columns: {len(high_card_cols)}")
            logging.info(f"Low cardinality columns: {len(low_card_cols)}")
            
            transformers = []
            
            # Numerical pipeline with feature selection
            if len(numerical_cols) > 0:
                num_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(copy=False)),
                    ("selector", SelectKBest(f_regression, k=min(50, len(numerical_cols))))
                ])
                transformers.append(("num", num_pipeline, numerical_cols))
            
            # Low cardinality categorical pipeline
            if len(low_card_cols) > 0:
                low_card_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32))
                ])
                transformers.append(("cat_low", low_card_pipeline, low_card_cols))
            
            # High cardinality categorical pipeline with feature selection
            if len(high_card_cols) > 0:
                high_card_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32, max_categories=20))
                ])
                transformers.append(("cat_high", high_card_pipeline, high_card_cols))
            
            transform = ColumnTransformer(
                transformers=transformers,
                remainder='drop',
                sparse_threshold=0.3
            )
            
            return transform
        
        except Exception as e:
            logging.error(f"Exception in preprocessing: {e}")
            raise e

    def initiate_data_transformation(self, train_path: str, test_path: str) -> Tuple[np.ndarray, np.ndarray]:
        try:
            # Load and optimize memory usage of dataframes
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Optimizing memory usage of training data...")
            train_df = self.memory_optimizer.reduce_mem_usage(train_df)
            logging.info("Optimizing memory usage of testing data...")
            test_df = self.memory_optimizer.reduce_mem_usage(test_df)
            
            logging.info(f"Training dataset shape: {train_df.shape}")
            logging.info(f"Testing dataset shape: {test_df.shape}")
            
            # Clean column names
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()
            
            # Validate target column
            self._validate_target_column(train_df, test_df)
            
            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[self.target_column])
            target_feature_train_df = train_df[self.target_column]
            input_feature_test_df = test_df.drop(columns=[self.target_column])
            target_feature_test_df = test_df[self.target_column]
            
            # Create and fit preprocessing pipeline
            processing = self.preprocessor(train_df)
            
            # Transform features with progress logging
            logging.info("Starting feature transformation...")
            logging.info("Transforming training features...")
            input_feature_train_arr = processing.fit_transform(input_feature_train_df)
            logging.info("Transforming testing features...")
            input_feature_test_arr = processing.transform(input_feature_test_df)
            
            # Convert sparse matrices to dense if needed
            if sparse.issparse(input_feature_train_arr):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if sparse.issparse(input_feature_test_arr):
                input_feature_test_arr = input_feature_test_arr.toarray()
            
            # Log feature reduction information
            logging.info(f"Original feature count: {input_feature_train_df.shape[1]}")
            logging.info(f"Reduced feature count: {input_feature_train_arr.shape[1]}")
            
            # Convert target arrays
            target_feature_train_arr = target_feature_train_df.values.reshape(-1, 1)
            target_feature_test_arr = target_feature_test_df.values.reshape(-1, 1)
            
            # Combine features and target
            train_arr = np.hstack([input_feature_train_arr, target_feature_train_arr])
            test_arr = np.hstack([input_feature_test_arr, target_feature_test_arr])
            
            # Save artifacts with detailed logging
            logging.info("Saving preprocessor object...")
            self.save_object(self.data_transformation_config.preprocessor_obj_file_path, processing)
            
            logging.info("Saving training array...")
            self.save_object(self.data_transformation_config.train_array_file_path, train_arr)
            
            logging.info("Saving testing array...")
            self.save_object(self.data_transformation_config.test_array_file_path, test_arr)
            
            # Save feature names
            feature_names = processing.get_feature_names_out()
            self.save_object(self.data_transformation_config.feature_names_file_path, feature_names)
            
            return train_arr, test_arr

        except Exception as e:
            logging.error(f"Transformation error: {str(e)}")
            raise e

    def _validate_target_column(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        if self.target_column not in train_df.columns:
            raise KeyError(f"Target column '{self.target_column}' not found in training DataFrame")
        if self.target_column not in test_df.columns:
            raise KeyError(f"Target column '{self.target_column}' not found in testing DataFrame")

if __name__ == "__main__":
    data_transformation = ColumnTransformation()
    data_transformation.initiate_data_transformation(
        train_path="artifacts/train.csv",
        test_path="artifacts/test.csv"
    )