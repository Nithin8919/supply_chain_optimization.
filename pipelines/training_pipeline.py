import os
import sys
from airflow.dags.src.components.Data_ingestion import DataIngestion
from airflow.dags.src.components.Data_preprocessing import ColumnTransformation
from airflow.dags.src.components.Model_training import ModelTrainer
from airflow.dags.src.components.model_evaluation import Model_evaluation
from airflow.dags.src.logging_config import logging

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = ColumnTransformation()
        self.model_trainer = ModelTrainer()

    def start_data_ingestion(self):
        """
        Initiates the data ingestion process
        """
        try:
            logging.info("Starting data ingestion process")
            train_data_path, test_data_path = self.data_ingestion.ingest_and_split(
                source_type='csv',
                file_path='/Users/nitin/Desktop/supply_chain_chain_/data/FMCG_data_augmented.csv'
            )
            logging.info(f"Data ingestion completed. Train path: {train_data_path}, Test path: {test_data_path}")
            return train_data_path, test_data_path
        
        except Exception as e:
            logging.error(f"Error in data ingestion pipeline: {str(e)}")
            raise e

    def start_data_transformation(self, train_data_path, test_data_path):
        """
        Initiates the data transformation process
        """
        try:
            logging.info("Starting data transformation process")
            train_arr, test_arr = self.data_transformation.initiate_data_transformation(
                train_path=train_data_path,
                test_path=test_data_path
            )
            logging.info(f"Data transformation completed. Train array shape: {train_arr.shape}, Test array shape: {test_arr.shape}")
            return train_arr, test_arr
        
        except Exception as e:
            logging.error(f"Error in data transformation pipeline: {str(e)}")
            raise e

    def start_model_training(self, train_arr, test_arr):
        """
        Initiates the model training process
        """
        try:
            logging.info("Starting model training process")
            # Create an instance of model trainer with the arrays
            training_result = self.model_trainer.initiate_model_training(
            )
            logging.info("Model training completed successfully")
            return training_result
        
        except Exception as e:
            logging.error(f"Error in model training pipeline: {str(e)}")
            raise e

    def start_training(self):
        """
        Executes the complete training pipeline
        """
        try:
            logging.info("Starting the complete training pipeline")
            
            # Step 1: Data Ingestion
            train_data_path, test_data_path = self.start_data_ingestion()
            
            # Step 2: Data Transformation
            train_arr, test_arr = self.start_data_transformation(train_data_path, test_data_path)
            
            # Step 3: Model Training
            training_result = self.start_model_training(train_arr, test_arr)
            
            logging.info("Training pipeline completed successfully")
            return training_result
        
        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise e


if __name__ == "__main__":
    try:
        logging.info("Initializing training pipeline")
        training_pipeline_obj = TrainingPipeline()
        training_pipeline_obj.start_training()
        logging.info("Training pipeline execution completed")
    except Exception as e:
        logging.error(f"Training pipeline failed: {str(e)}")
        raise e