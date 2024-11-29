import os
import sys
import mlflow
from airflow.dags.src.components.Data_ingestion import DataIngestion
from airflow.dags.src.components.Data_preprocessing import ColumnTransformation
from airflow.dags.src.components.Model_training import ModelTrainer
from airflow.dags.src.logging_config import logging

# Ensure logs are written to a file
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, "training_pipeline.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)


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
            mlflow.log_param("data_source_type", "csv")
            mlflow.log_artifact(train_data_path, artifact_path="data")
            mlflow.log_artifact(test_data_path, artifact_path="data")
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
            mlflow.log_metric("train_array_shape", train_arr.shape[0])
            mlflow.log_metric("test_array_shape", test_arr.shape[0])
            return train_arr, test_arr

        except Exception as e:
            logging.error(f"Error in data transformation pipeline: {str(e)}")
            raise e

    def start_model_training(self, train_arr, test_arr):
    
        try:
            logging.info("Starting model training process")
            
            # Call initiate_model_training
            best_model = self.model_trainer.initiate_model_training()

            if best_model is None:
                logging.warning("No model was successfully trained. Skipping further steps.")
                return None

            # Log the trained model with MLflow
            logging.info("Logging the best model to MLflow")
            mlflow.sklearn.log_model(best_model, artifact_path="model")

            logging.info("Model training completed successfully")
            return best_model

        except Exception as e:
            logging.error(f"Error in model training pipeline: {str(e)}")
            raise e


    def start_training(self):
   
        try:
            logging.info("Starting the complete training pipeline")
            
            # Step 1: Data Ingestion
            train_data_path, test_data_path = self.start_data_ingestion()
            
            # Step 2: Data Transformation
            train_arr, test_arr = self.start_data_transformation(train_data_path, test_data_path)
            
            # Step 3: Model Training
            best_model = self.start_model_training(train_arr, test_arr)
            
            if best_model is None:
                logging.warning("Training pipeline completed with no successful model training")
                return None

            logging.info("Training pipeline completed successfully")
            return best_model

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
