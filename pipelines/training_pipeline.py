import os
import sys
import mlflow
import logging
from datetime import datetime

from airflow.dags.src.components.Data_ingestion import DataIngestion
from airflow.dags.src.components.Data_preprocessing import ColumnTransformation
from airflow.dags.src.components.Model_training import ModelTrainer
from cloud.s3_syncer import S3Sync

# Ensure logs are written to a file
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, "training_pipeline.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)


class TrainingPipelineConfig:
    """
    Configuration for the training pipeline
    """
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.TRAINING_BUCKET_NAME = "supply-chainopti-bucket"  # Replace with your bucket name
        self.artifact_dir = "artifacts"
        self.model_dir = "saved_models"

        # Create directories if they don't exist
        os.makedirs(self.artifact_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)



class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = ColumnTransformation()
        self.model_trainer = ModelTrainer()
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()

    def start_data_ingestion(self):
        try:
            logging.info("Starting data ingestion process")
            train_data_path, test_data_path = self.data_ingestion.ingest_and_split(
                source_type='csv',
                file_path='/Users/nitin/Desktop/supply_chain_chain_/data/FMCG_data_augmented.csv'
            )
            logging.info(f"Data ingestion completed. Train path: {train_data_path}, Test path: {test_data_path}")
            # mlflow.log_param("data_source_type", "csv")
            # mlflow.log_artifact(train_data_path, artifact_path="data")
            # mlflow.log_artifact(test_data_path, artifact_path="data")
            return train_data_path, test_data_path
        except Exception as e:
            logging.error(f"Error in data ingestion pipeline: {str(e)}")
            raise e

    def start_data_transformation(self, train_data_path, test_data_path):
        try:
            logging.info("Starting data transformation process")
            train_arr, test_arr = self.data_transformation.initiate_data_transformation(
                train_path=train_data_path,
                test_path=test_data_path
            )
            logging.info(f"Data transformation completed. Train array shape: {train_arr.shape}, Test array shape: {test_arr.shape}")
            # mlflow.log_metric("train_array_shape", train_arr.shape[0])
            # mlflow.log_metric("test_array_shape", test_arr.shape[0])
            return train_arr, test_arr
        except Exception as e:
            logging.error(f"Error in data transformation pipeline: {str(e)}")
            raise e


    def start_model_training(self, train_arr, test_arr):
        try:
            # Start a new MLflow run
            # if not mlflow.active_run():
            #     mlflow.start_run(run_name="Model_Training")

            logging.info("Starting model training process")
            best_model = self.model_trainer.initiate_model_training()

            if best_model is None:
                logging.warning("No model was successfully trained. Skipping further steps.")
                return None

            # Log the model
            logging.info("Logging the best model to MLflow")
            # mlflow.sklearn.log_model(best_model, artifact_path="model")
            logging.info("Model training completed successfully")

            return best_model

        except Exception as e:
            logging.error(f"Error in model training process: {str(e)}")
            raise e

        # finally:
        #     # End the MLflow run
        #     if mlflow.active_run():
        #         mlflow.end_run()


    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{self.training_pipeline_config.TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.artifact_dir, aws_bucket_url=aws_bucket_url)
        except Exception as e:
            logging.error(f"Error during artifact sync: {str(e)}")
            raise e

    def sync_saved_model_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{self.training_pipeline_config.TRAINING_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.model_dir, aws_bucket_url=aws_bucket_url)
        except Exception as e:
            logging.error(f"Error during model sync: {str(e)}")
            raise e

    def start_training(self):
        try:
            logging.info("Starting the complete training pipeline")
            train_data_path, test_data_path = self.start_data_ingestion()
            train_arr, test_arr = self.start_data_transformation(train_data_path, test_data_path)
            best_model = self.start_model_training(train_arr, test_arr)

            if best_model is not None:
                self.sync_artifact_dir_to_s3()
                self.sync_saved_model_dir_to_s3()

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
