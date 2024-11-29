from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import os
import logging

from airflow.dags.src.components.Data_ingestion import DataIngestion
from airflow.dags.src.components.Data_preprocessing import ColumnTransformation
from airflow.dags.src.components.Model_training import ModelTrainer

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 3,
    'retry_delay': timedelta(minutes=1),
    'email_on_failure': False,
    'email_on_retry': False,
    'depends_on_past': False,
}

# Define the pipeline
class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = ColumnTransformation()
        self.model_trainer = ModelTrainer()

        # Define paths for intermediate data storage
        self.artifacts_path = '/Users/nitin/Desktop/supply_chain_chain_/artifacts'  # Path for artifacts
        os.makedirs(self.artifacts_path, exist_ok=True)  # Ensure the directory exists

        self.train_arr_path = os.path.join(self.artifacts_path, 'train_array.pkl')
        self.test_arr_path = os.path.join(self.artifacts_path, 'test_array.pkl')

    def data_ingestion_task(self, **kwargs):
        """Airflow task for data ingestion."""
        logging.info("Starting data ingestion process.")
        try:
            train_data_path, test_data_path = self.data_ingestion.ingest_and_split(
                source_type='csv',
                file_path='/Users/nitin/Desktop/supply_chain_chain_/data/FMCG_data_augmented.csv'
            )

            # Store paths in XCom
            kwargs['ti'].xcom_push(key='train_path', value=str(train_data_path))
            kwargs['ti'].xcom_push(key='test_path', value=str(test_data_path))
            logging.info("Data ingestion completed.")
        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise

    def data_transformation_task(self, **kwargs):
        """Airflow task for data transformation."""
        logging.info("Starting data transformation process.")
        try:
            # Retrieve paths from XCom
            ti = kwargs['ti']
            train_path = ti.xcom_pull(task_ids='data_ingestion', key='train_path')
            test_path = ti.xcom_pull(task_ids='data_ingestion', key='test_path')

            # Perform data transformation
            train_arr, test_arr = self.data_transformation.initiate_data_transformation(
                train_path=train_path,
                test_path=test_path
            )

            # Save transformed arrays
            import pickle
            with open(self.train_arr_path, 'wb') as f:
                pickle.dump(train_arr, f)
            with open(self.test_arr_path, 'wb') as f:
                pickle.dump(test_arr, f)

            logging.info("Data transformation completed.")
        except Exception as e:
            logging.error(f"Error during data transformation: {e}")
            raise

    def model_training_task(self, **kwargs):
        """Airflow task for model training."""
        logging.info("Starting model training process.")
        try:
            # Load transformed data
            import pickle
            with open(self.train_arr_path, 'rb') as f:
                train_arr = pickle.load(f)
            with open(self.test_arr_path, 'rb') as f:
                test_arr = pickle.load(f)

            # Train the model
            self.model_trainer.initiate_model_training(
                train_data=train_arr,
                test_data=test_arr
            )
            logging.info("Model training completed.")
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise

# Create the DAG
with DAG(
    dag_id='supply_chain_training_pipeline',
    default_args=default_args,
    description='Supply Chain ML Training Pipeline',
    schedule_interval='@daily',
    catchup=False
) as dag:
    pipeline = TrainingPipeline()

    data_ingestion = PythonOperator(
        task_id='data_ingestion',
        python_callable=pipeline.data_ingestion_task,
        provide_context=True
    )

    data_transformation = PythonOperator(
        task_id='data_transformation',
        python_callable=pipeline.data_transformation_task,
        provide_context=True
    )

    model_training = PythonOperator(
        task_id='model_training',
        python_callable=pipeline.model_training_task,
        provide_context=True
    )

    # Task dependencies
    data_ingestion >> data_transformation >> model_training
