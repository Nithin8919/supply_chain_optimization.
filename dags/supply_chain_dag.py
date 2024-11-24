from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys
from src.components.Data_ingestion import DataIngestion
from src.components.Data_preprocessing import ColumnTransformation
from src.components.Model_training import ModelTrainer
from src.components.model_evaluation import Model_evaluation
from src.logging_config import logging
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath('/Users/nitin/Desktop/supply_chain_chain_/src'))

from src.components.Data_ingestion import DataIngestion


# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
    'email_on_retry': False,
    'depends_on_past': False
}

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = ColumnTransformation()
        self.model_trainer = ModelTrainer()
        
        # Define paths for intermediate data storage
        self.base_path = '/opt/airflow/data'  # Adjust this path according to your Airflow setup
        self.train_path = os.path.join(self.base_path, 'train_data.csv')
        self.test_path = os.path.join(self.base_path, 'test_data.csv')
        self.train_arr_path = os.path.join(self.base_path, 'train_arr.pkl')
        self.test_arr_path = os.path.join(self.base_path, 'test_arr.pkl')

    def data_ingestion_task(self, **context):
        """
        Airflow task for data ingestion
        """
        try:
            logging.info("Starting data ingestion process")
            train_data_path, test_data_path = self.data_ingestion.ingest_and_split(
                source_type='csv',
                file_path='/Users/nitin/Desktop/supply_chain_chain_/data/FMCG_data_augmented.csv'
            )
            
            # Store the paths in XCom for the next task
            context['task_instance'].xcom_push(key='train_path', value=train_data_path)
            context['task_instance'].xcom_push(key='test_path', value=test_data_path)
            
            logging.info(f"Data ingestion completed. Train path: {train_data_path}, Test path: {test_data_path}")
            
        except Exception as e:
            logging.error(f"Error in data ingestion task: {str(e)}")
            raise e

    def data_transformation_task(self, **context):
        """
        Airflow task for data transformation
        """
        try:
            logging.info("Starting data transformation process")
            
            # Get paths from previous task
            ti = context['task_instance']
            train_path = ti.xcom_pull(task_ids='data_ingestion', key='train_path')
            test_path = ti.xcom_pull(task_ids='data_ingestion', key='test_path')
            
            train_arr, test_arr = self.data_transformation.initiate_data_transformation(
                train_path=train_path,
                test_path=test_path
            )
            
            # Save arrays to disk for next task
            import numpy as np
            np.save(self.train_arr_path, train_arr)
            np.save(self.test_arr_path, test_arr)
            
            logging.info(f"Data transformation completed. Arrays saved at: {self.train_arr_path}, {self.test_arr_path}")
            
        except Exception as e:
            logging.error(f"Error in data transformation task: {str(e)}")
            raise e

    def model_training_task(self, **context):
        """
        Airflow task for model training
        """
        try:
            logging.info("Starting model training process")
            
            # Load arrays
            import numpy as np
            train_arr = np.load(self.train_arr_path)
            test_arr = np.load(self.test_arr_path)
            
            # Train model
            training_result = self.model_trainer.initiate_model_training()
            
            logging.info("Model training completed successfully")
            return training_result
            
        except Exception as e:
            logging.error(f"Error in model training task: {str(e)}")
            raise e

# Create the DAG
with DAG(
    'supply_chain_training_pipeline',
    default_args=default_args,
    description='Supply Chain ML Training Pipeline',
    schedule_interval='@daily',  # Adjust as needed
    catchup=False
) as dag:
    
    # Instantiate the pipeline
    pipeline = TrainingPipeline()
    
    # Create tasks
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
    
    # Define task dependencies
    data_ingestion >> data_transformation >> model_training