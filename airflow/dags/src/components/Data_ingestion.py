import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sqlalchemy.engine.url import URL
from dataclasses import dataclass
from pathlib import Path
from sqlalchemy import create_engine
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from sqlalchemy.engine.url import URL
# Singleton configuration class
@dataclass
class DataIngestionConfig:
   """Configuration class for data ingestion."""
   artifacts_dir: Path = Path('artifacts')
   train_data_path: Path = artifacts_dir / 'train.csv'
   test_data_path: Path = artifacts_dir / 'test.csv'
   raw_data_path: Path = artifacts_dir / 'raw.csv'
   test_size: float = 0.2
   random_state: int = 42


   def __post_init__(self):
       self.artifacts_dir.mkdir(parents=True, exist_ok=True)


# Abstract class for data ingestion
class DataIngestionStrategy(ABC):
   @abstractmethod
   def ingest_data(self):
       pass


# Concrete class for CSV ingestion
class ExcelDataIngestion(DataIngestionStrategy):
   def __init__(self, file_path: str):
       self.file_path = file_path


   def ingest_data(self):
       """Ingest data from a CSV file."""
       logging.info(f"Ingesting data from {self.file_path}")
       df = pd.read_csv(self.file_path)
       logging.info("Data ingestion from CSV completed.")
       return df


# Concrete class for PostgreSQL ingestion
class PostgreSQLDataIngestion(DataIngestionStrategy):
   def __init__(self, db_params: Dict[str, Any], table_name: str):
       self.db_params = db_params
       self.table_name = table_name


   def ingest_data(self):
       """Ingest data from PostgreSQL."""
       logging.info(f"Ingesting data from PostgreSQL: {self.db_params['host']}, Table: {self.table_name}")
       engine = create_engine(URL.create(**self.db_params))
       df = pd.read_sql_table(self.table_name, con=engine)
       logging.info("Data ingestion from PostgreSQL completed.")
       return df


# Factory to create the appropriate data ingestion strategy
class DataIngestionFactory:
   @staticmethod
   def get_ingestion_strategy(source_type: str, **kwargs) -> DataIngestionStrategy:
       if source_type == 'csv':
           return ExcelDataIngestion(kwargs['file_path'])
       elif source_type == 'postgres':
           return PostgreSQLDataIngestion(kwargs['connection_string'], kwargs['table_name'])
       else:
           raise ValueError(f"Unknown source type: {source_type}")


# Main Data Ingestion Class
class DataIngestion:
   def __init__(self):
       self.config = DataIngestionConfig()
  
   def ingest_and_split(self, source_type: str, **kwargs):
       """Executes data ingestion and splits the data into train and test sets."""
       try:
           # Get the appropriate ingestion strategy
           ingestion_strategy = DataIngestionFactory.get_ingestion_strategy(source_type, **kwargs)
           df = ingestion_strategy.ingest_data()


           # Save raw data
           logging.info(f"Saving raw data to {self.config.raw_data_path}")
           df.to_csv(self.config.raw_data_path, index=False)


           # Split data into train and test sets
           logging.info("Splitting data into train and test sets...")
           train_data, test_data = train_test_split(df, test_size=self.config.test_size, random_state=self.config.random_state)


           # Save train and test datasets
           train_data.to_csv(self.config.train_data_path, index=False)
           test_data.to_csv(self.config.test_data_path, index=False)
           logging.info(f"Train data saved to {self.config.train_data_path}")
           logging.info(f"Test data saved to {self.config.test_data_path}")


           return self.config.train_data_path, self.config.test_data_path


       except Exception as e:
          
           raise e


# Usage
if __name__ == "__main__":
   data_ingestion = DataIngestion()


   # Ingest data from CSV
   train_path, test_path = data_ingestion.ingest_and_split(
       source_type='csv',
       file_path='FMCG_data_augmented.csv'
   )
