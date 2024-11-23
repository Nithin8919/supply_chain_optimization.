import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import joblib

# Set up logging configuration
logging.basicConfig(
    filename='logs/model_training.log',
    level=logging.INFO,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class ModelTrainerConfig:
    def __init__(self):
        self.trained_model_file_path = os.path.join('artifacts', 'model.pkl')
        self.train_data_path = os.path.join('artifacts', 'train_array.pkl')  # Training data path
        self.test_data_path = os.path.join('artifacts', 'test_array.pkl')    # Testing data path
        self.n_jobs = -1
        os.makedirs(os.path.dirname(self.trained_model_file_path), exist_ok=True)

def load_data(file_path):
    """
    Load and prepare data from a .pkl file
    """
    try:
        data = joblib.load(file_path)  # Directly load the .pkl file
        logging.info(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {str(e)}")
        raise

class ModelFactory:
    @staticmethod
    def get_models(n_jobs=-1):
        """Get models with optimized parameters"""
        models = {
            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                n_jobs=n_jobs,
                random_state=42,
                max_depth=10,
                min_samples_split=10
            ),
            "Decision Tree": DecisionTreeRegressor(
                random_state=42,
                max_depth=10,
                min_samples_split=10
            ),
            
            
            "XGB Regressor": XGBRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=n_jobs,
                max_depth=10,
                learning_rate=0.1,
                tree_method='hist'
            ),
            
        }
        logging.info(f"Initialized models with optimized parameters")
        return models

class ModelEvaluationStrategy:
    def __init__(self, metric='r2'):
        self.metric = metric
    
    def evaluate(self, y_true, y_pred):
        if self.metric == 'r2':
            return r2_score(y_true, y_pred)
        elif self.metric == 'mse':
            return mean_squared_error(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

def save_object(file_path, obj):
    """
    Save an object to the specified file path
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path, compress=3)
        logging.info(f"Successfully saved object to {file_path}")
    except Exception as e:
        logging.error(f"Error saving object to {file_path}: {str(e)}")
        raise

class ModelTrainer:
    def __init__(self, config=None, evaluation_strategy=None):
        self.config = config or ModelTrainerConfig()
        self.evaluation_strategy = evaluation_strategy or ModelEvaluationStrategy()
    
    def prepare_data(self, data):
        """
        Prepare features and target from the loaded numpy array
        """
        try:
            # Assuming the target is the last column in the numpy array
            X = data[:, :-1]  # All columns except the last
            y = data[:, -1]   # Only the last column
            return X, y
        except Exception as e:
            logging.error(f"Error preparing data: {str(e)}")
            raise

    def train_single_model(self, model_name, model, X_train, y_train, X_test, y_test):
        """
        Train and evaluate a single model
        """
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            score = self.evaluation_strategy.evaluate(y_test, predictions)
            training_time = time.time() - start_time
            
            logging.info(f"{model_name} - Score: {score:.4f}, Training Time: {training_time:.2f}s")
            return model_name, score, model
        except Exception as e:
            logging.error(f"Error training {model_name}: {str(e)}")
            return model_name, float('-inf'), None

    def initiate_model_training(self):
        """
        Load data, train models, and save the best model
        """
        try:
            start_time = time.time()
            logging.info("Starting model training process")

            # Load and prepare data
            train_data = load_data(self.config.train_data_path)
            test_data = load_data(self.config.test_data_path)
            
            X_train, y_train = self.prepare_data(train_data)
            X_test, y_test = self.prepare_data(test_data)
            
            logging.info(f"Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")

            # Get models with optimized parameters
            models = ModelFactory.get_models(n_jobs=self.config.n_jobs)
            
            # Train models in parallel
            results = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_model = {
                    executor.submit(
                        self.train_single_model,
                        model_name,
                        model,
                        X_train,
                        y_train,
                        X_test,
                        y_test
                    ): model_name
                    for model_name, model in models.items()
                }
                
                for future in as_completed(future_to_model):
                    results.append(future.result())

            # Process results
            model_report = {name: score for name, score, _ in results if score != float('-inf')}
            trained_models = {name: model for name, _, model in results if model is not None}

            if not model_report:
                logging.warning("No models were successfully trained")
                return None

            # Select best model
            best_model_name = max(model_report.items(), key=lambda x: x[1])[0]
            best_model = trained_models[best_model_name]
            best_score = model_report[best_model_name]

            logging.info(f"Best model: {best_model_name}, Score: {best_score:.4f}")
            
            if best_score < 0.6:
                logging.warning(f"Best model score {best_score} below threshold")
                return None

            # Save the best model
            save_object(self.config.trained_model_file_path, best_model)
            
            total_time = time.time() - start_time
            logging.info(f"Total training time: {total_time:.2f}s")
            
            return best_model

        except Exception as e:
            logging.error(f"Error in model training process: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        os.makedirs('logs', exist_ok=True)
        
        start_time = time.time()
        trainer = ModelTrainer()
        trained_model = trainer.initiate_model_training()
        
        total_time = time.time() - start_time
        logging.info(f"Complete process finished in {total_time:.2f}s")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise
