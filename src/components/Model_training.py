import os
import logging
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from src.components.utils import save_object
from src.config import ModelTrainerConfig  # Import from separate config file
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Factory for model creation
class ModelFactory:
    @staticmethod
    def get_models():
        return {
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "XGB Regressor": XGBRegressor(),
            "AdaBoost Regressor": AdaBoostRegressor()
        }

# Model Evaluation Strategy to allow flexible metric selection
class ModelEvaluationStrategy:
    def __init__(self, metric='r2'):
        self.metric = metric
    
    def evaluate(self, y_true, y_pred):
        if self.metric == 'r2':
            return r2_score(y_true, y_pred)
        elif self.metric == 'mse':
            return mean_squared_error(y_true, y_pred)
        else:
            raise ValueError("Unsupported metric provided.")

# Model Trainer class
class ModelTrainer:
    def __init__(self, config=ModelTrainerConfig(), evaluation_strategy=ModelEvaluationStrategy()):
        self.config = config
        self.evaluation_strategy = evaluation_strategy
    
    
    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Initiating Model Training")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Retrieve models from the factory
            models = ModelFactory.get_models()
            model_report = self.train_and_evaluate_models(X_train, y_train, X_test, y_test, models)

            # Selecting best model based on the chosen metric
            best_model_name, best_model_score = max(model_report.items(), key=lambda item: item[1])
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.warning("No model achieved the minimum performance threshold.")
                return None

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            save_object(file_path=self.config.trained_model_file_path, obj=best_model)
            return best_model

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise e

    def train_and_evaluate_models(self, X_train, y_train, X_test, y_test, models):
        model_report = {}
        for model_name, model in models.items():
            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                score = self.evaluation_strategy.evaluate(y_test, predictions)
                model_report[model_name] = score
                logging.info(f"{model_name} score: {score}")
            except Exception as e:
                logging.error(f"Error training {model_name}: {e}")
        return model_report