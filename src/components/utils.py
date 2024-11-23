import logging
import joblib
from sklearn.metrics import r2_score
import pickle
import sys
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    """Save the preprocessing object to a file."""
    try:
        with open(file_path, 'wb') as file:
            joblib.dump(obj, file)
        logging.info(f"Preprocessing object saved to {file_path}")
    except Exception as e:
        logging.error("Failed to save the preprocessing object.")
        raise e

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """Evaluate multiple models and return a report with R2 scores for each."""
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise e

import joblib

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return joblib.load(file_obj)
    except Exception as e:
        logging.error(f"Failed to load object from {file_path}: {e}")
        raise e