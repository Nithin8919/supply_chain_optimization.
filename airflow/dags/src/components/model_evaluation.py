import sys
import os
from src.logging_config import logging
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import pandas as pd
import numpy as np
from src.logging_config import logging
import logging
import pickle
from src.components.utils import load_object
import mlflow
from urllib.parse import urlparse





class Model_evaluation:
    def __init__(self):
        logging.info("The model evaluation has initiated!.")
        pass
    
    def evaluate_model(self,actual,pred):
        mse = mean_squared_error(actual,pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual,pred)
        r2  = r2_score(actual,pred)
        logging.info("evaluation metrics captured")
        return mse,rmse,mae,r2
    
    def initiate_model_evaluation(self, train_array, test_array):
        try:
            X_test, y_test = test_array[:,-1], test_array[:,-1]
            
            model_path = os.path.join('artifacts','model.pkl')
            
            model=load_object(model_path)

             #mlflow.set_registry_uri("")
             
            logging.info("model has register")

            tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

            print(tracking_url_type_store)



            with mlflow.start_run():

                prediction=model.predict(X_test)

                (rmse,mae,r2)=self.eval_metrics(y_test,prediction)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)

                 # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")


        except Exception as e:
            logging.error("There is an exception occured in model evaluation")
            raise e