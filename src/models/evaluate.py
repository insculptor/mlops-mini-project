import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
import yaml
import mlflow
import dagshub
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from dvclive import Live



# Setup Paths
base_dir = Path(__file__).resolve().parents[2]
print(f"[INFO]: Base Directory:", base_dir)
base_data_dir = os.path.join(base_dir, "data")
os.makedirs(base_data_dir, exist_ok=True)
raw_data_path = os.path.join(base_data_dir, "raw")
os.makedirs(raw_data_path, exist_ok=True)
reports_path = os.path.join(base_dir, "reports")
os.makedirs(reports_path, exist_ok=True)
logger_path = os.path.join(base_dir, "logs")
os.makedirs(os.path.dirname(logger_path), exist_ok=True)
log_file = os.path.join(logger_path, "model_building.log")
## Logging Configuration
def setup_logger(log_file_path):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()
    return logger
logger = setup_logger(log_file)





def create_directories(base_data_dir: str) -> Path:
    """
    Create necessary directories for features.

    Args:
        base_data_dir (str): Base directory for data.

    Returns:
        Path: Path for features directory.
    """
    features_path = Path(os.path.join(base_data_dir, "processed"))
    os.makedirs(features_path, exist_ok=True)
    logger.info(f"[INFO]: Features Directory: {features_path}")
    return features_path

def load_model(model_path: str):
    """
    Load the trained model from a file.

    Args:
        model_path (str): Path to the model file.

    Returns:
        The loaded model.

    Raises:
        FileNotFoundError: If the model file is not found.
        IOError: If there is an error loading the model.
    """
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            logger.info(f"[INFO]: Model loaded from {model_path}")
            return model
    except FileNotFoundError:
        print("[ERROR]: Model file not found.")
        raise
    except IOError as e:
        print(f"[ERROR]: Error loading model: {e}")
        raise

def load_test_data(features_path: Path) -> tuple:
    """
    Load test data from a CSV file.

    Args:
        features_path (Path): Path to the features directory.

    Returns:
        tuple: Test feature data and labels.

    Raises:
        FileNotFoundError: If the data file is not found.
        pd.errors.EmptyDataError: If the data file is empty.
        pd.errors.ParserError: If there is an error parsing the CSV file.
    """
    try:
        test_data = pd.read_csv(os.path.join(features_path, 'test_bow.csv'))
        X_test = test_data.iloc[:, 0:-1].values
        y_test = test_data.iloc[:, -1].values
        logger.info(f"[INFO]: Test Data Loaded.")
        logger.info(f"[INFO]: Test Data Shape: {test_data.shape}")
        return X_test, y_test
    except FileNotFoundError:
        print("[ERROR]: Data file not found.")
        raise
    except pd.errors.EmptyDataError:
        print("[ERROR]: Data file is empty.")
        raise
    except pd.errors.ParserError as exc:
        print(f"[ERROR]: Error parsing CSV file: {exc}")
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate the model using test data.

    Args:
        model: The trained model.
        X_test (np.ndarray): Test feature data.
        y_test (np.ndarray): Test labels.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        logger.info(f"[INFO]: Model Evaluation Metrics:")
        logger.info(f"[INFO]: Accuracy: {accuracy}")
        logger.info(f"[INFO]: Precision: {precision}")
        logger.info(f"[INFO]: Recall: {recall}")
        logger.info(f"[INFO]: AUC: {auc}")


        
        with open("params.yaml", "r") as file:
            params = yaml.safe_load(file)

        #Log metrics and parameters using dvclive
            
        with Live(save_dvc_exp=True) as live:

            live.log_metric("accuracy", accuracy)
            live.log_metric("precision", precision)
            live.log_metric("recall", recall)
            live.log_metric("auc", auc)


            for param, value in params.items():
                for key, val in value.items():

                    live.log_param(f'{param}_{key}', val)

        # Save metrics to a JSON file for compatibility  with DVC
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        return metrics_dict
    except Exception as e:
        print(f"[ERROR]: Error during model evaluation: {e}")
        raise

def save_metrics(metrics_dict: dict, reports_path: str) -> None:
    """
    Save evaluation metrics to a JSON file.

    Args:
        metrics_dict (dict): Dictionary containing evaluation metrics.
        reports_path (str): Path to the JSON file.

    Raises:
        IOError: If there is an error saving the metrics.
    """
    try:
        #Load the parameters for logging
        save_metrics_path = os.path.join(reports_path, "metrics.json")
        print("*"*50)
        print(f"[INFO]: Saving metrics to {save_metrics_path}")
        with open(save_metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
    except IOError as e:
        print(f"[ERROR]: Error saving metrics: {e}")
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.info('Model info saved to %s', file_path)
    except Exception as e:
        print('Error occurred while saving the model info: %s', e)
        raise
    
def main():
    """
    Main function to load model, load test data, evaluate model, and save evaluation metrics.
    """
    base_dir = Path(__file__).resolve().parents[2]
    print(f"[INFO]: Base Directory:", base_dir)
    base_model_dir = os.path.join(base_dir, "models")
    os.makedirs(base_model_dir, exist_ok=True)
    base_data_dir = os.path.join(base_dir, "data")
    model_path = os.path.join(Path(base_model_dir), "model.pkl")
    features_path = create_directories(base_data_dir)
    logger.info(f"[INFO]: Model Path: {model_path}")
    
    # Set up the MLflow tracking URI
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("[INFO]: Loading environment variables from .env file")
        dagshub_token = os.getenv("DAGSHUB_PAT")
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    # Set the MLflow tracking URI
    mlflow.set_tracking_uri('https://dagshub.com/insculptor/mlops-mini-project.mlflow')

    clf = load_model(model_path)
    X_test, y_test = load_test_data(features_path)
    metrics_dict = evaluate_model(clf, X_test, y_test)

    
    mlflow.set_experiment("dvc-pipeline")
    with mlflow.start_run() as run:  # Start an MLflow run
        try:
            clf = load_model(model_path)           
            metrics = evaluate_model(clf, X_test, y_test)
                        
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model parameters to MLflow
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(clf, "model")
            
            # Save model info
            print(f"[INFO]: Model Evaluation Metrics: {metrics_dict}")
            save_metrics(metrics_dict, reports_path)
            
            # Save model info
            save_model_info(run.info.run_id, "model", os.path.join(reports_path,"model_info.json"))
            
            # Log the metrics file to MLflow
            mlflow.log_artifact(os.path.join(reports_path,"metrics.json"))

            # Log the model info file to MLflow
            mlflow.log_artifact(os.path.join(reports_path,"model_info.json"))

        except Exception as e:
            print(f"[ERROR]: {e}")

if __name__ == '__main__':
    main()