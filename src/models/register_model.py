import os
import json
import logging
import mlflow
from pathlib import Path

# Set up the MLflow tracking URI
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    from dotenv import load_dotenv
    load_dotenv()
    print("[INFO]: Loading environment variables from .env file")
    dagshub_token = os.getenv("DAGSHUB_PAT")
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Set the MLflow tracking URI
mlflow.set_tracking_uri('https://dagshub.com/insculptor/mlops-mini-project.mlflow')

# Setup Paths
base_dir = Path(__file__).resolve().parents[2]
print(f"[INFO]: Base Directory:", base_dir)
base_model_dir = os.path.join(base_dir, "models")
os.makedirs(base_model_dir, exist_ok=True)
logger_path = os.path.join(base_dir, "logs", "model_registration.log")
os.makedirs(os.path.dirname(logger_path), exist_ok=True)
model_path = os.path.join(base_model_dir, "model.pkl")

# Logging Configuration
def setup_logger(log_file_path):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()
    return logger

logger = setup_logger(logger_path)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.info('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry and transition it to Staging and Production."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        logger.info(f'Model {model_name} version {model_version.version} registered.')
        
        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logger.info(f'Model {model_name} version {model_version.version} transitioned to Staging.')

        # Optionally, you can also transition to "Production"
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production"
        )
        logger.info(f'Model {model_name} version {model_version.version} transitioned to Production.')

    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        model_name = "model"
        logger.info('Registering model %s', model_name)
        logger.info('Model info: %s', model_info)
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
