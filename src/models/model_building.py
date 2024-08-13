import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import logging
import yaml
from sklearn.linear_model import LogisticRegression


# Setup Paths
base_dir = Path(__file__).resolve().parents[2]
print(f"[INFO]: Base Directory:", base_dir)
base_data_dir = os.path.join(base_dir, "data")
os.makedirs(base_data_dir, exist_ok=True)
raw_data_path = os.path.join(base_data_dir, "raw")
os.makedirs(raw_data_path, exist_ok=True)
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


def load_model_parameters(params_path: str) -> dict:
    """
    Load model parameters from a YAML file.

    Args:
        params_path (str): Path to the parameters YAML file.

    Returns:
        dict: Dictionary containing model parameters.

    Raises:
        FileNotFoundError: If the parameters file is not found.
        KeyError: If the required keys are not found in the parameters file.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    try:
        with open(params_path, 'r') as file:
            parameters = yaml.safe_load(file)['model_building']
            logger.info(f"[INFO]: Running Model with Following Parameters: n_estimators: {parameters['n_estimators']} | max_depth: {parameters['max_depth']} | learning_rate: {parameters['learning_rate']}")
            return parameters
    except FileNotFoundError:
        logger.info("[ERROR]: Parameters file not found.")
        raise
    except KeyError:
        logger.info("[ERROR]: Required keys not found in parameters file.")
        raise
    except yaml.YAMLError as exc:
        logger.info(f"[ERROR]: Error parsing YAML file: {exc}")
        raise

def create_directories(base_model_dir: str, base_data_dir: str) -> tuple:
    """
    Create necessary directories for model and data.

    Args:
        base_model_dir (str): Base directory for model.
        base_data_dir (str): Base directory for data.

    Returns:
        tuple: Paths for model, processed data, and features directories.
    """
    model_path = Path(base_model_dir)
    processed_data_path = Path(os.path.join(base_data_dir, "processed"))
    os.makedirs(model_path, exist_ok=True)
    return model_path, processed_data_path, 

def load_data(processed_data_path: Path) -> tuple:
    """
    Load train and test data from CSV files.

    Args:
        processed_data_path (Path): Path to the features directory.

    Returns:
        tuple: DataFrames for train and test data.

    Raises:
        FileNotFoundError: If the data files are not found.
        pd.errors.EmptyDataError: If the data files are empty.
        pd.errors.ParserError: If there is an error parsing the CSV files.
    """
    try:
        train_data = pd.read_csv(os.path.join(processed_data_path, 'train_bow.csv'))
        test_data = pd.read_csv(os.path.join(processed_data_path, 'test_bow.csv'))
        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)
        return train_data, test_data
    except FileNotFoundError:
        logger.info("[ERROR]: Data file not found.")
        raise
    except pd.errors.EmptyDataError:
        logger.info("[ERROR]: Data file is empty.")
        raise
    except pd.errors.ParserError as exc:
        logger.info(f"[ERROR]: Error parsing CSV file: {exc}")
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, parameters: dict) -> LogisticRegression:
    """
    Define and train the XGBoost model.

    Args:
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training labels.
        parameters (dict): Dictionary containing model parameters.

    Returns:
        LogisticRegression: Trained Linear Regression model.
    """
    try:
        clf = LogisticRegression(C=1, solver='liblinear', penalty='l2')
        clf.fit(X_train, y_train)
        logger.debug('Model training completed')
        return clf
    except Exception as e:
        logger.error(f"[ERROR]: Error training model: {e}")
        raise

def save_model(model: LogisticRegression, model_path: Path) -> None:
    """
    Save the trained model to a file.

    Args:
        model (xgb.XGBClassifier): Trained XGBoost model.
        model_path (Path): Path to the directory where the model will be saved.

    Raises:
        IOError: If there is an error saving the model.
    """
    try:
        with open(os.path.join(model_path, 'model.pkl'), 'wb') as file:
            pickle.dump(model, file)
    except IOError as e:
        logger.error(f"[ERROR]: Error saving model: {e}")
        raise

def main():
    """
    Main function to load parameters, create directories, load data, train model, and save the trained model.
    """
    try:
        parameters = load_model_parameters('params.yaml')
        base_dir = Path(__file__).resolve().parents[2]
        print(f"[INFO]: Base Directory:", base_dir)
        base_model_dir = os.path.join(base_dir, "models")
        os.makedirs(base_model_dir, exist_ok=True)
        base_data_dir = os.path.join(base_dir, "data")
        model_path, processed_data_path = create_directories(base_model_dir, base_data_dir)
        train_data, test_data = load_data(processed_data_path)
        X_train = train_data.iloc[:, 0:-1].values
        y_train = train_data.iloc[:, -1].values
        logger.info(X_train.shape)
        logger.info(y_train.shape)
        xgb_model = train_model(X_train, y_train, parameters)
        save_model(xgb_model, model_path)
    except Exception as e:
        logger.error(f"[ERROR]: {e}")

if __name__ == '__main__':
    main()
