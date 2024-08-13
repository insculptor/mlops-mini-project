import os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import pickle
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Setup Paths
base_dir = Path(__file__).resolve().parents[2]
print(f"[INFO]: Base Directory:", base_dir)
base_data_dir = os.path.join(base_dir, "data")
os.makedirs(base_data_dir, exist_ok=True)
raw_data_path = os.path.join(base_data_dir, "raw")
os.makedirs(raw_data_path, exist_ok=True)
logger_path = os.path.join(base_dir, "logs")
os.makedirs(os.path.dirname(logger_path), exist_ok=True)
log_file = os.path.join(logger_path, "feature_engineering.log")
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


def load_max_features(params_path: str) -> int:
    """
    Load the maximum number of features for CountVectorizer from a YAML file.

    Args:
        params_path (str): Path to the parameters YAML file.

    Returns:
        int: Maximum number of features for CountVectorizer.

    Raises:
        FileNotFoundError: If the parameters file is not found.
        KeyError: If the required keys are not found in the parameters file.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            max_features = params['feature_engineering']['max_features']
            model = params['feature_engineering']['model_type']
            logger.info(f"[INFO]: Running Model with Following Parameters: max_features: {max_features} | model: {model}")
            return max_features,model
    except FileNotFoundError:
        logger.error("[ERROR]: Parameters file not found.")
        raise
    except KeyError:
        print("[ERROR]: Key 'feature_engineering' or 'max_features' not found in parameters file.")
        raise
    except yaml.YAMLError as exc:
        print(f"[ERROR]: Error parsing YAML file: {exc}")
        raise

def create_directories(base_data_dir: str) -> tuple:
    """
    Create necessary directories for processed data and features.

    Args:
        base_data_dir (str): Base directory for data.

    Returns:
        tuple: Paths for processed data and features directories.
    """
    interim_data_path = Path(os.path.join(base_data_dir, "interim"))
    processed_data_path = Path(os.path.join(base_data_dir, "processed"))
    os.makedirs(processed_data_path, exist_ok=True)
    logger.info(f"[INFO]: Created Processed Data Directory: {processed_data_path}")
    return processed_data_path, interim_data_path

def load_data(interim_data_path: Path) -> tuple:
    """
    Load train and test data from CSV files.

    Args:
        processed_data_path (Path): Path to the processed data directory.

    Returns:
        tuple: DataFrames for train and test data.

    Raises:
        FileNotFoundError: If the data files are not found.
        pd.errors.EmptyDataError: If the data files are empty.
        pd.errors.ParserError: If there is an error parsing the CSV files.
    """
    try:
        train_data = pd.read_csv(os.path.join(interim_data_path, 'train_processed.csv'))
        test_data = pd.read_csv(os.path.join(interim_data_path, 'test_processed.csv'))
        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)
        logger.info(f"[INFO]: Loaded Train and Test Data.")
        return train_data, test_data
    except FileNotFoundError:
        print("[ERROR]: Data file not found.")
        raise
    except pd.errors.EmptyDataError:
        print("[ERROR]: Data file is empty.")
        raise
    except pd.errors.ParserError as exc:
        print(f"[ERROR]: Error parsing CSV file: {exc}")
        raise

def apply_bag_of_words(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """
    Apply Bag of Words (CountVectorizer) to the train and test data.

    Args:
        train_data (pd.DataFrame): DataFrame containing the train data.
        test_data (pd.DataFrame): DataFrame containing the test data.
        max_features (int): Maximum number of features for CountVectorizer.

    Returns:
        tuple: Transformed train and test data, and their corresponding labels.
    """
    X_train = train_data['content'].values
    y_train = train_data['sentiment'].values
    X_test = test_data['content'].values
    y_test = test_data['sentiment'].values

    vectorizer = CountVectorizer(max_features=max_features)
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)
    
    base_dir = Path(__file__).resolve().parents[2]
    base_model_dir = os.path.join(base_dir, "models")
    os.makedirs(base_model_dir, exist_ok=True)
    vectorizer_file_path = os.path.join(base_model_dir,"vectorizer.pkl")
    pickle.dump(vectorizer, open(vectorizer_file_path,'wb'))
    logger.info(f"[INFO]: Applied Bag of Words with {max_features} features.")
    return X_train_bow, y_train, X_test_bow, y_test

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """
    Apply TFIDF Vectorizer to the train and test data.

    Args:
        train_data (pd.DataFrame): DataFrame containing the train data.
        test_data (pd.DataFrame): DataFrame containing the test data.
        max_features (int): Maximum number of features for CountVectorizer.

    Returns:
        tuple: Transformed train and test data, and their corresponding labels.
    """
    X_train = train_data['content'].values
    y_train = train_data['sentiment'].values
    X_test = test_data['content'].values
    y_test = test_data['sentiment'].values

    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)
    
    base_dir = Path(__file__).resolve().parents[2]
    base_model_dir = os.path.join(base_dir, "models")
    os.makedirs(base_model_dir, exist_ok=True)
    vectorizer_file_path = os.path.join(base_model_dir,"vectorizer.pkl")
    pickle.dump(vectorizer, open(vectorizer_file_path,'wb'))

    return X_train_bow, y_train, X_test_bow, y_test

def save_transformed_data(X_bow: np.ndarray, y: np.ndarray, interim_data_path: Path, file_name: str) -> None:
    """
    Save transformed data into a CSV file.

    Args:
        X_bow (np.ndarray): Transformed feature data.
        y (np.ndarray): Labels corresponding to the feature data.
        interim_data_path (Path): Path to the directory where the CSV file will be saved.
        file_name (str): Name of the CSV file to save.
    
    Raises:
        IOError: If there is an error saving the CSV file.
    """
    try:
        df = pd.DataFrame(X_bow.toarray())
        df['label'] = y
        df.to_csv(os.path.join(interim_data_path, file_name), index=False)
    except IOError as e:
        print(f"[ERROR]: Error saving CSV file: {e}")
        raise

def main():
    """
    Main function to load parameters, create directories, load data, apply Bag of Words, and save transformed data.
    """
    try:
        max_features, model = load_max_features('params.yaml')
        # Get the base directory where the Python script is running
        base_dir = Path(__file__).resolve().parents[2]
        print(f"[INFO]: Base Directory:", base_dir)
        base_data_dir = os.path.join(base_dir, "data")
        os.makedirs(base_data_dir, exist_ok=True)
        raw_data_path = os.path.join(base_data_dir, "raw")
        os.makedirs(raw_data_path, exist_ok=True)
        logger_path = os.path.join(base_dir, "logs")
        processed_data_path, interim_data_path = create_directories(base_data_dir)
        train_data, test_data = load_data(interim_data_path)
        
        print(f"[INFO]: Applying {model} model...")
        if model == "tfidf":
            X_train_bow, y_train, X_test_bow, y_test = apply_tfidf(train_data, test_data, max_features)
        else:
            X_train_bow, y_train, X_test_bow, y_test = apply_bag_of_words(train_data, test_data, max_features)
            
        save_transformed_data(X_train_bow, y_train, processed_data_path, 'train_bow.csv')
        save_transformed_data(X_test_bow, y_test, processed_data_path, 'test_bow.csv')
    except Exception as e:
        print(f"[ERROR]: {e}")

if __name__ == '__main__':
    main()
