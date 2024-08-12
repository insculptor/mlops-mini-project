import os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import logging
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Load environment variables from .env file
load_dotenv()

# Get PYTHONPATH from environment variable
base_dir = Path(os.getenv('BASE_DIR'))
base_data_dir = Path(os.getenv('BASE_DATA_DIR'))
external_data_path = Path(os.getenv('EXTERNAL_DATA_DIR'))
raw_data_path = os.path.join(base_data_dir, "raw")
os.makedirs(raw_data_path, exist_ok=True)
logger_path = os.path.join(base_dir, "logs")

## Logging Configuration
logger = logging.getLogger(os.path.join(logger_path, "data_ingestion.log"))
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


def load_params(params_path: str) -> str:
    """
    Load parameters from a YAML file.

    Args:
        params_path (str): Path to the parameters YAML file.

    Returns:
        str: Test size for train-test split.

    Raises:
        FileNotFoundError: If the parameters file is not found.
        KeyError: If the required keys are not found in the parameters file.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            test_size = params['data_ingestion']['test_size']
            logger.info(f"[INFO]: Test Size: {test_size}")
            return test_size
    except FileNotFoundError:
        logger.error("[ERROR]: Parameters file not found.")
        raise
    except KeyError:
        logger.error("[ERROR]: Key 'data_ingestion' or 'test_size' not found in parameters file.")
        raise
    except yaml.YAMLError as exc:
        logger.error(f"[ERROR]: Error parsing YAML file: {exc}")
        raise

def read_data(data_path: str) -> pd.DataFrame:
    """
    Read data from a CSV file.

    Args:
        data_path (str): Path to the directory containing the data file.

    Returns:
        pd.DataFrame: DataFrame containing the data.

    Raises:
        FileNotFoundError: If the data file is not found.
        pd.errors.EmptyDataError: If the data file is empty.
        pd.errors.ParserError: If there is an error parsing the CSV file.
    """
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        logger.error("[ERROR]: Data file not found.")
        raise
    except pd.errors.EmptyDataError:
        logger.error("[ERROR]: Data file is empty.")
        raise
    except pd.errors.ParserError as exc:
        logger.error(f"[ERROR]: Error parsing CSV file: {exc}")
        raise

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the data by removing unnecessary columns and filtering rows.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Processed DataFrame.

    Raises:
        KeyError: If the required columns are not found in the DataFrame.
    """
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['neutral', 'sadness'])]
        final_df['sentiment'].replace({'neutral': 1, 'sadness': 0}, inplace=True)
        return final_df
    except KeyError as e:
        logger.error(f"[ERROR]: Column not found in DataFrame: {e}")
        raise

def save_data(df: pd.DataFrame, test_size: str, raw_data_path: str) -> bool:
    """
    Save the processed data into train and test CSV files.

    Args:
        df (pd.DataFrame): DataFrame to be split and saved.
        test_size (str): Proportion of the data to be used for testing.
        raw_data_path (str): Path to the directory where the CSV files will be saved.

    Returns:
        bool: True if the data is successfully saved, False otherwise.

    Raises:
        ValueError: If there is an error in the train-test split.
        IOError: If there is an error saving the CSV files.
    """
    try:
        train_data, test_data = train_test_split(df, test_size=float(test_size), random_state=42)
        logger.info(f"[INFO]: Dataframe Shape: {df.shape} | Train Data Shape: {train_data.shape} | Test Data Shape: {test_data.shape}")
        logger.info(f"[INFO]: Dataframe Head: {df.head()}")
        train_data.to_csv(os.path.join(raw_data_path, 'train_data.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test_data.csv'), index=False)
        return True
    except ValueError as e:
        logger.error(f"[ERROR]: Error in train-test split: {e}")
        raise
    except IOError as e:
        logger.error(f"[ERROR]: Error saving CSV files: {e}")
        raise

def main():
    """
    Main function to load parameters, read data, process data, and save the processed data.
    """
    try:
        params_path = 'params.yaml'
        test_size = load_params(params_path)
        #df = read_data(data_url='https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        df = process_data(df)
        save_data(df, test_size, raw_data_path)
    except Exception as e:
        logger.error(f"[ERROR]: {e}")

if __name__ == '__main__':
    main()
