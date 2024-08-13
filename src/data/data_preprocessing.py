import os
import re
import nltk
import string
import pandas as pd
import numpy as np
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')



def create_directories(base_data_dir: str) -> tuple:
    """
    Create necessary directories for raw and processed data.

    Args:
        base_data_dir (str): Base directory for data.

    Returns:
        tuple: Paths for raw data and processed data directories.
    """
    raw_data_path = Path(os.path.join(base_data_dir, "raw"))
    interim_data_path = Path(os.path.join(base_data_dir, "interim"))
    os.makedirs(interim_data_path, exist_ok=True)
    return raw_data_path, interim_data_path

def load_data(raw_data_path: Path) -> tuple:
    """
    Load train and test data from CSV files.

    Args:
        raw_data_path (Path): Path to the raw data directory.

    Returns:
        tuple: DataFrames for train and test data.

    Raises:
        FileNotFoundError: If the data files are not found.
        pd.errors.EmptyDataError: If the data files are empty.
        pd.errors.ParserError: If there is an error parsing the CSV files.
    """
    try:
        train_data = pd.read_csv(os.path.join(raw_data_path, 'train_data.csv'))
        test_data = pd.read_csv(os.path.join(raw_data_path, 'test_data.csv'))
        return train_data, test_data
    except FileNotFoundError:
        logging.error("Data file not found.")
        raise
    except pd.errors.EmptyDataError:
        logging.error("Data file is empty.")
        raise
    except pd.errors.ParserError as exc:
        logging.error(f"Error parsing CSV file: {exc}")
        raise

def download_nltk_resources():
    """
    Download necessary NLTK resources.
    """
    try:
        nltk.download('wordnet')
        nltk.download('stopwords')
    except Exception as e:
        logging.error(f"Error downloading NLTK resources: {e}")
        raise

def lemmatization(text: str) -> str:
    """
    Lemmatize the given text.

    Args:
        text (str): Input text.

    Returns:
        str: Lemmatized text.
    """
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text: str) -> str:
    """
    Remove stop words from the given text.

    Args:
        text (str): Input text.

    Returns:
        str: Text without stop words.
    """
    stop_words = set(stopwords.words("english"))
    text = [word for word in text.split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text: str) -> str:
    """
    Remove numbers from the given text.

    Args:
        text (str): Input text.

    Returns:
        str: Text without numbers.
    """
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text: str) -> str:
    """
    Convert text to lower case.

    Args:
        text (str): Input text.

    Returns:
        str: Lower case text.
    """
    return text.lower()

def removing_punctuations(text: str) -> str:
    """
    Remove punctuations from the given text.

    Args:
        text (str): Input text.

    Returns:
        str: Text without punctuations.
    """
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()

def removing_urls(text: str) -> str:
    """
    Remove URLs from the given text.

    Args:
        text (str): Input text.

    Returns:
        str: Text without URLs.
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df: pd.DataFrame) -> None:
    """
    Remove rows with sentences having less than 3 words.

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize text in the DataFrame by applying various text processing functions.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
    df['content'] = df['content'].apply(lower_case)
    df['content'] = df['content'].apply(remove_stop_words)
    df['content'] = df['content'].apply(removing_numbers)
    df['content'] = df['content'].apply(removing_punctuations)
    df['content'] = df['content'].apply(removing_urls)
    df['content'] = df['content'].apply(lemmatization)
    return df

def normalized_sentence(sentence: str) -> str:
    """
    Normalize a single sentence by applying various text processing functions.

    Args:
        sentence (str): Input sentence.

    Returns:
        str: Normalized sentence.
    """
    sentence = lower_case(sentence)
    sentence = remove_stop_words(sentence)
    sentence = removing_numbers(sentence)
    sentence = removing_punctuations(sentence)
    sentence = removing_urls(sentence)
    sentence = lemmatization(sentence)
    return sentence

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, interim_data_path: Path) -> None:
    """
    Save processed train and test data to CSV files.

    Args:
        train_data (pd.DataFrame): Processed train data.
        test_data (pd.DataFrame): Processed test data.
        processed_data_path (Path): Path to the processed data directory.

    Raises:
        IOError: If there is an error saving the data files.
    """
    try:
        train_data.to_csv(os.path.join(interim_data_path, 'train_processed.csv'), index=False)
        test_data.to_csv(os.path.join(interim_data_path, 'test_processed.csv'), index=False)
        logging.info("Data Preprocessing Done")
        logging.info(f"Dataframe Shape: {train_data.shape}")
        logging.info(f"Dataframe Head: {train_data.head()}")
    except IOError as e:
        logging.error(f"Error saving data files: {e}")
        raise

def main():
    """
    Main function to execute data preprocessing steps.
    """
    try:
        # Get the base directory where the Python script is running
        base_dir = Path(__file__).resolve().parents[2]
        print(f"[INFO]: Base Directory:", base_dir)
        base_data_dir = os.path.join(base_dir, "data")
        raw_data_path, interim_data_path = create_directories(base_data_dir)
        train_data, test_data = load_data(raw_data_path)
        download_nltk_resources()
        train_data = normalize_text(train_data)
        test_data = normalize_text(test_data)
        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)
        save_data(train_data, test_data, interim_data_path)
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")

if __name__ == '__main__':
    main()
