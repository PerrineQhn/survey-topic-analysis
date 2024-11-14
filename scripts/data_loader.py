"""
Data loading and preprocessing module for survey analysis.

This module handles the loading, validation, and preprocessing of survey data
from an Excel file. It also provides a method to clean text data by removing 
special characters, punctuation, and stopwords.

Typical usage example:
    loader = DataLoader()
    df = loader.load_data('data/survey_data.xlsx')
    if loader.valid_file(df):
        df = loader.preprocess_data(df)
        for col in df.columns:
            df[col] = df[col].apply(loader.clean_text)

"""

import os
from string import punctuation
import pandas as pd
import nltk

from nltk.corpus import stopwords

class DataLoader:
    def __init__(self):
        nltk.download("stopwords")
        self.stop_words = set(stopwords.words("english"))


    def load_data(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.endswith(".xlsx"):
            raise ValueError("Invalid file format. Please provide an Excel file.")
        return pd.read_excel(file_path)

    def valid_file(self, df: pd.DataFrame) -> bool:
        return "Respondent ID" in df.columns

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna().drop_duplicates()

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str) or len(text) < 1:
            return None
        text = text.lower()
        text = "".join([char for char in text if char not in punctuation or char == "-"])
        words = [word for word in text.split() if word not in self.stop_words]
        return " ".join(words) if len(words) > 1 else None


def main():
    """Main function to load and preprocess survey data."""
    try:
        path = os.path.join("data", "NLP_LLM_survey_example_1.xlsx")
        print("Starting data loading process")

        loader = DataLoader()
        df = loader.load_data(path)
        if not loader.valid_file(df):
            raise ValueError("Invalid file format")

        df = loader.preprocess_data(df)

        # Clean text in all columns
        for col in df.columns:
            df[col] = df[col].apply(loader.clean_text)

        print("Data processing completed successfully")
        print("Sample of processed data:")
        print(df.head())

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
