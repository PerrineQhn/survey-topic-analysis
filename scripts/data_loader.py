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
from typing import Optional

import pandas as pd
from nltk.corpus import stopwords


class DataLoader:
    """Handles data loading and preprocessing operations for survey analysis."""

    def __init__(self):
        """Initialize DataLoader with default settings."""
        self.data = None
        self.stop_words = set(stopwords.words("english"))

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load survey data from an Excel file.

        Args:
            file_path: Path to the Excel file. Can be string or Path object.

        Returns:
            DataFrame containing the survey data.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file is not a valid Excel file.
        """
        try:
            file_path = str(file_path)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            if not file_path.endswith(".xlsx"):
                raise ValueError("Invalid file format. Please provide an Excel file.")

            # Load data from Excel file
            self.data = pd.read_excel(file_path)
            return self.data

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def valid_file(self, df: Optional[pd.DataFrame] = None) -> bool:
        """Validate the structure of the loaded DataFrame.

        Args:
            df: DataFrame to validate. If None, uses the internally stored data.

        Returns:
            True if the file structure is valid, False otherwise
        """
        try:
            df = df if df is not None else self.data
            if df is None:
                return False
            is_valid = "Respondent ID" in df.columns
            return is_valid
        except Exception as e:
            print(f"Error validating file: {str(e)}")
            return False

    def preprocess_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Preprocess the survey data by handling missing values and duplicates.

        Args:
            df: Input DataFrame to preprocess. If None, uses the internally stored data.

        Returns:
            Preprocessed DataFrame with missing values and duplicates removed
        """
        df = df if df is not None else self.data
        if df is None:
            raise ValueError("No data available for preprocessing")

        # Remove rows with missing values and duplicates
        df = df.dropna()
        df = df.drop_duplicates()
        self.data = df
        return df

    def clean_text(self, text: str) -> Optional[str]:
        """Clean and normalize text data from survey responses.

        Args:
            text: Input text to clean

        Returns:
            Cleaned text string or None if text is invalid/too short
        """
        if not isinstance(text, str):
            return None

        if len(text) < 1:
            return None

        try:
            # Clean and normalize text
            text = text.lower()
            text = "".join(
                [char for char in text if char not in punctuation or char == "-"]
            )

            # Remove stopwords
            words = [word for word in text.split() if word not in self.stop_words]

            # Check if we have more than one word after cleaning
            if len(words) <= 1:
                return None

            return " ".join(words)
        except Exception as e:
            print(f"Error cleaning text: {str(e)}")
            return None


def main():
    """Main function to load and preprocess survey data."""
    try:
        # Example usage
        path = os.path.join("data", "NLP_LLM_survey_example_1.xlsx")
        print("Starting data loading process")

        # Load and validate data
        df = DataLoader.load_data(path)
        if not DataLoader.valid_file(df):
            raise ValueError("Invalid file format")

        # Preprocess data
        df = DataLoader.preprocess_data(df)

        # Clean text in all columns
        for col in df.columns:
            df[col] = df[col].apply(DataLoader.clean_text)

        print("Data processing completed successfully")
        print("Sample of processed data:")
        print(df.head())

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

    print(df.head())


if __name__ == "__main__":
    main()
