import os
from string import punctuation
import pandas as pd
from nltk.corpus import stopwords


class DataLoader:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load survey data from an Excel file.

        Args:
            file_path (str): Path to the Excel file.

        Returns:
            pd.DataFrame: Loaded data as a pandas DataFrame.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.endswith(".xlsx"):
            raise ValueError("Invalid file format. Please provide an Excel file.")
        return pd.read_excel(file_path)

    def valid_file(self, df: pd.DataFrame) -> bool:
        """Check if the loaded file contains the expected columns.

        Args:
            df (pd.DataFrame): Loaded data as a pandas DataFrame.

        Returns:
            bool: True if the file contains the expected columns, False otherwise.
        """
        return "Respondent ID" in df.columns

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the loaded survey data.

        Args:
            df (pd.DataFrame): Loaded data as a pandas DataFrame.

        Returns:
            pd.DataFrame: Processed data with duplicates and missing values removed.
        """
        return df.dropna().drop_duplicates()

    def clean_text(self, text: str) -> str:
        """Clean and preprocess text data.

        Args:
            text (str): Input text data to clean.

        Returns:
            str: Cleaned text data with special characters, punctuation, and stopwords removed.
        """
        if not isinstance(text, str) or len(text) < 1:
            return None
        text = text.lower()
        text = "".join(
            [char for char in text if char not in punctuation or char == "-"]
        )
        words = [word for word in text.split() if word not in self.stop_words]
        return " ".join(words) if len(words) > 1 else None


def main():
    """Main function to load and preprocess survey data."""
    try:
        path = os.path.join("./data", "NLP_LLM_survey_example_1.xlsx")
        print("Starting data loading process")
        print(path)
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
