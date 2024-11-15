"""
NLP Survey Analysis Application

This script processes survey data to extract and analyze topics from text responses.
It supports interactive feedback, topic tagging, and LLM-based analysis.

Example:
    $ python scripts/main.py --file_path data/NLP_LLM_survey_example_1.xlsx --column_name "Satisfaction (What did you like about the food/drinks?)"
"""

import argparse
import os
from dataclasses import dataclass

import pandas as pd
import nltk


from data_loader import DataLoader
from topic_extraction import topics_extraction_process


@dataclass
class AnalysisResults:
    """Container for analysis results"""
    processed_df: pd.DataFrame


class SurveyAnalyzer:
    """Main class for survey analysis operations"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the SurveyAnalyzer class."""
        self.model_name = model_name
        self.results = None
    

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess survey data"""
        try:
            loader = DataLoader()
            df = loader.load_data(file_path)
            df = loader.preprocess_data(df)
            print(f"Successfully loaded data from {file_path}")
            return df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def analyze_topics(self, df: pd.DataFrame, column_name: str, min_probability: float) -> pd.DataFrame:
        """Process topics for a specific column

        Args:
            df: Input DataFrame
            column_name: Name of the column to analyze
            min_probability: Minimum probability threshold for topic assignment

        Returns:
            Processed DataFrame with topic analysis
        """
        try:
            # Process topics
            processed_df = topics_extraction_process(
                df=df, column_name=column_name, model_name=self.model_name, min_probability=min_probability
            )

            # Store results
            self.results = AnalysisResults(processed_df=processed_df)

            return processed_df

        except Exception as e:
            print(f"Error in topic analysis: {str(e)}")
            raise

    def save_results(self, output_dir: str = "output") -> None:
        """Save analysis results to files

        Args:
            output_dir: Directory to save results
        """
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save main results
            output_path = os.path.join(output_dir, "survey_analysis_results.xlsx")
            self.results.processed_df.to_excel(output_path, index=False)

        except Exception as e:
            print(f"Error saving results: {str(e)}")
            raise


def main(args: argparse.Namespace) -> None:
    """Main execution function"""
    try:
        
        # Initialize analyzer
        analyzer = SurveyAnalyzer(model_name=args.model_name)

        # Load data
        df = analyzer.load_data(args.file_path)

        # Process topics
        processed_df = analyzer.analyze_topics(df, args.column_name, 0.2)

        print(processed_df.head())
        print("Analysis completed successfully!")

        # Save results
        analyzer.save_results("output")

    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="NLP Survey Analysis Application",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--file_path", type=str, required=True, help="Path to survey data file"
    )

    parser.add_argument(
        "--column_name", type=str, required=True, help="Name of column to analyze"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Name of LLM model to use",
    )

    args = parser.parse_args()
    main(args)
