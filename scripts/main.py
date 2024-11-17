from pathlib import Path
import nltk
from topic_analyzer import TopicAnalyzer, process_column
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class MyCustomEmbeddingModel:
    """Custom embedding model for text data. This class can be replaced with any other embedding model."""

    def __init__(self, base_model_name: str = "YourModel"):
        # Initialize with a base model
        self.base_model = SentenceTransformer(base_model_name)

    def embed(self, documents: List[str]) -> np.ndarray:
        # Custom preprocessing
        processed_docs = [doc.lower().strip() for doc in documents]

        # Get base embeddings
        embeddings = self.base_model.encode(processed_docs)

        # Add custom post-processing (e.g., normalization)
        normalized_embeddings = (
            embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        )

        return normalized_embeddings


def main():
    """Main function to run topic analysis"""
    print("Topic Analysis Tool")
    print("-" * 50)

    file_path = "./data/NLP_LLM_survey_example_1.xlsx"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # model = MyCustomEmbeddingModel()
    # embedding_model_type = "custom"

    model = "bert-base-uncased"
    embedding_model_type = "hugging-face"

    col = "Satisfaction (What did you like about the food/drinks?)"

    # embedding_model_type = "custom/hugging-face/sentence-transformer"
    # model = "nomDuModel"
    df = process_column(
        file_path=file_path,
        column_name=col,
        embedding_model_type=embedding_model_type,
        model=model,
        model_name=model,
    )

    while True:
        choice = input("\nWould you like to analyze another column? (yes/no): ")
        if choice.lower() != "yes":
            break
        df = process_column(file_path)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
