"""
Topic extraction pipeline using BERTopic and LLM models.

This module provides a complete pipeline for extracting topics 
from text data using BERTopic and LLM models. It includes methods
for processing data, extracting topics, and managing feedback.

Usage:
    python topic_extraction.py

"""

from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import os

import pandas as pd
import numpy as np
import hdbscan
import umap
from bertopic import BERTopic

from data_loader import DataLoader
import enhanced_topic_tagger
import topic_tagging_utils
from feedback_manager import TopicFeedbackManager
from llm_module import get_llm_model


os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class TopicExtractionResults:
    """Container for topic extraction results.

    Attributes:
        topics: List of assigned topic IDs
        topic_info: DataFrame with topic information
        topic_labels: Dictionary mapping topic IDs to labels
        embeddings: Document embeddings
        topic_embeddings: Topic centroid embeddings
        document_info: Additional document information
        probabilities: Topic assignment probabilities
    """

    topics: List[int]
    topic_info: pd.DataFrame
    topic_labels: Dict[int, str]
    embeddings: np.ndarray
    topic_embeddings: Dict[int, np.ndarray]
    document_info: pd.DataFrame
    probabilities: np.ndarray


class TopicExtractorBERTopic:
    """Topic extraction using BERTopic and LLM models.

    This class handles the complete topic extraction pipeline, including
    embedding generation, topic modeling, and topic management.

    Attributes:
        llm_model: Language model for embedding generation
        topic_model: BERTopic model instance
        feedback_manager: Manager for user feedback
        custom_topic_labels: Dictionary of custom topic labels
    """

    def __init__(
        self,
        model_type: str = "sentence-transformer",
        model_name="all-MiniLM-L6-v2",
        min_topic_size=3,
    ):
        """Initialize the topic extractor.

        Args:
            model_type: Type of LLM model to use
            model_name: Name of the specific model
            min_topic_size: Minimum number of documents per topic

        Raises:
            ValueError: If model_type is not supported
        """
        try:
            # Initialize the LLM model
            self.llm_model = get_llm_model(model_type, model_name)

            # Initialize UMAP for dimensionality reduction
            umap_model = umap.UMAP(
                n_neighbors=5,
                n_components=5,
                min_dist=0.0,
                metric="cosine",
                low_memory=False,
            )

            # Initialize HDBSCAN for clustering
            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=min_topic_size,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            )

            # Initialize BERTopic
            self.topic_model = BERTopic(
                embedding_model=self._embedding_function,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                min_topic_size=min_topic_size,
                calculate_probabilities=True,
            )

            self.feedback_manager = TopicFeedbackManager()
            self.custom_topic_labels = {}
        except Exception as e:
            raise ValueError(f"Error initializing topic extractor: {str(e)}")

    def _initialize_umap(self) -> umap.UMAP:
        """Initialize UMAP model with optimal parameters."""
        return umap.UMAP(
            n_neighbors=5,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            low_memory=False,
        )

    def _initialize_hdbscan(self, min_topic_size: int) -> hdbscan.HDBSCAN:
        """Initialize HDBSCAN model for clustering."""
        return hdbscan.HDBSCAN(
            min_cluster_size=min_topic_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

    def _embedding_function(self, texts) -> np.ndarray:
        """Generate embeddings using the LLM model.

        Args:
            texts: List of input texts

        Returns:
            Array of document embeddings
        """
        return self.llm_model.generate_embeddings(texts)

    def extract_topics(self, texts) -> TopicExtractionResults:
        """
        Extract topics from input texts.

        Args:
            texts (List[str]): List of text documents

        Returns:
            TopicExtractionResults: Container with topic extraction results

        Raises:
            ValueError: If no input texts are provided
        """
        if not texts:
            raise ValueError("No input texts provided for topic extraction.")

        try:
            # Generate embeddings and initial topics
            embeddings = self.llm_model.generate_embeddings(texts)
            topics, probabilities = self.topic_model.fit_transform(texts, embeddings)

            # Get topic information
            topic_info = self.topic_model.get_topic_info()
            document_info = self.topic_model.get_document_info(texts)

            # Generate topic labels
            topic_labels = self._generate_topic_labels(topic_info)

            # Update with any custom labels
            topic_labels.update(self.custom_topic_labels)

            # Calculate topic embeddings
            topic_embeddings = self._calculate_topic_embeddings(topics, embeddings)

            return TopicExtractionResults(
                topics=topics,
                topic_info=topic_info,
                topic_labels=topic_labels,
                embeddings=embeddings,
                topic_embeddings=topic_embeddings,
                document_info=document_info,
                probabilities=probabilities,
            )

        except Exception as e:
            raise ValueError(f"Error extracting topics: {str(e)}")

    def _generate_topic_labels(self, topic_info: pd.DataFrame) -> Dict[int, str]:
        """Generate topic labels from topic information."""
        base_labels = {
            row["Topic"]: row["Name"]
            for _, row in topic_info.iterrows()
            if row["Topic"] != -1
        }
        base_labels.update(self.custom_topic_labels)
        return base_labels

    def _calculate_topic_embeddings(self, topics, embeddings) -> Dict[int, np.ndarray]:
        """Calculate mean embeddings for each topic.

        Args:
            topics: List of topic assignments
            embeddings: Document embeddings

        Returns:
            Dictionary mapping topic IDs to their centroid embeddings
        """
        unique_topics = set(topics)
        topic_embeddings = {}

        for topic_id in unique_topics:
            if topic_id == -1:  # Skip outliers
                continue
            topic_indices = [i for i, t in enumerate(topics) if t == topic_id]
            if topic_indices:  # Only calculate if there are documents in the topic
                topic_embeddings[topic_id] = np.mean(
                    [embeddings[i] for i in topic_indices], axis=0
                )

        return topic_embeddings

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current LLM model.

        Returns:
            Dict[str, Any]: Model metadata
        """
        return self.llm_model.get_model_info()

    def get_topic_keywords(self, topic_id: int, top_n: int = 10) -> list:
        """Get the top keywords for a topic.

        Args:
            topic_id: Topic ID
            top_n: Number of top keywords to return

        Returns:
            List of top keywords for the topic
        """
        if topic_id in self.topic_model.get_topics():
            words = self.topic_model.get_topic(topic_id)
            return [word for word, _ in words[:top_n]]
        return []


def clean_and_extract_texts(
    df: pd.DataFrame, column_name: str
) -> Tuple[List[str], List[int]]:
    """Clean and extract valid texts and indices from a DataFrame column.

    Args:
        df: Input DataFrame
        column_name: Name of the column to process

    Returns:
        Tuple: List of cleaned texts and corresponding indices
    """
    # Create DataLoader instance
    loader = DataLoader()

    texts_series = df[column_name].apply(lambda x: loader.clean_text(x))
    valid_texts = texts_series.dropna()
    return valid_texts.tolist(), valid_texts.index.tolist()


def display_extracted_topics(
    topic_extractor: TopicExtractorBERTopic, topics: List[int]
):
    """Display extracted topics and their keywords.

    Args:
        topic_extractor: TopicExtractorBERTopic instance
        topics: List of assigned topic IDs

    """
    print("\nExtracted topics:")
    for topic_id in sorted(set(topics)):
        if topic_id != -1:  # Ignore outliers
            keywords = topic_extractor.get_topic_keywords(topic_id)
            print(f"Topic {topic_id}: {', '.join(keywords)}")


def process_feedback(
    tagger,
    topic_extractor: TopicExtractorBERTopic,
    topic_info: pd.DataFrame,
    topic_labels: Dict[int, str],
):
    """Manage user feedback and update topic labels.

    Args:
        tagger: Enhanced TopicTagger instance
        topic_extractor: TopicExtractorBERTopic instance
        topic_info: DataFrame with topic information
        topic_labels: Dictionary of topic labels

    """
    enhanced_topic_tagger.interactive_topic_editing(tagger)
    enhanced_topic_tagger.get_stats_summary(tagger)

    for topic_id, topic_data in tagger.topics.items():
        if topic_data.feedback_history:
            topic_extractor.custom_topic_labels[topic_id] = topic_data.name
            if topic_id in topic_extractor.topic_model.topic_labels_:
                topic_extractor.topic_model.topic_labels_[topic_id] = topic_data.name
                if hasattr(topic_extractor.topic_model, "_topics"):
                    topic_extractor.topic_model._topics[topic_id] = [
                        (word, 0.5) for word in topic_data.keywords
                    ]
                topic_info.loc[topic_info["Topic"] == topic_id, "Name"] = (
                    topic_data.name
                )
                topic_info.loc[topic_info["Topic"] == topic_id, "Keywords"] = ", ".join(
                    topic_data.keywords
                )
                topic_labels[topic_id] = topic_data.name


def save_results(tagging_results, output_path: str):
    """Save the tagging results to an Excel file.

    Args:
        tagging_results: TopicTaggingResults instance
        output_path: Output file path

    """
    tagging_results.tagged_df.to_excel(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


def topics_extraction_process(
    df: pd.DataFrame, column_name: str, model_name: str = "all-MiniLM-L6-v2", min_probability: float = 0.2
):
    """Process the complete topic extraction pipeline.

    Args:
        df: Input DataFrame
        column_name: Name of the column to process
        model_name: Name of the LLM model to use

    Returns:
        Tuple: Processed DataFrame, extracted topics, topic information, embeddings, topic embeddings, document info, topic extractor, probabilities, texts

    """
    # Create a copy of input DataFrame to preserve original data
    df_processed = df.copy()

    texts, indices = clean_and_extract_texts(df, column_name)
    print(f"\nAnalyzing column: {column_name}")

    topic_extractor = TopicExtractorBERTopic(model_name=model_name, min_topic_size=3)
    results = topic_extractor.extract_topics(texts)

    topics, topic_info, topic_labels = (
        results.topics,
        results.topic_info,
        results.topic_labels,
    )
    embeddings, topic_embeddings, document_info, probabilities = (
        results.embeddings,
        results.topic_embeddings,
        results.document_info,
        results.probabilities,
    )

    display_extracted_topics(topic_extractor, topics)

    topic_converter = topic_tagging_utils.TopicTaggingConverter()
    initial_tagging_results = topic_converter.process_dataset(
        df=df,
        topic_info=topic_info,
        topics=topics,
        probabilities=probabilities,
        indices=indices,
        column_prefix=column_name,
        min_probability=min_probability,
    )

    base_df = pd.DataFrame({column_name: pd.Series(texts, index=indices)})
    tagger = enhanced_topic_tagger.TopicTaggerEnhanced(model_name=model_name)
    tagger.set_topics(topic_info, topic_extractor.topic_model)

    # Process current column's topics
    column_results = tagger.process_dataset(
        df=base_df, text_column=column_name, include_scores=False
    )

    if input("\nWould you like to manage topics? (yes/no): ").lower() == "yes":
        process_feedback(tagger, topic_extractor, topic_info, topic_labels)
        tagging_results = topic_converter.process_dataset(
            df=df,
            topic_info=topic_info,
            topics=topics,
            probabilities=probabilities,
            indices=indices,
            column_prefix=column_name,
            min_probability=min_probability,
        )
        column_results = tagger.process_dataset(
            df=base_df.copy(), text_column=column_name
        )

        print(f"\nTopics for {column_name}:")
        for topic_id in sorted(set(topics)):
            if topic_id != -1:
                keywords = (
                    tagger.topics[topic_id].keywords
                    if topic_id in tagger.topics
                    else []
                )
                label = topic_labels.get(topic_id, f"{topic_id}")
                print(f"{label} (Topic {topic_id}): {', '.join(keywords)}")
    else:
        tagging_results = initial_tagging_results

    # Add results to processed DataFrame with unique column names
    for col in tagging_results.tagged_df.columns:
        if col != column_name:  # Don't duplicate the original column
            new_col = f"{col}"  # Create unique column name
            df_processed[new_col] = tagging_results.tagged_df[col]

    # Calculate and display metrics
    quality_metrics = tagger.calculate_quality_metrics(column_results)
    enhanced_topic_tagger.get_metrics_summary(quality_metrics)

    # Save intermediate results
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "survey_analysis_results.xlsx")
    tagging_results.tagged_df.to_excel(output_path, index=False)
    save_results(tagging_results, output_path)

    return df_processed


def main():
    """Main function to load and preprocess survey data."""
    path = os.path.join("data", "NLP_LLM_survey_example_1.xlsx")
    print("Starting data loading process")
    # Create DataLoader instance
    loader = DataLoader()

    # Load and preprocess data
    df = loader.load_data(path)
    df = loader.preprocess_data(df)
    df_with_topics = df.copy()

    model_name = (
        input(
            "Enter the name of the LLM model you want to use (default 'all-MiniLM-L6-v2'): "
        )
        or "all-MiniLM-L6-v2"
    )
    while True:
        print("\nAvailable columns for topic analysis:")
        for idx, col in enumerate(df.columns, 1):
            print(f"{idx}. {col}")

        choice = input("Choose a column or type 'exit' to finish: ")
        if choice.lower() == "exit":
            print("Analyse terminée.")
            break

        if choice not in [str(i) for i in range(1, len(df.columns) + 1)]:
            print("Invalid choice. Please select a valid column.")
            continue

        column_name = df.columns[int(choice) - 1]

        # Extraire les sujets
        df_with_topics = topics_extraction_process(
            df_with_topics, column_name, model_name=model_name, min_probability=0.2
        )


if __name__ == "__main__":
    main()
