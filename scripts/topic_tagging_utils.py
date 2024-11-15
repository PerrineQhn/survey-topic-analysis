"""
Topic Tagging Utility for NLP Applications

This script provides utility classes and functions for tagging and analyzing topics within a dataset.
It includes methods to process a dataset for topic tagging based on predefined topics, convert data
to a multiple-choice format for topics, and calculate statistics on topic distributions and labeling
accuracy. The utility is designed to support NLP models that assign topics to text data.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

@dataclass
class TopicTaggingResults:
    """
    Data structure to store the results of topic tagging.
    
    Attributes:
        tagged_df (pd.DataFrame): The DataFrame with tagged topics.
        topic_distribution (Dict[str, float]): Distribution of topics as a percentage of total documents.
        single_label_stats (Dict[str, float]): Statistics on single-topic labeling.
    """
    tagged_df: pd.DataFrame
    topic_distribution: Dict[str, float]
    single_label_stats: Dict[str, float]

class TopicTaggingConverter:
    def process_dataset(
        self,
        df: pd.DataFrame,
        topic_info: pd.DataFrame,
        topics: List[int],
        probabilities: np.ndarray,
        indices: List[int],
        column_prefix: str,
        min_probability: float = 0.1
    ) -> TopicTaggingResults:
        """
        Process a dataset to tag topics based on a probability threshold and compute statistics.
        
        Args:
            df (pd.DataFrame): Input DataFrame to process.
            topic_info (pd.DataFrame): DataFrame containing topic information with 'Topic' and 'Name' columns.
            topics (List[int]): List of topic IDs assigned to each document.
            probabilities (np.ndarray): Array of probabilities for each topic per document.
            indices (List[int]): List of indices to map documents back to the original DataFrame.
            column_prefix (str): Prefix for column names in the output DataFrame.
            min_probability (float): Minimum probability to assign a topic to a document.

        Returns:
            TopicTaggingResults: Results containing the tagged DataFrame, topic distribution, and single-label stats.
        """
        df_tagged = df.copy()
        
        # Create columns for each topic and initialize with zeros
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id != -1:
                column_name = f"{column_prefix}_topic_{topic_id}"
                df_tagged[column_name] = 0

        df_tagged[f"{column_prefix}_topic_other"] = 0
        
        # Assign topics and probabilities to each document
        for idx, probs in enumerate(probabilities):
            original_idx = indices[idx]
            assigned_topics = [topic_id for topic_id, prob in enumerate(probs) 
                             if prob >= min_probability]
            
            if assigned_topics:
                for topic_id in assigned_topics:
                    column_name = f"{column_prefix}_topic_{topic_id}"
                    if column_name in df_tagged.columns:
                        df_tagged.at[original_idx, column_name] = 1
                        df_tagged.at[original_idx, f"{column_prefix}_probability_{topic_id}"] = probs[topic_id]
            else:
                df_tagged.at[original_idx, f"{column_prefix}_topic_other"] = 1

        # Convert to multiple-choice format
        df_tagged = self.create_multiple_choice_format(df_tagged, column_prefix, topic_info)
        
        # Calculate statistics
        topic_distribution, single_label_stats = self.calculate_statistics(df_tagged, column_prefix)
        
        return TopicTaggingResults(
            tagged_df=df_tagged,
            topic_distribution=topic_distribution,
            single_label_stats=single_label_stats
        )

    def create_multiple_choice_format(
        self, 
        df: pd.DataFrame, 
        prefix: str, 
        topic_info: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert the DataFrame to a multiple-choice format where selected topics are listed per document.
        
        Args:
            df (pd.DataFrame): DataFrame containing topic assignments.
            prefix (str): Prefix used for the topic columns.
            topic_info (pd.DataFrame): DataFrame with topic metadata (including 'Topic' and 'Name').

        Returns:
            pd.DataFrame: DataFrame in a multiple-choice format with selected topics per document.
        """
        df_mcq = df.copy()
        df_mcq[f"{prefix}_selected_topics"] = ""
        topic_columns = [col for col in df_mcq.columns if col.startswith(f"{prefix}_topic_")]
        
        # Iterate through each row to list selected topics
        for idx in df_mcq.index:
            selected_topics = []
            for col in topic_columns:
                if df_mcq.at[idx, col]:
                    topic_id = col.split('_')[-1]
                    if topic_id == 'other':
                        selected_topics.append(("Other", 0.0))
                    else:
                        topic_info_row = topic_info[topic_info['Topic'] == int(topic_id)]
                        if not topic_info_row.empty:
                            topic_name = topic_info_row.iloc[0].get('Name', f'Topic_{topic_id}')
                            prob = df_mcq.at[idx, f"{prefix}_probability_{topic_id}"]
                            selected_topics.append((f"{topic_name} ({prob:.2f})", prob))
            
            # Sort topics by probability and format for display
            selected_topics.sort(key=lambda x: x[1], reverse=True)
            df_mcq.at[idx, f"{prefix}_selected_topics"] = ", ".join(topic[0] for topic in selected_topics) if selected_topics else "No topic"
        
        return df_mcq

    def calculate_statistics(
        self, 
        df: pd.DataFrame, 
        prefix: str
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate topic distribution and single-label statistics for the dataset.
        
        Args:
            df (pd.DataFrame): DataFrame containing topic information.
            prefix (str): Prefix used for the topic columns.

        Returns:
            Tuple[Dict[str, float], Dict[str, float]]: 
                - topic_distribution (Dict[str, float]): Distribution of each topic as a percentage of total documents.
                - single_label_stats (Dict[str, float]): Ratios for single-topic documents, no-topic documents, and accuracy.
        """
        topic_columns = [col for col in df.columns if col.startswith(f"{prefix}_topic_")]
        
        # Calculate the distribution of topics as a percentage of total documents
        topic_distribution = {
            topic_name: (df[col].sum() / len(df)) * 100
            for col in topic_columns
            if (topic_name := col.replace(f"{prefix}_topic_", ""))
        }
            
        # Calculate single-label statistics (single topic, no topic, and average topic accuracy)
        responses_with_topics = df[topic_columns].sum(axis=1)
        single_label_stats = {
            'single_topic_ratio': (responses_with_topics == 1).mean() * 100,
            'no_topic_ratio': (responses_with_topics == 0).mean() * 100,
            'accuracy': np.mean([row.max() for _, row in df[topic_columns].iterrows()])
        }
        
        return topic_distribution, single_label_stats