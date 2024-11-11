from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

@dataclass
class TopicTaggingResults:
    """Stores the results of topic tagging"""
    tagged_df: pd.DataFrame
    topic_distribution: Dict[str, float]
    single_label_stats: Dict[str, float]

class TopicTaggingConverter:
    def __init__(self):
        pass
        
    def create_topic_columns(self, df: pd.DataFrame, topic_info: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Create columns for each topic and an 'other' column"""
        df_tagged = df.copy()
        
        # Create a column for each topic
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id != -1:  # Ignore outliers
                column_name = f"{prefix}_topic_{topic_id}"
                df_tagged[column_name] = 0
        
        # Add the "other" column
        df_tagged[f"{prefix}_topic_other"] = 0
        
        return df_tagged

    def assign_topics(
        self,
        df: pd.DataFrame,
        topics: List[int],
        probabilities: np.ndarray,
        indices: List[int],
        prefix: str,
        min_probability: float = 0.1
    ) -> pd.DataFrame:
        """Assign the most probable topic to each response"""
        df_tagged = df.copy()

        for idx, probs in enumerate(probabilities):
            original_idx = indices[idx]

            # Find the topics with a probability higher than the threshold
            assigned_topics = [
                topic_id
                for topic_id, prob in enumerate(probs)
                if prob >= min_probability
            ]

            # Mark the assigned topics
            for topic_id in assigned_topics:
                column_name = f"{prefix}_topic_{topic_id}"
                if column_name in df_tagged.columns:
                    df_tagged.at[original_idx, column_name] = 1
                # Store the probability to be used for sorting
                df_tagged.at[original_idx, f"{prefix}_probability_{topic_id}"] = probs[topic_id]

            # If no topic is assigned, mark as "other"
            if not assigned_topics:
                df_tagged.at[original_idx, f"{prefix}_topic_other"] = 1

        return df_tagged

    def create_multiple_choice_format(
        self,
        df: pd.DataFrame,
        prefix: str,
        topic_info: pd.DataFrame
    ) -> pd.DataFrame:
        """Convert the responses to a multiple-choice format with topics sorted by probability"""
        df_mcq = df.copy()
        
        # Create a column for the selected topics
        df_mcq[f"{prefix}_selected_topics"] = ""
        
        topic_columns = [col for col in df_mcq.columns if col.startswith(f"{prefix}_topic_")]
        
        for idx in df_mcq.index:
            selected_topics = []
            
            # Find all the assigned topics
            for col in topic_columns:
                if df_mcq.at[idx, col]:
                    topic_id = col.split('_')[-1]
                    if topic_id == 'other':
                        selected_topics.append(("Other", 0.0))  # Probability 0 for "Other"
                    else:
                        # Find the name/label of the topic and its probability
                        topic_info_row = topic_info[topic_info['Topic'] == int(topic_id)]
                    
                        if not topic_info_row.empty:
                            topic_name = topic_info_row.iloc[0].get('Name', f'Topic_{topic_id}')
                            prob = df_mcq.at[idx, f"{prefix}_probability_{topic_id}"]
                            selected_topics.append((f"{topic_name} ({prob:.2f})", prob))
            
            # Sort the topics by probability in descending order
            selected_topics.sort(key=lambda x: x[1], reverse=True)
            
            # Join the topic names (without the probabilities used for sorting)
            df_mcq.at[idx, f"{prefix}_selected_topics"] = ", ".join(topic[0] for topic in selected_topics) if selected_topics else "No topic"
        
        return df_mcq

    def calculate_statistics(self, df: pd.DataFrame, prefix: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate statistics on the distribution of topics"""
        topic_columns = [col for col in df.columns if col.startswith(f"{prefix}_topic_")]
        
        # Topic distribution
        topic_distribution = {}
        for col in topic_columns:
            topic_name = col.replace(f"{prefix}_topic_", "")
            topic_distribution[topic_name] = (df[col].sum() / len(df)) * 100
            
        # Statistics on single labeling
        responses_with_topics = df[topic_columns].sum(axis=1)
        single_label_stats = {
            'single_topic_ratio': (responses_with_topics == 1).mean() * 100,
            'no_topic_ratio': (responses_with_topics == 0).mean() * 100,
            'accuracy': np.mean([row.max() for _, row in df[topic_columns].iterrows()])
        }
        
        return topic_distribution, single_label_stats

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
        """Complete process of tagging and converting to multiple-choice format"""
        df_processed = self.create_topic_columns(df, topic_info, column_prefix)
        df_processed = self.assign_topics(df_processed, topics, probabilities, indices, column_prefix, min_probability=min_probability)
        df_processed = self.create_multiple_choice_format(df_processed, column_prefix, topic_info)
        
        topic_dist, single_label_stats = self.calculate_statistics(df_processed, column_prefix)
        
        return TopicTaggingResults(
            tagged_df=df_processed,
            topic_distribution=topic_dist,
            single_label_stats=single_label_stats
        )