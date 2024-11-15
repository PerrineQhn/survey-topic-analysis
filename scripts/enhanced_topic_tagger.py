"""

Topic tagger utility functions for managing topics and tagging text data with topics.

This module contains the enhanced topic tagger class and related
functions for managing topics and tagging text data with topics.
It includes methods for updating topics, tagging responses, and
calculating quality metrics for the topic tagging process.

"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import feedback_manager


@dataclass
class TopicData:
    """
    Data structure for holding information about a topic.

    Attributes:
        id (int): Unique identifier of the topic.
        name (str): Name of the topic.
        keywords (List[str]): List of keywords associated with the topic.
        comments (Optional[str]): Optional comment on the topic.
        guidance (Optional[str]): Optional guidance provided by the user for the topic.
        feedback_history (Optional[List[Dict[str, Any]]]): List of feedback entries for the topic.
    """

    id: int
    name: str
    keywords: List[str]
    guidance: Optional[str] = None  # Classification instructions
    comment: List[str] = field(default_factory=list)  # Notes and observations
    feedback_history: Optional[List[Dict[str, Any]]] = None


class TopicTaggerEnhanced:
    """
    Enhanced topic tagging class that manages topic data and provides methods for tagging text data
    with relevant topics.

    Attributes:
        model (SentenceTransformer): Transformer model for encoding sentences.
        threshold (float): Similarity threshold for assigning topics to a response.
        topics (Dict[int, TopicData]): Dictionary storing topic information by ID.
        feedback_manager (feedback_manager.TopicFeedbackManager): Manager for handling feedback history.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.3):
        """
        Initializes the TopicTaggerEnhanced class with a sentence transformer model and a threshold.

        Args:
            model_name (str): Name of the sentence transformer model to use.
            threshold (float): Threshold for similarity scoring.
        """
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.topics: Dict[int, TopicData] = {}
        self.feedback_manager = feedback_manager.TopicFeedbackManager()
        self.weights = {"keyword": 0.3, "guidance": 0.7}

    def set_topics(self, topic_info: pd.DataFrame, topic_model: Any) -> None:
        """
        Initialize topics from a topic model's results.

        Args:
            topic_info (pd.DataFrame): DataFrame with topic information.
            topic_model: Model with topic information.
        """
        for _, row in topic_info.iterrows():
            topic_id = row["Topic"]
            if topic_id != -1:  # Ignore outliers
                topic_words = [word for word, _ in topic_model.get_topic(topic_id)]
                # Initialize with empty feedback history list
                self.topics[topic_id] = TopicData(
                    id=topic_id,
                    name=f"{row['Name']}",
                    keywords=topic_words,
                    feedback_history=[],  # Initialize as empty list, not None
                )

                # Log initial creation as first feedback entry
                initial_feedback = {
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "creation",
                    "changes": {"name": row["Name"], "keywords": topic_words},
                }
                self.topics[topic_id].feedback_history.append(initial_feedback)

    def update_topic(
        self, topic_id: int, updates: Dict[str, Any], feedback_type: str = "edit"
    ) -> None:
        """Update topic with improved feedback tracking

        Args:
            topic_id (int): ID of the topic to update.
            updates (Dict[str, Any]): Dictionary of updates to apply to the topic.
            feedback_type (str): Type of feedback for the update.

        Raises:
            ValueError: If the specified topic ID is not found.
        """
        if topic_id not in self.topics:
            raise ValueError(f"Topic {topic_id} not found")

        topic = self.topics[topic_id]
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        # Ensure feedback_history is initialized
        if topic.feedback_history is None:
            topic.feedback_history = []

        # Create detailed feedback entry
        feedback_entry = {
            "timestamp": timestamp,
            "type": feedback_type,
            "changes": {},
            "previous_state": {  # Store previous state for reference
                "name": topic.name,
                "keywords": topic.keywords.copy() if topic.keywords else [],
                "comment": topic.comment,
                "guidance": topic.guidance,
            },
        }

        # Process updates and track changes
        if "name" in updates:
            feedback_entry["changes"]["name"] = {
                "from": topic.name,
                "to": updates["name"],
            }
            topic.name = updates["name"]

        if "keywords" in updates:
            previous_keywords = topic.keywords.copy()
            if updates.get("action") == "add":
                new_keywords = [
                    k for k in updates["keywords"] if k not in topic.keywords
                ]
                topic.keywords.extend(new_keywords)
                feedback_entry["changes"]["keywords"] = {
                    "action": "add",
                    "added": new_keywords,
                }
            elif updates.get("action") == "remove":
                removed_keywords = [
                    k for k in updates["keywords"] if k in topic.keywords
                ]
                topic.keywords = [
                    k for k in topic.keywords if k not in updates["keywords"]
                ]
                feedback_entry["changes"]["keywords"] = {
                    "action": "remove",
                    "removed": removed_keywords,
                }
            else:
                feedback_entry["changes"]["keywords"] = {
                    "from": previous_keywords,
                    "to": updates["keywords"],
                }
                topic.keywords = updates["keywords"]

        if "comment" in updates:
            feedback_entry["changes"]["comment"] = {
                "from": topic.comment,
                "to": updates["comment"],
            }
            topic.comment = updates["comment"]

        if "guidance" in updates:
            feedback_entry["changes"]["guidance"] = {
                "from": getattr(topic, "guidance", None),
                "to": updates["guidance"],
            }
            topic.guidance = updates["guidance"]

        # Add feedback to history
        topic.feedback_history.append(feedback_entry)
        self.feedback_manager.add_topic_feedback(
            topic_id, feedback_type, feedback_entry
        )

    def get_topic_history(
        self, topic_id: Optional[int] = None
    ) -> Dict[int, List[Dict]]:
        """Get complete history for one or all topics"""
        if topic_id is not None:
            if topic_id not in self.topics:
                raise ValueError(f"Topic {topic_id} not found")
            return {topic_id: self.topics[topic_id].feedback_history}

        return {tid: topic.feedback_history for tid, topic in self.topics.items()}

    def get_topic_stats(self) -> Dict[str, Any]:
        """
        Get statistics about topics and their updates.

        Returns:
            Dict[str, Any]: Dictionary with statistics on each topic.
        """
        stats = {}
        for topic_id, topic in self.topics.items():
            stats[topic_id] = {
                "name": topic.name,
                "keyword_count": len(topic.keywords),
                "has_guidance": bool(topic.guidance),
                "feedback_count": (
                    len(topic.feedback_history) if topic.feedback_history else 0
                ),
                "last_update": (
                    max([f["timestamp"] for f in topic.feedback_history])
                    if topic.feedback_history
                    else None
                ),
            }
        return stats

    def calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate quality metrics for the topic tagging.

        Args:
            df (pd.DataFrame): Processed DataFrame with topic columns.

        Returns:
            Dict[str, float]: Dictionary with quality metrics, including:
                - topic_sizes: Distribution of documents across topics.
                - coverage_ratio: Percentage of responses assigned to defined topics. (0-100)
                - avg_confidence: Mean confidence score across all assignments. (0-1)
        """
        # topic_cols = [col for col in df.columns if col.startswith("topic_")]
        # topic_sample = df[topic_cols].head(35)
        # print("\n",topic_sample)

        topic_columns = [
            col
            for col in df.columns
            if col.startswith("topic_") and col != "topic_other"
        ]

        confidence_columns = [
            col for col in df.columns if col.startswith("confidence_")
        ]

        metrics = {
            # Topic size metrics - Shows distribution of responses across topics
            "topic_sizes": {
                col.replace("topic_", ""): df[col].sum() for col in topic_columns
            },
            
            # Coverage metric - Percentage of responses successfully categorized
            "coverage_ratio": (
                (1 - df["topic_other"].mean()) * 100
                if "topic_other" in df.columns
                else (
                    (df[topic_columns].sum(axis=1) > 0).mean() * 100
                    if topic_columns
                    else 0
                )
            ),
        }

        # Confidence metric - Average confidence of topic assignments
        if confidence_columns:
            avg_conf = df[confidence_columns].mean().mean()
            metrics["avg_confidence"] = {
                "value": round(avg_conf, 3),
                "interpretation": {
                    "score": avg_conf,
                    "level": (
                        "high"
                        if avg_conf > 0.7
                        else "medium" if avg_conf > 0.4 else "low"
                    ),
                    "scale": "0-1 (similarity score)",
                    "threshold": f"current threshold: {self.threshold}",
                },
            }

        return metrics

    def tag_response(
        self,
        text: str,
        embeddings_cache: Dict[str, np.ndarray] = None,
        debug: bool = False,  # Collect detailed scoring information for debugging
    ) -> Tuple[List[int], Dict[int, float], Optional[Dict]]:
        """
        Tag a response with relevant topics.

        Args:
            text: Text to tag
            embeddings_cache: Cache for precomputed embeddings
            debug: Whether to collect detailed scoring information (default: False)

        Returns:
            Tuple containing:
            - List of assigned topic IDs
            - Dictionary of similarity scores
            - Dictionary of debug information (only if debug=True)
        """
        if embeddings_cache and text in embeddings_cache:
            text_embedding = embeddings_cache[text]
        else:
            text_embedding = self.model.encode([text])[0]
            if embeddings_cache is not None:
                embeddings_cache[text] = text_embedding

        similarities = {}
        debug_info = {} if debug else None

        for topic_id, topic in self.topics.items():
            # Calculate keyword similarity
            keyword_text = " ".join(topic.keywords)
            keyword_embedding = self.model.encode([keyword_text])[0]
            keyword_similarity = cosine_similarity(
                [text_embedding], [keyword_embedding]
            )[0][0]

            # Calculate final similarity
            if topic.guidance:
                guidance_embedding = self.model.encode([topic.guidance])[0]
                guidance_similarity = cosine_similarity(
                    [text_embedding], [guidance_embedding]
                )[0][0]
                final_similarity = (
                    self.weights["keyword"] * keyword_similarity
                    + self.weights["guidance"] * guidance_similarity
                )
            else:
                final_similarity = keyword_similarity

            similarities[topic_id] = final_similarity

            # Only collect debug info if requested
            if debug:
                debug_info[topic_id] = {
                    "keyword_similarity": keyword_similarity,
                    "guidance_similarity": (
                        guidance_similarity if topic.guidance else None
                    ),
                    "final_similarity": final_similarity,
                    "has_guidance": bool(topic.guidance),
                }

        # Select topics based on threshold
        assigned_topics = [
            tid for tid, score in similarities.items() if score >= self.threshold
        ]

        if not assigned_topics:
            assigned_topics = [-1]
            similarities[-1] = 1.0

        return assigned_topics, similarities, debug_info

    def process_dataset(
        self,
        df: pd.DataFrame,
        text_column: str,
        batch_size: int = 32,
        include_scores: bool = False,  # Include detailed scores
    ) -> pd.DataFrame:
        """
        Process a dataset to assign topics.

        Args:
            df: Input DataFrame
            text_column: Name of the text column
            batch_size: Size of batches for processing
            include_scores: Whether to include detailed similarity scores (default: False)

        Returns:
            DataFrame with topic assignments and optionally detailed scores
        """
        df_processed = df.copy()

        # Initialize basic topic columns
        for topic_id, topic in self.topics.items():
            df_processed[f"topic_{topic.name}"] = 0
            if include_scores:
                df_processed[f"score_{topic.name}"] = 0.0

        df_processed["topic_other"] = 0

        # Process in batches
        embeddings_cache = {}

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            for idx, text in batch[text_column].items():
                if pd.isna(text):
                    continue

                # Get topic assignments and scores
                assigned_topics, similarities, _ = self.tag_response(
                    str(text),
                    embeddings_cache,
                    debug=False,  # No debug info needed for basic processing
                )

                # Update assignments
                for topic_id in assigned_topics:
                    if topic_id == -1:
                        df_processed.at[idx, "topic_other"] = 1
                    else:
                        topic_name = self.topics[topic_id].name
                        df_processed.at[idx, f"topic_{topic_name}"] = 1
                        if include_scores:
                            df_processed.at[idx, f"score_{topic_name}"] = similarities[
                                topic_id
                            ]

                # Add confidence scores
                for topic_id, score in similarities.items():
                    if topic_id != -1:
                        df_processed.at[
                            idx, f"confidence_{self.topics[topic_id].name}"
                        ] = score

        return df_processed

    def analyze_guidance_impact(
        self, df: pd.DataFrame, text_column: str, sample_size: int = 5
    ) -> None:
        """
        Analyze the impact of guidance on classification.
        This function temporarily enables debug mode for analysis.
        """
        print("\nAnalyzing Guidance Impact")
        print("------------------------")

        # Process sample texts with debug info
        sample_texts = df[text_column].sample(n=min(sample_size, len(df)))

        for text in sample_texts:
            print(f"\nText: {text}")

            # Get classification with debug info
            _, _, debug_info = self.tag_response(
                text, debug=True
            )  # Enable debug just for analysis

            # Show impact for topics with guidance
            guidance_topics = [
                (tid, topic) for tid, topic in self.topics.items() if topic.guidance
            ]
            if guidance_topics:
                print("\nGuidance Impact:")
                for topic_id, topic in guidance_topics:
                    info = debug_info[topic_id]
                    print(f"\n{topic.name}:")
                    print(f"Score with keywords only: {info['keyword_similarity']:.3f}")
                    print(f"Score with guidance:     {info['final_similarity']:.3f}")
                    impact = info["final_similarity"] - info["keyword_similarity"]
                    print(f"Impact: {impact:+.3f}")
            else:
                print("\nNo topics have guidance defined.")

            print("-" * 50)

    def add_guidance(self, topic_id: int, guidance_text: str) -> None:
        """
        Ajouter ou mettre à jour les instructions de classification pour un topic.

        Les guidances sont utilisées activement dans l'algorithme de classification
        pour déterminer si un texte appartient à ce topic.

        Args:
            topic_id: ID du topic
            guidance_text: Instructions de classification détaillées
        """
        if topic_id not in self.topics:
            raise ValueError(f"Topic {topic_id} not found")

        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        feedback_entry = {
            "timestamp": timestamp,
            "type": "guidance_update",
            "changes": {
                "guidance": {
                    "from": self.topics[topic_id].guidance,
                    "to": guidance_text,
                }
            },
        }

        self.topics[topic_id].guidance = guidance_text
        self.topics[topic_id].feedback_history.append(feedback_entry)
        print(f"\nClassification instructions added for topic {topic_id}")
        print(f"These instructions will be used for automatic classification.")

    def add_comment(self, topic_id: int, comment_text: str) -> None:
        """
        Ajouter un commentaire ou une note sur un topic.

        Les commentaires sont des notes pour référence humaine et ne sont pas
        utilisés dans l'algorithme de classification.

        Args:
            topic_id: ID du topic
            comment_text: Texte du commentaire
        """
        if topic_id not in self.topics:
            raise ValueError(f"Topic {topic_id} not found")

        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        feedback_entry = {
            "timestamp": timestamp,
            "type": "comment_added",
            "changes": {"comment": comment_text},
        }

        if not hasattr(self.topics[topic_id], "comments"):
            self.topics[topic_id].comment = []

        self.topics[topic_id].comment.append(comment_text)
        self.topics[topic_id].feedback_history.append(feedback_entry)
        print(f"\nComment added for topic {topic_id}")
        print("Note: Comments are for reference only and don't affect classification.")


def interactive_topic_editing(tagger: TopicTaggerEnhanced) -> None:
    """
    Interactive interface for topic modification and feedback.
    """
    while True:
        print("\nCurrent Topics:")
        for topic_id, topic in tagger.topics.items():
            print(f"\n{topic_id}: {topic.name}")
            print(f"  Keywords: {', '.join(topic.keywords)}")
            if topic.guidance:
                print(f"Classification instructions: {topic.guidance}")
            if hasattr(topic, "comments") and topic.comment:
                print(f"Comments ({len(topic.comment)}):")
                for i, comment in enumerate(topic.comment, 1):
                    print(f"    {i}. {comment}")

        print("\nAvailable Actions:")
        print("1. Rename a topic (rename)")
        print("2. Edit keywords (edit)")
        print("3. Add/Modify classification instructions (guidance)")
        print("4. Add a comment/note (comment)")
        print("5. View modification history (history)")
        print("6. View instructions impact (impact)")
        print("7. Quit (quit)")

        action = input("\nChoose an action (1-7 or action name): ").lower()

        try:
            if action in ["7", "quit", "q"]:
                break

            elif action in ["1", "rename"]:
                topic_id = int(input("Topic ID to rename: "))
                new_name = input("New name: ")
                tagger.update_topic(
                    topic_id, updates={"name": new_name}, feedback_type="rename"
                )
                tagger.feedback_manager.save_feedback()
                print(f"Topic {topic_id} renamed to: {new_name}")

            elif action in ["2", "edit"]:
                topic_id = int(input("Topic ID to modify: "))
                edit_type = input("Add or remove keywords? (add/remove): ").lower()

                if edit_type in ["add", "remove"]:
                    keywords = [
                        k.strip()
                        for k in input("Keywords (separated by commas): ").split(",")
                    ]
                    tagger.update_topic(
                        topic_id,
                        updates={"keywords": keywords, "action": edit_type},
                        feedback_type="keyword_edit",
                    )
                    tagger.feedback_manager.save_feedback()
                    print(f"Keywords {edit_type}ed for topic {topic_id}")

            elif action in ["3", "guidance"]:
                topic_id = int(input("Topic ID: "))
                guidance = input("Instructions for this topic: ")
                tagger.update_topic(
                    topic_id,
                    updates={"guidance": guidance},
                    feedback_type="guidance update",
                )
                tagger.feedback_manager.save_feedback()
                print(f"\nClassification instructions added for topic {topic_id}")
                print("These instructions will be used in the classification process")

            elif action in ["4", "comment"]:
                topic_id = int(input("Topic ID: "))
                comment = input("Your comment: ")
                tagger.update_topic(
                    topic_id,
                    updates={"comment": comment},
                    feedback_type="comment added",
                )
                tagger.feedback_manager.save_feedback()
                print("Comment saved")

            elif action in ["5", "history"]:
                print("\nModification history:")
                history = tagger.get_topic_history()
                for topic_id, feedback_history in history.items():
                    if feedback_history:
                        topic = tagger.topics[topic_id]
                        print(f"\nTopic {topic_id}: {topic.name}")
                        for entry in feedback_history:
                            print(f"\n{entry['timestamp']} | Type: {entry['type']}")
                            for change_type, change_details in entry["changes"].items():
                                if isinstance(change_details, dict):
                                    if (
                                        "from" in change_details
                                        and "to" in change_details
                                    ):
                                        print(f"    • {change_type}:")
                                        print(
                                            f"      - Before: {change_details['from']}"
                                        )
                                        print(f"      - After: {change_details['to']}")
                                    elif "action" in change_details:
                                        if change_details["action"] == "add":
                                            print(
                                                f"    • Keywords added: {', '.join(change_details['added'])}"
                                            )
                                        elif change_details["action"] == "remove":
                                            print(
                                                f"    • Keywords removed: {', '.join(change_details['removed'])}"
                                            )
                                else:
                                    print(f"    • {change_type}: {change_details}")
                input("\nPress Enter to continue...")

            elif action in ["6", "impact"]:
                print("\nImpact of classification instructions:")
                for topic_id, topic in tagger.topics.items():
                    print(f"\nTopic {topic_id}: {topic.name}")
                    if topic.guidance:
                        print(f"Active instructions: {topic.guidance}")
                        print(
                            f"Weight in classification: {tagger.weights['guidance']*100}%"
                        )
                    else:
                        print("No classification instructions defined")
                        print("Classification based on keywords only")
                input("\nPress Enter to continue...")

            else:
                print("Action non reconnue")

        except ValueError as e:
            print(f"Erreur de format: {e}")
        except Exception as e:
            print(f"Une erreur s'est produite: {e}")


def get_stats_summary(tagger: TopicTaggerEnhanced) -> None:
    """Display a summary of topic statistics"""
    stats = tagger.get_topic_stats()
    print("\nTopic Summary:")
    for topic_id, topic_stats in stats.items():
        print(f"\nTopic {topic_id}:")
        print(f"  Name: {topic_stats['name']}")
        print(f"  Number of keywords: {topic_stats['keyword_count']}")
        print(f"  Guidance defined: {'Yes' if topic_stats['has_guidance'] else 'No'}")
        print(f"  Number of modifications: {topic_stats['feedback_count']}")
        if topic_stats["last_update"]:
            print(f"  Last update: {topic_stats['last_update']}")


def get_confidence_level(confidence: float) -> str:
    """Returns a detailed confidence level assessment."""
    if confidence > 0.7:
        return "HIGH - Strong topic matches"
    elif confidence > 0.4:
        return "MEDIUM - Moderate topic matches"
    else:
        return "LOW - Weak topic matches"


def explain_metrics(coverage: float, confidence: float, threshold: float) -> str:
    """Creates a detailed explanation of metrics with interpretation."""
    explanation = [
        "\nMetrics Explanation:",
        "=" * 50,
        "\n1. Coverage Ratio (0-100%)",
        f"Current value: {coverage:.2f}%",
        "Interpretation:",
        f"- {coverage:.2f}% of texts were assigned to specific topics",
        f"- {100-coverage:.2f}% were classified as 'other'",
        "Good coverage is typically >80%",
        "\n2. Average Confidence (0-1)",
        f"Current value: {confidence:.3f}",
        "Interpretation:",
        f"- Similarity score threshold: {threshold}",
        "- Confidence levels:",
        "  • High: > 0.7",
        "  • Medium: 0.4 - 0.7",
        "  • Low: < 0.4",
        f"Current level: {get_confidence_level(confidence)}",
        # "\nRelationship between metrics:",
        # "- High coverage + Low confidence: Many texts are assigned but with low certainty",
        # "- Low coverage + High confidence: Few texts are assigned but with high certainty",
        # "- High coverage + High confidence: Ideal scenario",
        # "- Low coverage + Low confidence: Model might need adjustment"
    ]
    return "\n".join(explanation)


def get_metrics_summary(metrics) -> None:
    """Display a readable summary of topic statistics"""
    print("\nClassification Quality Metrics:")
    print("-" * 30)

    coverage = metrics["coverage_ratio"]
    confidence = metrics["avg_confidence"]

    # Detailed explanation
    print(
        explain_metrics(
            coverage, confidence["value"], confidence["interpretation"]["threshold"]
        )
    )

    # Topic distribution
    print("\nTopic Distribution:")
    print(metrics["topic_sizes"])
    total_documents = sum(metrics["topic_sizes"].values())
    for topic_name, count in metrics["topic_sizes"].items():
        percentage = (count / total_documents) * 100
        print(f"{topic_name}:")
        print(f"  Documents: {count}")
        print(f"  Percentage: {percentage:.2f}%")
