from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import hdbscan
import umap
from bertopic import BERTopic
from embedding_model import get_embedding_model
from data_loader import DataLoader
import os
from feedback_manager import TopicFeedbackManager
from topic_tagger import TopicTagger, display_tagging_quality_scores
from topic_guidance import TopicGuidanceManager
import numpy as np
from scipy.spatial.distance import pdist, squareform


@dataclass
class TopicExtractionResults:
    """Container for topic extraction results
    
    Attributes:
        topics (List[int]): List of topic assignments
        topic_info (pd.DataFrame): DataFrame containing topic information
        topic_labels (Dict[int, str]): Dictionary mapping topic IDs to descriptive labels
        embeddings (np.ndarray): Array of document embeddings
        topic_embeddings (Dict[int, np.ndarray]): Dictionary mapping topic IDs to mean embeddings
        document_info (pd.DataFrame): DataFrame containing document information
        probabilities (np.ndarray): Array of topic assignment probabilities
    """

    topics: List[int]
    topic_info: pd.DataFrame
    topic_labels: Dict[int, str]
    embeddings: np.ndarray
    topic_embeddings: Dict[int, np.ndarray]
    document_info: pd.DataFrame
    probabilities: np.ndarray


class TopicAnalyzer:
    def __init__(
        self,
        min_topic_size: int = 3,
        embedding_model_type: str = "sentence-transformer",
        **kwargs,
    ):
        """Initialize topic analyzer with BERTopic and embedding model

        Args:
            min_topic_size (int): Minimum number of documents for a valid topic
            embedding_model_type (str): Type of embedding model to use
            **kwargs: Additional keyword arguments for the embedding model
        """
        # Store min_topic_size as instance variable
        self.min_topic_size = min_topic_size

        # Initialize the embedding model
        self.embedding_model = get_embedding_model(embedding_model_type, **kwargs)

        # Initialize UMAP
        self.umap_model = umap.UMAP(
            n_neighbors=5,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            low_memory=False,
        )

        # Initialize HDBSCAN
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_topic_size,
            min_samples=1,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        # Initialize BERTopic
        self.topic_model = BERTopic(
            embedding_model=self._embedding_function,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            min_topic_size=self.min_topic_size,
            calculate_probabilities=True,
            verbose=True,
        )

        self.data_loader = DataLoader()

    def _embedding_function(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using the embedding model"""
        return self.embedding_model.embed(texts)

    def _is_text_column(self, series: pd.Series) -> bool:
        """
        Check if a column contains analyzable text data.

        Args:
            series: pandas Series to check

        Returns:
            bool: True if the column contains enough text data for analysis
        """
        # Drop empty values
        values = series.dropna()
        if len(values) == 0:
            return False

        # Convert all values to strings and check content
        text_values = values.astype(str)

        # Filter out values that are too short or just numbers
        meaningful_text = text_values[
            text_values.str.len()
            > 5  # More than 5 characters
            & ~text_values.str.match(r"^\d+$")  # Not just numbers
        ]

        # Need at least 5 meaningful text entries
        return len(meaningful_text) >= 5

    def _find_text_columns(self, df: pd.DataFrame) -> List[str]:
        """Find columns containing text data suitable for analysis
        
        Args:
            df (pd.DataFrame): Input DataFrame for analysis
        
        Returns:
            List[str]: List of column names containing text data
        """
        text_columns = []

        print("\nAnalyzing columns for text content...")
        for column in df.columns:
            if self._is_text_column(df[column]):
                text_columns.append(column)
                # Display a sample from the column
                sample = (
                    df[column].dropna().iloc[0]
                    if not df[column].dropna().empty
                    else "No sample available"
                )
                print(f"\nFound text column: {column}")
                print(
                    f"Sample content: {str(sample)[:100]}..."
                    if len(str(sample)) > 100
                    else f"Sample content: {sample}"
                )

        return text_columns

    def extract_topics(self, texts: List[str]) -> TopicExtractionResults:
        """Extract topics from input texts using BERTopic
        
        Args:
            texts (List[str]): List of input texts for topic extraction
            
        Returns:
            TopicExtractionResults: Object containing topic extraction results
        """
        if not texts:
            raise ValueError("No input texts provided for analysis")

        # Generate embeddings and extract topics
        embeddings = self._embedding_function(texts)
        topics, probs = self.topic_model.fit_transform(texts, embeddings)
        self.topic_model.embedding_model = self._embedding_function

        # Get topic information
        topic_info = self.topic_model.get_topic_info()
        document_info = self.topic_model.get_document_info(texts)

        # Generate topic labels
        topic_labels = self._generate_topic_labels(topic_info)

        # Calculate topic embeddings
        topic_embeddings = self._calculate_topic_embeddings(topics, embeddings)

        return TopicExtractionResults(
            topics=topics,
            topic_info=topic_info,
            topic_labels=topic_labels,
            embeddings=embeddings,
            topic_embeddings=topic_embeddings,
            document_info=document_info,
            probabilities=probs,
        )

    def _generate_topic_labels(self, topic_info: pd.DataFrame) -> Dict[int, str]:
        """Generate descriptive labels for topics
        
        Args:
            topic_info (pd.DataFrame): DataFrame containing topic information
        
        Returns:
            Dict[int, str]: Dictionary mapping topic IDs to descriptive labels
        """
        labels = {}
        for _, row in topic_info.iterrows():
            if row["Topic"] != -1:  # Skip outlier topic
                top_words = [
                    word for word, _ in self.topic_model.get_topic(row["Topic"])
                ][:5]
                labels[row["Topic"]] = " | ".join(top_words)
        return labels

    def _calculate_topic_embeddings(
        self, topics: List[int], embeddings: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """Calculate mean embeddings for each topic
        
        Args:
            topics (List[int]): List of topic assignments
            embeddings (np.ndarray): Array of document embeddings
        
        Returns:
            Dict[int, np.ndarray]: Dictionary mapping topic IDs to mean embeddings
        """
        unique_topics = set(topics)
        topic_embeddings = {}

        for topic_id in unique_topics:
            if topic_id == -1:  # Skip outliers
                continue
            topic_docs = [i for i, t in enumerate(topics) if t == topic_id]
            if topic_docs:
                topic_embeddings[topic_id] = np.mean(embeddings[topic_docs], axis=0)

        return topic_embeddings

    def get_topic_keywords(self, topic_id: int, top_n: int = 10) -> List[str]:
        """Get top keywords for a specific topic"""
        if topic_id in self.topic_model.get_topics():
            words = self.topic_model.get_topic(topic_id)
            return [word for word, _ in words[:top_n]]
        return []


def calculate_topic_quality_scores(topic_model, results):
    """
    Calculate quality scores for extracted topics using available metrics:
    - Coherence: Based on top words co-occurrence
    - Distinctiveness: How different each topic is from others
    - Coverage: Proportion of documents assigned to meaningful topics
    - Confidence: Average probability of topic assignments

    Args:
        topic_model: Trained BERTopic model
        results: TopicExtractionResults object

    Returns:
        dict: Dictionary containing quality scores
    """

    # Calculate coverage (proportion of docs not in outlier topic -1)
    total_docs = len(results.topics)
    meaningful_docs = sum(1 for topic in results.topics if topic != -1)
    coverage = meaningful_docs / total_docs if total_docs > 0 else 0

    # Calculate average topic assignment probability
    confidence = np.mean([max(prob) for prob in results.probabilities])

    # Calculate topic distinctiveness using topic embeddings
    topic_dists = []
    topic_ids = sorted(results.topic_embeddings.keys())
    if len(topic_ids) > 1:
        topic_vectors = np.array([results.topic_embeddings[tid] for tid in topic_ids])
        distances = pdist(topic_vectors, metric="cosine")
        distinctiveness = np.mean(distances) if len(distances) > 0 else 0
    else:
        distinctiveness = 0

    # Calculate approximate coherence using top keywords
    coherence_scores = []
    for topic_id in topic_ids:
        # Get top 10 words for the topic
        top_words = topic_model.get_topic(topic_id)[:10]
        if len(top_words) > 1:
            # Calculate average word probability
            word_probs = [prob for _, prob in top_words]
            coherence_scores.append(np.mean(word_probs))

    coherence = np.mean(coherence_scores) if coherence_scores else 0

    # Calculate overall quality score (weighted average)
    overall_score = (
        coherence * 0.3  # Weight for topic coherence
        + distinctiveness * 0.3  # Weight for topic distinctiveness
        + coverage * 0.2  # Weight for document coverage
        + confidence * 0.2  # Weight for assignment confidence
    )

    return {
        "coherence": coherence,
        "distinctiveness": distinctiveness,
        "coverage": coverage,
        "confidence": confidence,
        "overall_score": overall_score,
    }


def display_topic_quality_metrics(topic_model, results: TopicExtractionResults):
    """Display topic quality metrics for the extracted topics
    
    Args:
        topic_model: Trained BERTopic model
        results: TopicExtractionResults object
    """
    print("\nTopic Quality Evaluation")
    print("=" * 50)

    quality_scores = calculate_topic_quality_scores(topic_model, results)

    print("\nTopic Model Metrics:")
    print(f"1. Topic Coherence: {quality_scores['coherence']:.3f}")
    print("   - Measures how well topic words co-occur")

    print(f"\n2. Topic Distinctiveness: {quality_scores['distinctiveness']:.3f}")
    print("   - Measures how different topics are from each other")

    print(f"\n3. Topic Coverage: {quality_scores['coverage']:.3f}")
    print("   - Measures proportion of documents assigned to meaningful topics")

    print(f"\nOverall Topic Quality Score: {quality_scores['overall_score']:.3f}")


def deduplicate_keywords(keywords: List[str]) -> List[str]:
    """Remove duplicate keywords while preserving order
    
    Args:
        keywords (List[str]): List of keywords
    
    Returns:
        List[str]: List of deduplicated keywords
    """
    seen = set()
    return [x for x in keywords if not (x.lower() in seen or seen.add(x.lower()))]


def process_column(
    file_path: str,
    column_name: str = None,
    embedding_model_type: str = "sentence-transformer",
    min_topic_size: int = 3,
    **kwargs,
) -> pd.DataFrame:
    """Process a single column for topic extraction
    
    Args:
        file_path (str): Path to the input data file
        column_name (str): Name of the column to process
        embedding_model_type (str): Type of embedding model to use
        min_topic_size (int): Minimum number of documents for a valid topic
        **kwargs: Additional keyword arguments for the embedding model
    
    Returns:
        pd.DataFrame: Processed DataFrame with topic assignments
    """
    try:
        # 1. Initialize analyzer and load data
        analyzer = TopicAnalyzer(
            min_topic_size=min_topic_size,
            embedding_model_type=embedding_model_type,
            **kwargs,
        )
        print("\nLoading data file...")
        df = analyzer.data_loader.load_data(file_path)

        # Clean the dataframe by removing completely empty rows
        df = df.dropna(how="all")

        # Reset index to ensure continuous indices
        df = df.reset_index(drop=True)

        df = analyzer.data_loader.preprocess_data(df)

        # 2. Find and validate text columns
        text_columns = analyzer._find_text_columns(df)
        if not text_columns:
            print("\nNo suitable text columns found for analysis.")
            print(
                "Columns should contain text data with at least 5 meaningful entries."
            )
            return df

        # 3. Handle column selection
        if column_name is None:
            print("\nAvailable text columns:")
            for idx, col in enumerate(text_columns, 1):
                print(f"{idx}. {col}")

            while True:
                choice = input("\nChoose a column number (or 'q' to quit): ")
                if choice.lower() == "q":
                    return df
                try:
                    column_name = text_columns[int(choice) - 1]
                    break
                except (ValueError, IndexError):
                    print("Invalid selection. Please try again.")
        elif column_name not in text_columns:
            print(f"\nError: Column '{column_name}' is not suitable for text analysis.")
            return df

        # Load existing results if available and remove previous analysis for this column
        output_dir = "output"
        output_path = os.path.join(output_dir, "topic_analysis_results.xlsx")
        if os.path.exists(output_path):
            try:
                existing_df = pd.read_excel(output_path)
                # Remove any existing column-specific results
                cols_to_drop = [
                    col
                    for col in existing_df.columns
                    if (
                        col.startswith(f"{column_name}_Topic")
                        or col == f"{column_name}_multi_choice_format"
                    )
                ]
                if cols_to_drop:
                    print(f"\nRemoving previous analysis results for {column_name}...")
                    existing_df.drop(columns=cols_to_drop, inplace=True)
                df = existing_df
            except Exception as e:
                print(f"Error loading existing results: {str(e)}")

        # 4. Process texts and create valid indices
        print(f"\nProcessing column: {column_name}")

        # Get the actual data range by finding the last non-empty row in the column
        last_valid_index = df[column_name].last_valid_index()
        if last_valid_index is None:
            print(f"\nNo valid data found in column '{column_name}'.")
            return df

        # Only process up to the last valid index
        df_subset = df.loc[:last_valid_index]

        cleaned_texts = {}
        valid_indices = []
        for idx, text in df_subset[column_name].items():
            if pd.notna(text):
                cleaned = analyzer.data_loader.clean_text(str(text))
                if cleaned and len(cleaned.strip()) > 5:
                    cleaned_texts[idx] = cleaned
                    valid_indices.append(idx)

        if not cleaned_texts:
            print(
                f"\nNo valid text data found in column '{column_name}' after cleaning."
            )
            return df

        texts = [cleaned_texts[idx] for idx in valid_indices]
        print(f"Number of documents to analyze: {len(texts)}")

        # 5. Extract topics and initialize managers
        results = analyzer.extract_topics(texts)
        topic_tagger = TopicTagger(analyzer.topic_model)
        guidance_manager = TopicGuidanceManager(analyzer.topic_model, topic_tagger)

        # 6. Display initial topics and quality metrics
        display_topic_quality_metrics(analyzer.topic_model, results)
        print("\nExtracted Topics:")
        for topic_id in sorted(set(results.topics)):
            if topic_id != -1:  # Skip outlier topic
                keywords = analyzer.get_topic_keywords(topic_id)
                print(f"\nTopic {topic_id}:")
                print(f"Keywords: {', '.join(keywords)}")
                doc_count = len([t for t in results.topics if t == topic_id])
                print(
                    f"Size: {doc_count} ({(doc_count / len(texts) * 100):.1f}% of valid documents)"
                )

        # 7. Handle feedback if requested
        feedback_manager = None
        print("\nWould you like to provide feedback to improve the topics?")
        if (
            input(
                "Enter 'yes' to provide feedback, any other key to continue: "
            ).lower()
            == "yes"
        ):
            feedback_manager = TopicFeedbackManager(analyzer.topic_model)
            feedback = feedback_manager.get_feedback(results.topics)

            if (
                feedback.merge_suggestions
                or feedback.additional_keywords
                or feedback.irrelevant_keywords
            ):
                print("\nTopic Changes Summary:")

                # Show merges
                if feedback.merge_suggestions:
                    print("\nMerged Topics:")
                    for source, target in feedback.merge_suggestions:
                        print(f"Topics {source} and {target} were merged")

                # Show keyword additions
                if feedback.additional_keywords:
                    print("\nKeywords Added:")
                    for topic_id, keywords in feedback.additional_keywords.items():
                        print(f"Topic {topic_id}: Added {', '.join(keywords)}")

                # Show keyword removals
                if feedback.irrelevant_keywords:
                    print("\nKeywords Removed:")
                    for topic_id, keywords in feedback.irrelevant_keywords.items():
                        print(f"Topic {topic_id}: Removed {', '.join(keywords)}")

                # Apply feedback
                print("\nApplying feedback to refine topics...")
                new_topics, new_probs = feedback_manager.apply_feedback(
                    texts, results.topics
                )
                feedback_manager.summarize_changes(results.topics, new_topics)
                results.topics = new_topics
                results.probabilities = new_probs

        # 8. Get topic states
        active_topics = set()
        merged_topics = {}
        removed_keywords = {}
        added_keywords = {}

        if feedback_manager:
            if hasattr(feedback_manager, "active_topics"):
                active_topics = feedback_manager.active_topics
            if hasattr(feedback_manager, "merged_topics"):
                merged_topics = feedback_manager.merged_topics
            if hasattr(feedback_manager, "feedback"):
                removed_keywords = feedback_manager.feedback.irrelevant_keywords
                added_keywords = feedback_manager.feedback.additional_keywords

        if not active_topics:
            active_topics = set(
                tid for tid in topic_tagger.topic_model.get_topics().keys() if tid != -1
            )

        # 9. Collect guidance
        print("\nProviding guidance for topics...")
        guidance_manager.collect_guidance_after_feedback(feedback_manager)

        # 10. Get topic assignments
        tag_results = topic_tagger.tag_responses(texts, results.embeddings)

        # Initialize topic columns with column prefix
        for topic_id in sorted(active_topics):
            if topic_id not in merged_topics:  # Skip merged topics
                df[f"{column_name}_Topic {topic_id}"] = 0

        # Initialize multi_choice_format with column prefix
        df[f"{column_name}_multi_choice_format"] = ""

        # Track assignments only for active topics
        assignment_counts = {
            tid: 0 for tid in active_topics if tid not in merged_topics
        }

        # 11. Process assignments with multiple topics allowed
        # Adjust confidence threshold to be more lenient
        min_confidence = 0.15  # Lower threshold for assignments
        min_similarity = 0.25  # Minimum similarity score for topic assignment

        # Process assignments with adjusted thresholds
        for idx, text in enumerate(texts):
            original_idx = valid_indices[idx]
            assignments = tag_results.assignments[idx]
            text_lower = text.lower()

            # Score and filter assignments
            valid_assignments = []
            for topic_id, base_confidence in assignments:
                if topic_id not in active_topics or topic_id in merged_topics:
                    continue

                score = base_confidence
                is_valid = True

                # Enhanced keyword matching
                topic_keywords = [
                    word.lower()
                    for word, _ in topic_tagger.topic_model.get_topic(topic_id)
                ]
                text_words = set(text_lower.split())

                # Calculate keyword overlap
                keyword_matches = sum(
                    1 for word in topic_keywords if word in text_lower
                )
                partial_matches = sum(
                    1
                    for kw in topic_keywords
                    if any(w in text_lower for w in kw.split())
                )

                # Adjust score based on keyword matches
                if keyword_matches > 0:
                    score *= 1 + 0.1 * keyword_matches  # Boost for exact matches
                if partial_matches > keyword_matches:
                    score *= 1 + 0.05 * (
                        partial_matches - keyword_matches
                    )  # Smaller boost for partial matches

                # Handle removed keywords
                if topic_id in removed_keywords:
                    if any(
                        kw.lower() in text_lower for kw in removed_keywords[topic_id]
                    ):
                        is_valid = False

                # Handle added keywords with more weight
                if topic_id in added_keywords:
                    matches = sum(
                        1 for kw in added_keywords[topic_id] if kw.lower() in text_lower
                    )
                    if matches > 0:
                        boost = min(
                            1.75, 1 + 0.15 * matches
                        )  # Increased boost for added keywords
                        score *= boost

                # Consider guidance if available
                guidance = topic_tagger.get_user_guidance(topic_id)
                if guidance:
                    guidance_words = set(
                        word.lower() for word in guidance.split() if len(word) > 3
                    )
                    guidance_matches = sum(
                        1 for word in guidance_words if word in text_lower
                    )
                    if guidance_matches > 0:
                        score *= 1 + 0.1 * guidance_matches

                if is_valid and score >= min_confidence:
                    score = min(1.0, score)
                    valid_assignments.append((topic_id, score))

            # Assign to all valid topics with relative scoring
            if valid_assignments:
                # Sort by score and get top assignments
                valid_assignments.sort(key=lambda x: x[1], reverse=True)
                max_score = valid_assignments[0][1]

                topic_labels = []
                assigned_topics = []

                # Assign topics that are close to the max score
                for topic_id, score in valid_assignments:
                    if score >= max_score * 0.7:  # More lenient relative threshold
                        df.loc[original_idx, f"{column_name}_Topic {topic_id}"] = 1
                        assignment_counts[topic_id] = (
                            assignment_counts.get(topic_id, 0) + 1
                        )
                        topic_labels.append(f"Topic {topic_id}")
                        assigned_topics.append(topic_id)

                if topic_labels:
                    df.loc[original_idx, f"{column_name}_multi_choice_format"] = (
                        ", ".join(topic_labels)
                    )
                else:
                    df.loc[original_idx, f"{column_name}_multi_choice_format"] = "Other"
            else:
                # Before assigning "Other", try one more time with even lower threshold
                fallback_assignments = []
                for topic_id, base_confidence in assignments:
                    if topic_id not in active_topics or topic_id in merged_topics:
                        continue
                    if (
                        base_confidence >= min_confidence * 0.7
                    ):  # Try with 70% of min_confidence
                        fallback_assignments.append((topic_id, base_confidence))

                if fallback_assignments:
                    topic_labels = []
                    for topic_id, _ in fallback_assignments:
                        df.loc[original_idx, f"{column_name}_Topic {topic_id}"] = 1
                        assignment_counts[topic_id] = (
                            assignment_counts.get(topic_id, 0) + 1
                        )
                        topic_labels.append(f"Topic {topic_id}")
                    df.loc[original_idx, f"{column_name}_multi_choice_format"] = (
                        ", ".join(topic_labels)
                    )
                else:
                    df.loc[original_idx, f"{column_name}_multi_choice_format"] = "Other"

        # 12. Display results
        display_tagging_quality_scores(topic_tagger, tag_results)

        total_responses = len(texts)
        print("\nTopic Assignment Summary")
        print("=" * 50)
        print(f"\nTotal valid responses analyzed: {total_responses}")

        for topic_id in sorted(active_topics):
            if topic_id in merged_topics:
                continue

            assigned_count = assignment_counts.get(topic_id, 0)
            if assigned_count > 0:
                print(f"\nTopic {topic_id}:")
                print("-" * 30)

                # Get all keywords without limit
                base_keywords = [
                    word for word, _ in topic_tagger.topic_model.get_topic(topic_id)
                ]

                # Apply keyword modifications
                if topic_id in removed_keywords:
                    base_keywords = [
                        w for w in base_keywords if w not in removed_keywords[topic_id]
                    ]

                final_keywords = base_keywords.copy()
                if topic_id in added_keywords:
                    final_keywords.extend(added_keywords[topic_id])

                # Deduplicate keywords
                final_keywords = deduplicate_keywords(final_keywords)
                # Show all keywords
                print(f"Keywords: {', '.join(final_keywords)}")

                guidance = topic_tagger.get_user_guidance(topic_id)
                if guidance:
                    print(f"Guidance: {guidance}")

                percentage = assigned_count / total_responses * 100
                print(f"\nAssignments:")
                print(f"- {assigned_count} responses ({percentage:.1f}%)")

                print("\nExample Responses:")
                examples = []
                response_set = set()

                # First try to find examples with added keywords
                if topic_id in added_keywords:
                    for orig_idx, text in zip(valid_indices, texts):
                        if df.loc[orig_idx, f"{column_name}_Topic {topic_id}"] == 1:
                            text_lower = text.lower()
                            if any(
                                kw.lower() in text_lower
                                for kw in added_keywords[topic_id]
                            ):
                                if text not in response_set:
                                    examples.append(text)
                                    response_set.add(text)
                                    if len(examples) >= 3:
                                        break

                # Add more examples if needed
                if len(examples) < 3:
                    for orig_idx, text in zip(valid_indices, texts):
                        if (
                            df.loc[orig_idx, f"{column_name}_Topic {topic_id}"] == 1
                            and text not in response_set
                        ):
                            text_lower = text.lower()

                            # Skip if contains removed keywords
                            if topic_id in removed_keywords and any(
                                kw.lower() in text_lower
                                for kw in removed_keywords[topic_id]
                            ):
                                continue

                            examples.append(text)
                            response_set.add(text)
                            if len(examples) >= 3:
                                break

                for i, example in enumerate(examples, 1):
                    topic_assignments = [
                        str(t)
                        for t in sorted(active_topics)
                        if t not in merged_topics
                        and df.loc[
                            valid_indices[texts.index(example)],
                            f"{column_name}_Topic {t}",
                        ]
                        == 1
                    ]
                    print(f"{i}. [{', '.join(topic_assignments)}] {example[:100]}...")

        # 13. Count and show responses with no assignments
        topic_cols = [
            f"{column_name}_Topic {tid}"
            for tid in active_topics
            if tid not in merged_topics
        ]
        unassigned = sum(
            1 for idx in valid_indices if df.loc[idx, topic_cols].sum() == 0
        )

        if unassigned > 0:
            print(f"\nUnassigned Responses:")
            print(
                f"- {unassigned} responses ({(unassigned / total_responses * 100):.1f}%)"
            )

        # 14. Save results
        os.makedirs(output_dir, exist_ok=True)
        df.to_excel(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        return df

    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback

        traceback.print_exc()
        raise
