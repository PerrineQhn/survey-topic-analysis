from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


@dataclass
class TopicTagResults:
    """Store results of topic tagging"""
    assignments: Dict[int, List[Tuple[int, float]]]  # response_idx -> [(topic_id, confidence)]
    quality_scores: Dict[int, float]  # response_idx -> quality_score
    multi_label_matrix: pd.DataFrame  # Multi-label format of assignments


class TopicTagger:
    def __init__(self, topic_model, min_confidence: float = 0.3, max_topics_per_response: int = 3):
        """ Initialize the TopicTagger with a trained topic model
        
        Args:
            topic_model: TopicModel
            min_confidence: Minimum confidence score for valid topics
            max_topics_per_response: Maximum number of topics to assign per response    
        """
        self.topic_model = topic_model
        self.min_confidence = min_confidence
        self.max_topics_per_response = max_topics_per_response
        self.user_guidance: Dict[int, str] = {}
        self.topic_embeddings = {}  # Store topic embeddings explicitly

    def add_user_guidance(self, topic_id: int, guidance: str):
        """Add user guidance for interpreting/assigning a specific topic
        
        Args:
            topic_id: Topic ID
            guidance: User guidance text
        """
        if guidance and isinstance(guidance, str):
            self.user_guidance[topic_id] = guidance.strip()
            print(f"Added guidance for Topic {topic_id}")

    def get_user_guidance(self, topic_id: int) -> str:
        """Get user guidance for a specific topic
        
        Args:
            topic_id: Topic ID
        Returns:
            User guidance text
        """
        return self.user_guidance.get(topic_id, "")

    def _initialize_topic_embeddings(self):
        """Initialize topic embeddings from the model"""
        try:
            for topic_id in self.topic_model.get_topics():
                if topic_id != -1:  # Skip outlier topic
                    # Get topic representative docs
                    topic_docs = self.topic_model.get_representative_docs(topic_id)
                    if topic_docs:
                        # Use the embedding model directly instead of _extract_embeddings
                        # Check if embedding_model is a function or has embed method
                        if callable(self.topic_model.embedding_model):
                            doc_embeddings = self.topic_model.embedding_model([topic_docs[0]])
                        elif hasattr(self.topic_model.embedding_model, 'embed'):
                            doc_embeddings = self.topic_model.embedding_model.embed([topic_docs[0]])
                        else:
                            raise AttributeError("Embedding model must be callable or have embed method")

                        self.topic_embeddings[topic_id] = doc_embeddings[0]
        except Exception as e:
            print(f"Error initializing topic embeddings: {str(e)}")
            raise

    def _calculate_topic_similarities(self, doc_embedding: np.ndarray) -> List[Tuple[int, float]]:
        """Calculate similarity between document and all topics
        
        Args:
            doc_embedding: Document embedding
        Returns:
            List of (topic_id, similarity) tuples
        """
        if not self.topic_embeddings:
            self._initialize_topic_embeddings()

        similarities = []
        try:
            # Always include "Other" topic with base confidence
            similarities.append((-1, 0.1))

            for topic_id, topic_embedding in self.topic_embeddings.items():
                similarity = cosine_similarity(
                    doc_embedding.reshape(1, -1),
                    topic_embedding.reshape(1, -1)
                )[0][0]

                # Consider user guidance if available
                if topic_id in self.user_guidance:
                    # Slight boost for topics with user guidance
                    similarity = min(similarity * 1.1, 1.0)  # Cap at 1.0

                similarities.append((topic_id, float(similarity)))

            return sorted(similarities, key=lambda x: x[1], reverse=True)
        except Exception as e:
            print(f"Error calculating similarities: {str(e)}")
            return [(-1, 1.0)]  # Return "Other" if calculation fails

    def tag_responses(self, texts: List[str], embeddings: Optional[np.ndarray] = None) -> TopicTagResults:
        """Tag responses with topics and calculate quality scores
        
        Args:
            texts: List of response texts
        
        Returns:
            TopicTagResults object 
        """
        if embeddings is None:
            try:
                # Use the embedding model directly
                if callable(self.topic_model.embedding_model):
                    embeddings = self.topic_model.embedding_model(texts)
                elif hasattr(self.topic_model.embedding_model, 'embed'):
                    embeddings = self.topic_model.embedding_model.embed(texts)
                else:
                    raise AttributeError("Embedding model must be callable or have embed method")
            except Exception as e:
                print(f"Error extracting embeddings: {str(e)}")
                return self._create_empty_results(len(texts))

        # Rest of the method remains the same...
        assignments = {}
        quality_scores = {}

        # Process each response
        for idx, (text, embedding) in enumerate(zip(texts, embeddings)):
            try:
                # Get topic similarities
                similarities = self._calculate_topic_similarities(embedding)

                # Filter by confidence and limit number of topics
                valid_topics = [(tid, conf) for tid, conf in similarities
                                if conf >= self.min_confidence][:self.max_topics_per_response]

                # If no valid topics, assign to "Other"
                if not valid_topics:
                    valid_topics = [(-1, 1.0)]

                assignments[idx] = valid_topics
                quality_scores[idx] = self._calculate_quality_score(text, valid_topics)
            except Exception as e:
                print(f"Error processing response {idx}: {str(e)}")
                assignments[idx] = [(-1, 1.0)]
                quality_scores[idx] = 0.0

        # Create multi-label matrix code remains the same...
        topic_ids = sorted(list(self.topic_model.get_topics().keys()) + [-1])
        multi_label_data = []

        for idx in range(len(texts)):
            row = {f'topic_{tid}': 0 for tid in topic_ids}
            row['text'] = texts[idx]

            # Add topic assignments
            for topic_id, _ in assignments[idx]:
                row[f'topic_{topic_id}'] = 1

            # Add quality score
            row['quality_score'] = quality_scores[idx]

            # Add multiple-choice format
            row['multi_choice_format'] = self.format_multi_choice(texts[idx], assignments[idx])

            multi_label_data.append(row)

        multi_label_df = pd.DataFrame(multi_label_data)
        return TopicTagResults(
            assignments=assignments,
            quality_scores=quality_scores,
            multi_label_matrix=multi_label_df
        )

    def _calculate_quality_score(self, text: str, assignments: List[Tuple[int, float]]) -> float:
        """Calculate quality score for topic assignments
        
        Args:
            text: Input text
            assignments: List of (topic_id, confidence) tuples
        Returns:
            Quality score (0.0-1.0)
        """
        try:
            if not assignments or assignments[0][0] == -1:
                return 0.0

            scores = []

            # Improve Confidence score (40%)
            # Instead of just mean, consider the best confidence score more heavily
            confidences = [score for _, score in assignments]
            confidence = max(confidences) * 0.7 + (np.mean(confidences) * 0.3)  # Weight max confidence more
            scores.append(confidence * 0.4)

            # Improve Keyword overlap score (30%)
            text_lower = text.lower()
            keyword_matches = 0
            total_keywords = 0

            for topic_id, _ in assignments:
                if topic_id != -1:
                    keywords = self.topic_model.get_topic(topic_id)
                    if keywords:
                        topic_words = [word for word, _ in keywords[:10]]
                        # Improve matching by considering partial matches and word stems
                        matches = 0
                        for word in topic_words:
                            word_lower = word.lower()
                            # Full match
                            if word_lower in text_lower:
                                matches += 1
                            # Partial match (if word length > 4)
                            elif len(word) > 4 and word_lower[:4] in text_lower:
                                matches += 0.5
                        keyword_matches += matches
                        total_keywords += len(topic_words)

            if total_keywords > 0:
                overlap_score = min(keyword_matches / total_keywords * 1.2, 1.0)  # Boost overlap score
                scores.append(overlap_score * 0.3)

            # Improve Coverage score (30%)
            # Give higher scores for having optimal number of topics (not too few, not too many)
            num_assignments = len([a for a in assignments if a[0] != -1])
            if num_assignments > 0:
                # Optimal range is 1-2 topics
                if num_assignments <= 2:
                    coverage = 1.0
                else:
                    coverage = 0.7  # Still decent score for more topics
            else:
                coverage = 0.0
            scores.append(coverage * 0.3)

            # Add bonus for having user guidance
            final_score = sum(scores)
            if any(topic_id in self.user_guidance for topic_id, _ in assignments):
                final_score *= 1.15  # 15% bonus for guided topics

            return min(final_score, 1.0)  # Cap at 1.0

        except Exception as e:
            print(f"Error calculating quality score: {str(e)}")
            return 0.0

    def format_multi_choice(self, text: str, assignments: List[Tuple[int, float]]) -> str:
        """Format response as multiple-choice with topics
        
        Args:
            text: Input text
            assignments: List of (topic_id, confidence) tuples
        Returns:
            categories: Formatted multi-choice string
        """
        try:
            topic_strings = []
            for topic_id, conf in assignments:
                if topic_id == -1:
                    topic_strings.append("Other")
                else:
                    keywords = [word for word, _ in self.topic_model.get_topic(topic_id)][:3]
                    # Include user guidance if available
                    guidance = self.get_user_guidance(topic_id)
                    if guidance:
                        topic_strings.append(f"Topic {topic_id}")
                    else:
                        topic_strings.append(f"Topic {topic_id}")

            categories = ', '.join(topic_strings)
            return categories
            
        except Exception as e:
            print(f"Error formatting multi-choice: {str(e)}")
            return f"Text: {text}\nCategories: Other"


    def _create_empty_results(self, num_texts: int) -> TopicTagResults:
        """Create empty results when processing fails
        
        Args:
            num_texts: Number of input texts
        Returns:
            TopicTagResults object with empty data
        """
        empty_assignments = {i: [(-1, 1.0)] for i in range(num_texts)}
        empty_scores = {i: 0.0 for i in range(num_texts)}
        empty_df = pd.DataFrame({'topic_-1': [1] * num_texts})
        return TopicTagResults(empty_assignments, empty_scores, empty_df)


def display_tagging_quality_scores(topic_tagger: TopicTagger, tag_results: TopicTagResults):
    """Display quality scores for topic tagging
    
    Args:
        topic_tagger: TopicTagger
        tag_results: TopicTagResults
    """
    print("\nTopic Tagging Quality Scores")
    print("=" * 50)

    # Get all quality scores from tag_results
    quality_scores = tag_results.quality_scores

    # Calculate overall statistics
    scores = list(quality_scores.values())
    avg_score = sum(scores) / len(scores) if scores else 0
    max_score = max(scores) if scores else 0
    min_score = min(scores) if scores else 0

    print(f"\nOverall Tagging Quality:")
    print(f"Average Quality Score: {avg_score:.3f}")
    print(f"Maximum Quality Score: {max_score:.3f}")
    print(f"Minimum Quality Score: {min_score:.3f}")

    # Show score breakdown
    print("\nQuality Score Components:")
    print("1. Confidence Score (40%):")
    print("   - Based on model's confidence in topic assignments")

    print("\n2. Keyword Overlap Score (30%):")
    print("   - Measures how well document keywords match assigned topics")

    print("\n3. Coverage Score (30%):")
    print("   - Evaluates how many topics were assigned vs maximum possible")

    # Show distribution of scores
    print("\nQuality Score Distribution:")
    ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]  # Removed >1.0 range
    for low, high in ranges:
        count = sum(1 for s in scores if low <= s < high)
        percentage = (count / len(scores)) * 100 if scores else 0
        print(f"{low:.1f}-{high:.1f}: {count} documents ({percentage:.1f}%)")
