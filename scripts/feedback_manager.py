from typing import List, Dict, Set, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class TopicFeedback:
    """Store user feedback about topics
    
    Attributes:
        split_suggestions: List of suggested topics to split
        merge_suggestions: List of (source, target) tuples for merging topics
        rename_suggestions: Dictionary of topic ID to new name mappings
        irrelevant_keywords: Dictionary of topic ID to set of irrelevant keywords
        additional_keywords: Dictionary of topic ID to set of additional keywords
    """
    merge_suggestions: List[tuple] = field(default_factory=list)
    rename_suggestions: Dict[int, str] = field(default_factory=dict)
    irrelevant_keywords: Dict[int, Set[str]] = field(default_factory=dict)
    additional_keywords: Dict[int, Set[str]] = field(default_factory=dict)


class TopicFeedbackManager:
    def __init__(self, topic_model):
        self.topic_model = topic_model
        self.feedback = TopicFeedback()
        self.merged_topics = {}  # source -> target mapping
        self.active_topics = set()  # Set of currently active topics
        self.topic_words = {}  # Store custom topic words

    def get_topic_words(self, topic_id: int) -> List[Tuple[str, float]]:
        """Get topic words, checking both custom storage and model
        
        Args:
            topic_id (int): The topic ID to retrieve words for
        
        Returns:
            List[Tuple[str, float]]: List of (word, weight) tuples
        """
        try:
            # First check custom storage
            if topic_id in self.topic_words:
                return self.topic_words[topic_id]

            # Then check the topic model
            if hasattr(self.topic_model, 'get_topic'):
                return self.topic_model.get_topic(topic_id)

            # If no topic words found
            print(f"Warning: No words found for topic {topic_id}")
            return []

        except Exception as e:
            print(f"Error getting topic words: {str(e)}")
            return []

    def _validate_topic_id(self, topic_id: int, available_topics: list) -> bool:
        """Validate if a topic ID is valid and available
        
        Args:
            topic_id (int): The topic ID to validate
            available_topics (list): List of available topic IDs
        
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            if not isinstance(topic_id, int):
                print(f"Error: Topic ID must be an integer, got {type(topic_id)}")
                return False
            if topic_id not in available_topics:
                print(f"Error: Topic ID {topic_id} is not available")
                return False
            return True
        except Exception as e:
            print(f"Error validating topic ID: {str(e)}")
            return False

    def _update_topic_words(self, topic_id: int, word_scores: List[Tuple[str, float]]):
        """Update topic words with error handling
        
        Args:
            topic_id (int): The topic ID to update
            word_scores (List[Tuple[str, float]]): List of (word, score) tuples
        """
        try:
            # Validate input
            if not isinstance(word_scores, list):
                raise ValueError("word_scores must be a list of tuples")

            # Validate each tuple in word_scores
            for item in word_scores:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ValueError("Each item in word_scores must be a (word, score) tuple")
                if not isinstance(item[0], str):
                    raise ValueError("Word must be a string")
                if not isinstance(item[1], (int, float)):
                    raise ValueError("Score must be a number")

            # Store in our local dictionary
            self.topic_words[topic_id] = word_scores

            # Update the topic model's internal representation
            if hasattr(self.topic_model, 'topics_'):
                self.topic_model.topics_[topic_id] = [word for word, _ in word_scores]

            if hasattr(self.topic_model, 'topic_words_'):
                self.topic_model.topic_words_[topic_id] = word_scores

        except Exception as e:
            print(f"Error updating topic words: {str(e)}")
            raise

    def _handle_merge_feedback(self):
        """Handle user feedback for merging topics with error handling"""
        try:
            # Show available topics for merging
            print("\nAvailable topics for merging:")
            available_topics = sorted(set(t for t in self.active_topics if t != -1))
            if len(available_topics) < 2:
                print("Error: Need at least 2 topics to perform merging")
                return

            for topic_id in available_topics:
                try:
                    keywords = [word for word, _ in self.get_topic_words(topic_id)]
                    print(f"Topic {topic_id}: {', '.join(keywords[:10])}")  # Show only first 10 keywords
                except Exception as e:
                    print(f"Error displaying topic {topic_id}: {str(e)}")
                    continue

            while True:
                try:
                    topics_input = input("\nEnter two topic IDs to merge (e.g., '1 2') or 'q' to quit: ").strip()
                    if topics_input.lower() == 'q':
                        return

                    # Parse and validate input
                    topics = topics_input.split()
                    if len(topics) != 2:
                        print("Error: Please enter exactly two topic IDs")
                        continue

                    topic1, topic2 = map(int, topics)

                    # Validate topics
                    if not self._validate_topic_id(topic1, available_topics) or \
                            not self._validate_topic_id(topic2, available_topics):
                        continue

                    if topic1 == topic2:
                        print("Error: Cannot merge a topic with itself")
                        continue

                    # Get words from both topics
                    words1 = dict(self.get_topic_words(topic1))
                    words2 = dict(self.get_topic_words(topic2))

                    # Simple but effective merge logic
                    merged_words = {}

                    # First pass: combine all words with their highest weight
                    for word in set(words1.keys()) | set(words2.keys()):
                        weight1 = words1.get(word, 0)
                        weight2 = words2.get(word, 0)
                        base_weight = max(weight1, weight2)

                        # Small boost for words that appear in both topics
                        if word in words1 and word in words2:
                            base_weight *= 1.1

                        merged_words[word] = base_weight

                    # Normalize weights to maintain scale
                    max_weight = max(merged_words.values(), default=1.0)
                    merged_words = {k: v / max_weight for k, v in merged_words.items()}

                    # Create sorted list of tuples
                    merged_topic_words = sorted(
                        merged_words.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    # Update topic model and local storage
                    self._update_topic_words(topic1, merged_topic_words)
                    self._update_topic_words(topic2, [])  # Empty the second topic

                    # Update tracking
                    self.merged_topics[topic2] = topic1
                    if topic2 in self.active_topics:
                        self.active_topics.remove(topic2)

                    # Add to merge suggestions
                    self.feedback.merge_suggestions.append((topic1, topic2))

                    print(f"\nTopics {topic1} and {topic2} have been merged successfully.")
                    print("\nMerged topic keywords:")
                    print(f"Keywords: {', '.join([word for word, _ in merged_topic_words[:10]])}")
                    break

                except ValueError:
                    print("Error: Please enter valid integer topic IDs")
                except Exception as e:
                    print(f"Error during merge operation: {str(e)}")
                    print("Please try again or enter 'q' to quit")

        except Exception as e:
            print(f"Critical error in merge feedback handler: {str(e)}")
            print("Returning to main menu")

    def get_feedback(self, topics: List[int]) -> TopicFeedback:
        """Interactive function to get user feedback with error handling
        
        Args:
            topics (List[int]): List of topic IDs
        
        Returns:
            TopicFeedback: The collected feedback
        """
        try:
            # Initialize active topics if not already set
            if not self.active_topics:
                self.active_topics = set(t for t in set(topics) if t != -1)

            while True:
                try:
                    print("\nCurrent Topics:")
                    # Get all unique topics that haven't been merged into others
                    display_topics = set(topics) - set(self.merged_topics.keys())

                    if not display_topics:
                        display_topics = set(self.merged_topics.values())

                    for topic_id in sorted(display_topics):
                        if topic_id != -1:  # Skip outlier topic
                            try:
                                keywords = [word for word, _ in self.get_topic_words(topic_id)]
                                print(f"\nTopic {topic_id}:")
                                print(f"Keywords: {', '.join(keywords[:10])}")

                                # Show which topics were merged into this one
                                merged_into_this = [src for src, tgt in self.merged_topics.items()
                                                    if tgt == topic_id]
                                if merged_into_this:
                                    print(f"(Contains merged topics: {', '.join(map(str, merged_into_this))})")
                            except Exception as e:
                                print(f"Error displaying topic {topic_id}: {str(e)}")
                                continue

                    print("\nFeedback Options:")
                    print("1. Merge similar topics")
                    print("2. Add keywords to topic")
                    print("3. Remove irrelevant keywords")
                    print("4. Continue with current topics")

                    choice = input("\nEnter your choice (1-5) or 'q' to quit: ").strip()
                    if choice.lower() == 'q':
                        break

                    if choice == '1':
                        if len(display_topics) < 2:
                            print("Not enough topics available for merging.")
                            continue
                        self._handle_merge_feedback()
                    elif choice == '2':
                        self._handle_keyword_addition()
                    elif choice == '3':
                        self._handle_keyword_removal()
                    elif choice == '4':
                        break
                    else:
                        print("Invalid choice. Please enter a number between 1-5 or 'q' to quit.")

                except Exception as e:
                    print(f"Error processing feedback: {str(e)}")
                    print("Please try again")

            return self.feedback

        except Exception as e:
            print(f"Critical error in feedback collection: {str(e)}")
            return self.feedback  # Return current feedback even if there's an error

    def _handle_keyword_addition(self):
        """Handle user feedback for adding keywords with error handling"""
        try:
            # Show available topics
            print("\nAvailable topics:")
            available_topics = sorted(set(t for t in self.active_topics if t != -1))
            if not available_topics:
                print("Error: No topics available")
                return

            for topic_id in available_topics:
                try:
                    keywords = [word for word, _ in self.get_topic_words(topic_id)]
                    print(f"Topic {topic_id}: {', '.join(keywords[:10])}")
                except Exception as e:
                    print(f"Error displaying topic {topic_id}: {str(e)}")
                    continue

            while True:
                try:
                    topic_input = input("\nEnter topic ID or 'q' to quit: ").strip()
                    if topic_input.lower() == 'q':
                        return

                    topic = int(topic_input)
                    if not self._validate_topic_id(topic, available_topics):
                        continue

                    keywords_input = input("Enter additional keywords (space-separated) or 'q' to quit: ").strip()
                    if keywords_input.lower() == 'q':
                        return

                    keywords = keywords_input.split()
                    if not keywords:
                        print("No keywords entered.")
                        continue

                    # Validate keywords
                    valid_keywords = []
                    for keyword in keywords:
                        if not keyword.strip():
                            continue
                        if len(keyword) < 2:
                            print(f"Warning: Skipping '{keyword}' - too short")
                            continue
                        valid_keywords.append(keyword.strip())

                    if not valid_keywords:
                        print("No valid keywords provided.")
                        continue

                    # Get current words and weights
                    current_words = dict(self.get_topic_words(topic))

                    # Add new keywords with high weight
                    for keyword in valid_keywords:
                        current_words[keyword] = 0.85  # High confidence for user-added keywords

                    # Convert back to sorted list of tuples
                    updated_words = sorted(
                        current_words.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    # Update the topic words
                    self._update_topic_words(topic, updated_words)

                    # Store in feedback
                    if topic not in self.feedback.additional_keywords:
                        self.feedback.additional_keywords[topic] = set()
                    self.feedback.additional_keywords[topic].update(valid_keywords)

                    print(f"\nKeywords {', '.join(valid_keywords)} have been added to topic {topic}.")
                    print("\nUpdated topic keywords:")
                    print(', '.join(word for word, _ in updated_words[:10]))
                    break

                except ValueError:
                    print("Error: Please enter a valid integer topic ID")
                except Exception as e:
                    print(f"Error adding keywords: {str(e)}")
                    print("Please try again or enter 'q' to quit")

        except Exception as e:
            print(f"Critical error in keyword addition handler: {str(e)}")
            print("Returning to main menu")

    def _handle_keyword_removal(self):
        """Handle user feedback for removing keywords with error handling"""
        try:
            # Show available topics
            print("\nAvailable topics:")
            available_topics = sorted(set(t for t in self.active_topics if t != -1))
            if not available_topics:
                print("Error: No topics available")
                return

            for topic_id in available_topics:
                try:
                    keywords = [word for word, _ in self.get_topic_words(topic_id)]
                    print(f"Topic {topic_id}: {', '.join(keywords[:10])}")
                except Exception as e:
                    print(f"Error displaying topic {topic_id}: {str(e)}")
                    continue

            while True:
                try:
                    topic_input = input("\nEnter topic ID or 'q' to quit: ").strip()
                    if topic_input.lower() == 'q':
                        return

                    topic = int(topic_input)
                    if not self._validate_topic_id(topic, available_topics):
                        continue

                    current_keywords = [word for word, _ in self.get_topic_words(topic)]
                    print(f"\nCurrent keywords for Topic {topic}:")
                    print(', '.join(current_keywords[:10]))

                    keywords_input = input("\nEnter keywords to remove (space-separated) or 'q' to quit: ").strip()
                    if keywords_input.lower() == 'q':
                        return

                    keywords = keywords_input.split()
                    if not keywords:
                        print("No keywords entered.")
                        continue

                    # Validate keywords exist in topic
                    valid_keywords = []
                    for keyword in keywords:
                        if keyword not in current_keywords:
                            print(f"Warning: '{keyword}' not found in topic")
                            continue
                        valid_keywords.append(keyword)

                    if not valid_keywords:
                        print("No valid keywords to remove.")
                        continue

                    # Get current words and remove specified keywords
                    current_words = dict(self.get_topic_words(topic))
                    for keyword in valid_keywords:
                        current_words.pop(keyword, None)

                    # Convert back to sorted list of tuples
                    updated_words = sorted(
                        current_words.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    # Update the topic words
                    self._update_topic_words(topic, updated_words)

                    # Store in feedback
                    if topic not in self.feedback.irrelevant_keywords:
                        self.feedback.irrelevant_keywords[topic] = set()
                    self.feedback.irrelevant_keywords[topic].update(valid_keywords)

                    print(f"\nKeywords {', '.join(valid_keywords)} have been removed from topic {topic}.")
                    print("\nUpdated topic keywords:")
                    print(', '.join(word for word, _ in updated_words[:10]))
                    break

                except ValueError:
                    print("Error: Please enter a valid integer topic ID")
                except Exception as e:
                    print(f"Error removing keywords: {str(e)}")
                    print("Please try again or enter 'q' to quit")

        except Exception as e:
            print(f"Critical error in keyword removal handler: {str(e)}")
            print("Returning to main menu")

    def summarize_changes(self, original_topics: List[int], new_topics: List[int]):
        """Summarize the changes made based on feedback"""
        try:
            print("\nTopic Changes Summary:")

            # Track original topic distribution
            orig_topic_counts = {}
            new_topic_counts = {}
            for topic in set(original_topics) | set(new_topics):
                if topic != -1:  # Skip outlier topic
                    orig_count = len([t for t in original_topics if t == topic])
                    new_count = len([t for t in new_topics if t == topic])

                    if orig_count > 0:
                        orig_topic_counts[topic] = orig_count
                    if new_count > 0:
                        new_topic_counts[topic] = new_count

            # Report merges
            if self.feedback.merge_suggestions:
                print("\nMerged Topics:")
                for source, target in self.feedback.merge_suggestions:
                    print(f"\nTopics {source} and {target} were merged:")
                    # Show final keywords for merged topic
                    if target in self.topic_words:
                        keywords = [word for word, _ in self.topic_words[target]]
                        print(f"Final keywords: {', '.join(keywords[:10])}")

            # Report keyword changes
            if self.feedback.additional_keywords:
                print("\nKeywords Added:")
                for topic, keywords in self.feedback.additional_keywords.items():
                    print(f"Topic {topic}: Added {', '.join(keywords)}")

            if self.feedback.irrelevant_keywords:
                print("\nKeywords Removed:")
                for topic, keywords in self.feedback.irrelevant_keywords.items():
                    print(f"Topic {topic}: Removed {', '.join(keywords)}")

            # Report significant size changes
            print("\nSignificant Topic Size Changes:")
            for topic in sorted(set(orig_topic_counts.keys()) | set(new_topic_counts.keys())):
                old_count = orig_topic_counts.get(topic, 0)
                new_count = new_topic_counts.get(topic, 0)
                if old_count != new_count:
                    change_pct = ((new_count - old_count) / old_count * 100) if old_count > 0 else float('inf')
                    print(f"Topic {topic}: {old_count} → {new_count} documents ({change_pct:+.1f}%)")

        except Exception as e:
            print(f"Error generating change summary: {str(e)}")
            print("Some changes may not be reflected in this summary.")

    def apply_feedback(self, texts: List[str], topics: List[int]) -> Tuple[List[int], np.ndarray]:
        """Apply feedback with error handling"""
        try:
            # Validate inputs
            if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                raise ValueError("texts must be a list of strings")
            if not isinstance(topics, list) or not all(isinstance(t, int) for t in topics):
                raise ValueError("topics must be a list of integers")
            if len(texts) != len(topics):
                raise ValueError("texts and topics must have the same length")

            # Track original topics for comparison
            original_topics = topics.copy()

            # Apply splits if any
            if self.feedback.split_suggestions and hasattr(self.topic_model, 'min_topic_size'):
                try:
                    # Store original settings
                    current_min_size = self.topic_model.min_topic_size
                    current_embeddings = self.topic_model.embedding_model

                    # Temporarily reduce min_topic_size to allow finer-grained topics
                    self.topic_model.min_topic_size = max(2, current_min_size // 2)

                    # For each topic to split
                    for topic_to_split in self.feedback.split_suggestions:
                        # Get documents belonging to this topic
                        topic_docs = [text for i, text in enumerate(texts)
                                      if topics[i] == topic_to_split]
                        topic_indices = [i for i, t in enumerate(topics)
                                         if t == topic_to_split]

                        if len(topic_docs) > current_min_size:
                            print(f"\nSplitting topic {topic_to_split} ({len(topic_docs)} documents)...")

                            # Fit a new model on just these documents
                            subtopic_model = BERTopic(
                                embedding_model=current_embeddings,
                                min_topic_size=max(2, len(topic_docs) // 4),
                                verbose=False
                            )

                            # Extract subtopics
                            subtopics, subtopic_probs = subtopic_model.fit_transform(topic_docs)

                            # Map subtopics to new global topic IDs
                            max_topic = max(t for t in set(topics) if t != -1)
                            topic_map = {-1: topic_to_split}  # Map outliers back to original
                            next_topic = max_topic + 1

                            for st in set(subtopics):
                                if st != -1:
                                    topic_map[st] = next_topic
                                    next_topic += 1

                            # Update global topics and probs
                            for local_idx, global_idx in enumerate(topic_indices):
                                if subtopics[local_idx] != -1:
                                    topics[global_idx] = topic_map[subtopics[local_idx]]
                                    # Update probabilities
                                    old_probs = list(probs[global_idx])
                                    old_probs[topics[global_idx]] = max(subtopic_probs[local_idx])
                                    probs[global_idx] = np.array(old_probs)

                            # Update topic model
                            for new_topic_id in set(topic_map.values()):
                                if new_topic_id != topic_to_split:
                                    docs_in_topic = [doc for i, doc in enumerate(topic_docs)
                                                     if topic_map[subtopics[i]] == new_topic_id]
                                    if docs_in_topic:
                                        # Update topic keywords
                                        topic_words = subtopic_model.extract_representative_docs(
                                            docs_in_topic,
                                            n_words=10
                                        )
                                        self.topic_words[new_topic_id] = [
                                            (word, 0.8) for word in topic_words
                                        ]

                            print(f"Created {len(set(topic_map.values())) - 1} new subtopics")
                        else:
                            print(f"Topic {topic_to_split} too small to split ({len(topic_docs)} documents)")

                    # Restore original settings
                    self.topic_model.min_topic_size = current_min_size

                except Exception as e:
                    print(f"Error applying splits: {str(e)}")
                    # Fall back to regular transform
                    topics, probs = self.topic_model.transform(texts)
            else:
                # Transform with current settings
                topics, probs = self.topic_model.transform(texts)

            # Apply merge mappings to the transformed topics
            new_topics = [self.merged_topics.get(t, t) for t in topics]

            # Create probability matrix with updated topics
            try:
                unique_topics = sorted(set(new_topics))
                topic_to_idx = {t: i for i, t in enumerate(unique_topics)}
                new_probs = np.zeros((len(texts), len(unique_topics)))

                for i, topic in enumerate(new_topics):
                    if topic != -1:  # Skip outlier topic
                        new_probs[i, topic_to_idx[topic]] = np.max(probs[i])

                return new_topics, new_probs

            except Exception as e:
                print(f"Error creating probability matrix: {str(e)}")
                # Return original topics and probability matrix
                return topics, probs

        except Exception as e:
            print(f"Critical error applying feedback: {str(e)}")
            # Return original topics and empty probability matrix
            return topics, np.zeros((len(texts), len(set(topics))))