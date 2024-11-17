from typing import Dict, List
import pandas as pd


class TopicGuidanceManager:
    """Manages user guidance for topic interpretation and assignment"""

    def __init__(self, topic_model, topic_tagger):
        self.topic_model = topic_model
        self.topic_tagger = topic_tagger

    def collect_guidance_after_feedback(self, feedback_manager):
        """Collect guidance for topics after feedback modifications

        Args:
            feedback_manager: FeedbackManager instance
        """
        print("\nUpdating Topic Guidance After Modifications")
        print("=" * 50)
        print(
            "\nPlease provide guidance for the current topics (including modified ones):"
        )

        # Get active topics (excluding merged ones)
        active_topics = (
            feedback_manager.active_topics
            if hasattr(feedback_manager, "active_topics")
            else set()
        )
        if not active_topics:
            active_topics = set(t for t in self.topic_model.get_topics() if t != -1)

        # Clear existing guidance to avoid stale information
        self.topic_tagger.user_guidance.clear()

        # Sort active topics to ensure consistent order
        for topic_id in sorted(active_topics):
            if hasattr(feedback_manager, "merged_topics"):
                # Skip topics that were merged into others
                if topic_id in feedback_manager.merged_topics:
                    continue

            print(f"\nTopic {topic_id}:")

            # Get current keywords (checking both feedback manager and topic model)
            if (
                hasattr(feedback_manager, "topic_words")
                and topic_id in feedback_manager.topic_words
            ):
                keywords = [word for word, _ in feedback_manager.topic_words[topic_id]]
            else:
                keywords = [word for word, _ in self.topic_model.get_topic(topic_id)][
                    :10
                ]
            print(f"Current Keywords: {', '.join(keywords)}")

            # Show merged topics history if applicable
            if hasattr(feedback_manager, "merged_topics"):
                merged_sources = [
                    src
                    for src, tgt in feedback_manager.merged_topics.items()
                    if tgt == topic_id
                ]
                if merged_sources:
                    print(
                        f"(Merged from topics: {', '.join(map(str, merged_sources))})"
                    )

            # Show example documents
            topic_docs = self.topic_model.get_representative_docs(topic_id)
            if topic_docs:
                print("\nExample responses:")
                for i, doc in enumerate(topic_docs[:3], 1):
                    print(f"{i}. {doc[:100]}...")

            # Get guidance
            guidance = input(
                "\nEnter guidance for this topic (or press Enter to skip): "
            ).strip()
            if guidance:
                self.topic_tagger.add_user_guidance(topic_id, guidance)

    def collect_initial_guidance(self):
        """Collect initial guidance before any modifications"""
        print("\nInitial Topic Guidance Collection")
        print("=" * 50)
        print("\nFor each topic, provide guidance on how it should be interpreted")
        print("and when it should be assigned to responses.")

        for topic_id in self.topic_model.get_topics():
            if topic_id != -1:  # Skip outlier topic
                print(f"\nTopic {topic_id}:")
                keywords = [word for word, _ in self.topic_model.get_topic(topic_id)][
                    :10
                ]
                print(f"Keywords: {', '.join(keywords)}")

                # Show example documents
                topic_docs = self.topic_model.get_representative_docs(topic_id)
                if topic_docs:
                    print("\nExample responses:")
                    for i, doc in enumerate(topic_docs[:3], 1):
                        print(f"{i}. {doc[:100]}...")

                # Get guidance
                guidance = input(
                    "\nEnter guidance for this topic (or press Enter to skip): "
                ).strip()
                if guidance:
                    self.topic_tagger.add_user_guidance(topic_id, guidance)

    def display_topic_summary(self, tag_results: pd.DataFrame):
        """Display clear summary of topic assignments with examples

        Args:
            tag_results: DataFrame with topic assignments
        """
        print("\nTopic Analysis Summary")
        print("=" * 50)

        total_responses = len(tag_results)
        print(f"\nTotal Responses Analyzed: {total_responses}")

        # For each main topic
        for topic_id in sorted(self.topic_model.get_topics().keys()):
            if topic_id != -1:  # Skip outlier topic for now
                print(f"\nTopic {topic_id}:")
                print("-" * 30)

                # Topic details
                keywords = [word for word, _ in self.topic_model.get_topic(topic_id)][
                    :10
                ]
                print(f"Keywords: {', '.join(keywords)}")

                # Show guidance
                guidance = self.topic_tagger.get_user_guidance(topic_id)
                if guidance:
                    print(f"Guidance: {guidance}")

                # Count assignments
                topic_col = f"topic_{topic_id}"
                if topic_col in tag_results.columns:
                    assigned_count = tag_results[topic_col].sum()
                    percentage = assigned_count / total_responses * 100
                    print(f"\nAssignments:")
                    print(f"- {assigned_count} responses ({percentage:.1f}%)")

                    # Example responses
                    if "text" in tag_results.columns:
                        examples = tag_results[tag_results[topic_col] == 1][
                            "text"
                        ].head(3)
                        if not examples.empty:
                            print("\nExample Responses:")
                            for i, example in enumerate(examples, 1):
                                print(f"{i}. {example[:100]}...")

        # Other category summary
        other_col = "topic_-1"
        if other_col in tag_results.columns:
            other_count = tag_results[other_col].sum()
            if other_count > 0:
                print("\nOther/Unassigned Responses:")
                print("-" * 30)
                print(
                    f"Count: {other_count} ({(other_count / total_responses * 100):.1f}%)"
                )

                # Example unassigned responses
                if "text" in tag_results.columns:
                    other_examples = tag_results[tag_results[other_col] == 1][
                        "text"
                    ].head(2)
                    if not other_examples.empty:
                        print("\nExamples:")
                        for i, example in enumerate(other_examples, 1):
                            print(f"{i}. {example[:100]}...")

        print("\nNote: Responses may be assigned to multiple topics")
        print("=" * 50)

    def export_topic_documentation(self, output_path: str):
        """Export detailed topic documentation including guidance

        Args:
            output_path: Output file path
        """
        with open(output_path, "w") as f:
            f.write("Topic Analysis Documentation\n")
            f.write("=" * 50 + "\n\n")

            for topic_id in self.topic_model.get_topics():
                if topic_id != -1:  # Skip outlier topic
                    f.write(f"Topic {topic_id}:\n")
                    f.write("-" * 20 + "\n")

                    # Keywords
                    keywords = [
                        word for word, _ in self.topic_model.get_topic(topic_id)
                    ][:10]
                    f.write(f"Keywords: {', '.join(keywords)}\n\n")

                    # Guidance
                    guidance = self.topic_tagger.user_guidance.get(
                        topic_id, "No guidance provided"
                    )
                    f.write(f"Assignment Guidance:\n{guidance}\n\n")

                    # Example documents
                    topic_docs = self.topic_model.get_representative_docs(topic_id)
                    if topic_docs:
                        f.write("Example Responses:\n")
                        for i, doc in enumerate(topic_docs[:3], 1):
                            f.write(f"{i}. {doc[:200]}...\n")
                        f.write("\n")

                    f.write("\n" + "=" * 50 + "\n\n")
