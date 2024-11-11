"""

Feedback Manager Module

This module contains a class for managing topic feedback and a Streamlit interface for interactive feedback management.

"""

import json
import os
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st


class TopicFeedbackManager:
    """
    A class for managing topic feedback and saving it to a JSON file.

    Attributes:
        feedback_file (str): Path to the JSON file where feedback is saved.
        feedback_history (Dict[str, Any]): Loaded history of feedback data.
        current_session (str): Unique identifier for the current feedback session.
    """

    def __init__(self, feedback_file: str = "topic_feedback.json"):
        """
        Initializes the TopicFeedbackManager with a feedback file and loads history.

        Args:
            feedback_file (str): Name of the feedback JSON file.
        """
        # Create data directory if it does not exist
        os.makedirs("data", exist_ok=True)

        self.feedback_file = os.path.join("data", feedback_file)
        self.feedback_history = self.load_feedback_history()
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_feedback_history(self: "TopicFeedbackManager") -> dict:
        """
        Loads feedback history from the JSON file, or creates a new structure if none exists.

        Returns:
            Dict[str, Any]: The loaded feedback history, with default structure if file is missing.
        """
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"sessions": {}, "topic_updates": {}}

    def save_feedback(self):
        """
        Saves the feedback history to the JSON file.
        """
        try:
            with open(self.feedback_file, "w", encoding="utf-8") as f:
                json.dump(self.feedback_history, f, indent=2, ensure_ascii=False)
            print(f"Feedbacks sauvegardés dans {self.feedback_file}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des feedbacks : {e}")

    def add_topic_feedback(
        self, topic_id: int, feedback_type: str, feedback_content: str
    ):
        """
        Adds a new feedback entry for a topic and saves it.

        Args:
            topic_id (int): ID of the topic receiving feedback.
            feedback_type (str): Type of feedback (e.g., "comment", "edit").
            feedback_content (str): Content or details of the feedback.
        """
        if self.current_session not in self.feedback_history["sessions"]:
            self.feedback_history["sessions"][self.current_session] = []

        feedback_entry = {
            "topic_id": topic_id,
            "type": feedback_type,
            "content": feedback_content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.feedback_history["sessions"][self.current_session].append(feedback_entry)
        self.save_feedback()

    def update_topic_label(self, topic_id: int, new_label: str):
        """
        Updates the label for a specified topic and saves the change.

        Args:
            topic_id (int): ID of the topic to update.
            new_label (str): New label to assign to the topic.
        """
        self.feedback_history["topic_updates"][str(topic_id)] = new_label
        self.save_feedback()


def interactive_topic_feedback_streamlit(
    topic_extractor: Any, topics: list, texts: list
):
    """
    Streamlit interface for managing topic feedback, allowing users to modify topic labels,
    add comments, and view history.

    Args:
        topic_extractor (Any): Instance of the topic extraction model.
        topics (list): List of topic IDs to display.
        texts (list): List of texts associated with topics.
    """
    # Initialize session variables if they do not exist
    if "topic_extractor" not in st.session_state:
        st.session_state.topic_extractor = topic_extractor
    if "topics" not in st.session_state:
        st.session_state.topics = topics
    if "texts" not in st.session_state:
        st.session_state.texts = texts

    st.subheader("Topic Feedback Management")

    # Display current topics
    st.write("Current Topics:")
    topic_df = pd.DataFrame(
        [
            {
                "Topic ID": tid,
                "Keywords": ", ".join(
                    st.session_state.topic_extractor.get_topic_keywords(tid)
                ),
            }
            for tid in set(st.session_state.topics)
            if tid != -1
        ]
    )
    st.dataframe(topic_df)

    # Create tabs for different actions
    tab1, tab2, tab3 = st.tabs(["Modify Topic Label", "Add Comment", "View History"])

    # Tab 1: Modify a topic label
    with tab1:
        topic_id = st.selectbox(
            "Select topic to modify",
            options=[t for t in set(st.session_state.topics) if t != -1],
            key="modify_label",
        )
        new_label = st.text_input("New label", key="new_label")

        # Use a form to prevent automatic re-execution
        with st.form("update_label_form"):
            submit_button = st.form_submit_button("Update Label")
            if submit_button:
                st.session_state.topic_extractor.feedback_manager.update_topic_label(
                    topic_id, new_label
                )
                st.success(f"Label for topic {topic_id} updated to: {new_label}")

    # Tab 2: Add a comment
    with tab2:
        comment_topic_id = st.selectbox(
            "Select topic to comment",
            options=[t for t in set(st.session_state.topics) if t != -1],
            key="add_comment",
        )
        comment = st.text_area("Your comment", key="comment")

        # Use a form for adding a comment
        with st.form("add_comment_form"):
            submit_comment = st.form_submit_button("Add Comment")
            if submit_comment and comment:
                st.session_state.topic_extractor.feedback_manager.add_topic_feedback(
                    comment_topic_id, "comment", comment
                )
                st.success("Comment recorded")

    # Tab 3: View history
    with tab3:
        st.json(st.session_state.topic_extractor.feedback_manager.feedback_history)

        # Move download button outside the form
        json_str = json.dumps(
            st.session_state.topic_extractor.feedback_manager.feedback_history,
            indent=2,
            ensure_ascii=False,
        )

        st.download_button(
            label="Download Feedback History",
            data=json_str,
            file_name="feedback_history.json",
            mime="application/json",
        )
