import streamlit as st
import pandas as pd
import plotly.express as px
import io
import plotly.graph_objects as go
from topic_analyzer import TopicAnalyzer, calculate_topic_quality_scores
from topic_tagger import TopicTagger
from topic_guidance import TopicGuidanceManager
from data_loader import DataLoader
from feedback_manager import TopicFeedbackManager

st.set_page_config(layout="wide")
st.title("Topic Analysis Dashboard")


def init_session_state():
    """Initialize session state variables."""
    if "analyzed" not in st.session_state:
        st.session_state.analyzed = False
    if "results" not in st.session_state:
        st.session_state.results = None
    if "texts" not in st.session_state:
        st.session_state.texts = None
    if "valid_indices" not in st.session_state:
        st.session_state.valid_indices = None
    if "topic_tagger" not in st.session_state:
        st.session_state.topic_tagger = None
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = None
    if "feedback_manager" not in st.session_state:
        st.session_state.feedback_manager = None
    if "tag_results" not in st.session_state:
        st.session_state.tag_results = None
    if "response_df" not in st.session_state:
        st.session_state.response_df = None


@st.cache_data
def plot_topic_distribution(topic_info: pd.DataFrame) -> go.Figure:
    """Plot the distribution of topics.

    Args:
        topic_info (pd.DataFrame): DataFrame containing topic information.

    Returns:
        plotly.graph_objects.Figure: Plotly figure object.
    """
    fig = px.bar(
        topic_info[topic_info["Topic"] != -1],
        x="Topic",
        y="Count",
        title="Topic Distribution",
    )
    return fig


def plot_quality_metrics(quality_scores: dict) -> go.Figure:
    """Plot the quality metrics of the topic model.

    Args:
        quality_scores (dict): Dictionary containing quality scores.

    Returns:
        plotly.graph_objects.Figure: Plotly figure object.
    """
    fig = go.Figure(
        data=[
            go.Bar(
                x=list(quality_scores.keys()),
                y=list(quality_scores.values()),
                text=[f"{v:.3f}" for v in quality_scores.values()],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title="Topic Quality Metrics",
        xaxis_title="Metric",
        yaxis_title="Score",
        yaxis_range=[0, 1],
    )
    return fig


def update_topic_assignments():
    """Update topic assignments based on feedback."""
    # Recalculate topic assignments
    st.session_state.tag_results = st.session_state.topic_tagger.tag_responses(
        st.session_state.texts, st.session_state.results.embeddings
    )

    # Update response dataframe
    st.session_state.response_df = pd.DataFrame(
        {
            "Text": st.session_state.texts,
            "Topics": [
                st.session_state.tag_results.multi_label_matrix.loc[
                    i, "multi_choice_format"
                ]
                for i in range(len(st.session_state.texts))
            ],
            "Quality Score": [
                st.session_state.tag_results.quality_scores[i]
                for i in range(len(st.session_state.texts))
            ],
        }
    )


def display_feedback_section():
    """Display the feedback interface in Streamlit."""
    st.header("Topic Feedback")

    if not st.session_state.analyzed:
        st.warning("Please analyze topics first before providing feedback.")
        return

    # Create tabs for different feedback types
    feedback_tab, review_tab = st.tabs(["Provide Feedback", "Review Changes"])

    with feedback_tab:
        # Topic Management Section
        st.subheader("Topic Management")
        col1, col2 = st.columns(2)

        with col1:
            # Merge Topics
            st.write("**Merge Topics**")
            available_topics = sorted(st.session_state.feedback_manager.active_topics)

            topics_to_merge = st.multiselect(
                "Select topics to merge",
                available_topics,
                format_func=lambda x: f"Topic {x}: {st.session_state.results.topic_labels.get(x, '')}",
            )

            if len(topics_to_merge) >= 2 and st.button("Merge Selected Topics"):
                primary_topic = topics_to_merge[0]
                topics_to_remove = topics_to_merge[1:]

                # Update feedback manager
                for topic in topics_to_remove:
                    st.session_state.feedback_manager.merged_topics[topic] = (
                        primary_topic
                    )
                    if topic in st.session_state.feedback_manager.active_topics:
                        st.session_state.feedback_manager.active_topics.remove(topic)

                # Add to merge suggestions
                st.session_state.feedback_manager.feedback.merge_suggestions.append(
                    (primary_topic, topics_to_remove)
                )

                # Update topic model and recalculate assignments
                update_topic_assignments()
                st.success(
                    f"Topics {topics_to_remove} merged into Topic {primary_topic}"
                )
                st.rerun()

        with col2:
            # Keyword Management
            st.write("**Keyword Management**")
            selected_topic = st.selectbox(
                "Select topic",
                available_topics,
                format_func=lambda x: f"Topic {x}: {st.session_state.results.topic_labels.get(x, '')}",
            )

            if selected_topic is not None:
                # Show current keywords
                current_keywords = st.session_state.analyzer.get_topic_keywords(
                    selected_topic
                )
                st.write("Current keywords:", ", ".join(current_keywords))

                # Add keywords
                new_keywords = st.text_input(
                    "Add keywords (comma-separated)", key=f"add_kw_{selected_topic}"
                )
                if new_keywords and st.button("Add Keywords"):
                    keywords_list = [
                        k.strip() for k in new_keywords.split(",") if k.strip()
                    ]
                    if keywords_list:
                        if (
                            selected_topic
                            not in st.session_state.feedback_manager.feedback.additional_keywords
                        ):
                            st.session_state.feedback_manager.feedback.additional_keywords[
                                selected_topic
                            ] = set()
                        st.session_state.feedback_manager.feedback.additional_keywords[
                            selected_topic
                        ].update(keywords_list)
                        st.success("Keywords added successfully!")
                        st.rerun()

                # Remove keywords
                keywords_to_remove = st.multiselect(
                    "Select keywords to remove",
                    current_keywords,
                    key=f"remove_kw_{selected_topic}",
                )
                if keywords_to_remove and st.button("Remove Selected Keywords"):
                    if (
                        selected_topic
                        not in st.session_state.feedback_manager.feedback.irrelevant_keywords
                    ):
                        st.session_state.feedback_manager.feedback.irrelevant_keywords[
                            selected_topic
                        ] = set()
                    st.session_state.feedback_manager.feedback.irrelevant_keywords[
                        selected_topic
                    ].update(keywords_to_remove)
                    st.success("Keywords removed successfully!")
                    st.rerun()

    with review_tab:
        st.subheader("Feedback Summary")

        # Display merge history
        if st.session_state.feedback_manager.merged_topics:
            st.write("**Merged Topics:**")
            for (
                source,
                target,
            ) in st.session_state.feedback_manager.merged_topics.items():
                st.write(f"- Topic {source} merged into Topic {target}")

        # Display keyword changes
        if st.session_state.feedback_manager.feedback.additional_keywords:
            st.write("**Added Keywords:**")
            for (
                topic,
                keywords,
            ) in st.session_state.feedback_manager.feedback.additional_keywords.items():
                st.write(f"- Topic {topic}: Added {', '.join(keywords)}")

        if st.session_state.feedback_manager.feedback.irrelevant_keywords:
            st.write("**Removed Keywords:**")
            for (
                topic,
                keywords,
            ) in st.session_state.feedback_manager.feedback.irrelevant_keywords.items():
                st.write(f"- Topic {topic}: Removed {', '.join(keywords)}")

        # # Apply Changes Button
        # if st.button("Apply All Changes"):
        #     with st.spinner("Applying feedback changes..."):
        #         new_topics, new_probs = st.session_state.feedback_manager.apply_feedback(
        #             st.session_state.texts,
        #             st.session_state.results.topics
        #         )
        #         st.session_state.results.topics = new_topics
        #         update_topic_assignments()
        #         st.success("Changes applied successfully!")
        #         st.rerun()


def main():
    init_session_state()

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # Model type and name selection with reset functionality on change
    model_type = st.sidebar.selectbox(
        "Select LLM Model Type", ["sentence-transformer", "hugging-face"]
    )
    model_name = (
        st.sidebar.selectbox(
            "Select Model",
            ["all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2"],
        )
        if model_type == "sentence-transformer"
        else st.sidebar.selectbox("Select Model", ["bert-base-uncased", "roberta-base"])
    )

    # Reset results when model or settings change
    if "model_type" in st.session_state and st.session_state.model_type != model_type:
        st.session_state.results = None
    if "model_name" in st.session_state and st.session_state.model_name != model_name:
        st.session_state.results = None

    # Save the current settings in session state
    st.session_state.model_type = model_type
    st.session_state.model_name = model_name

    st.header("Topic Analysis")

    # File upload and analysis section
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

    if uploaded_file:
        # Load and process data
        loader = DataLoader()
        with st.spinner("Chargement des données..."):
            df = pd.read_excel(uploaded_file)
            df = loader.preprocess_data(df)
            st.session_state.df = df  # Store df in session state
            st.success(f"Dataset loaded successfully. Found {len(df)} rows.")

        # Column selection and analysis
        analyzer = TopicAnalyzer()
        text_columns = analyzer._find_text_columns(df)

        if not text_columns:
            st.error("No suitable text columns found for analysis.")
            return

        selected_column = st.selectbox("Select text column to analyze", text_columns)
        min_topic_size = st.slider("Minimum topic size", 2, 10, 3)

        if st.button("Analyze Topics"):
            with st.spinner("Analyzing topics..."):
                # Prepare texts
                texts = []
                valid_indices = []
                for idx, text in df[selected_column].items():
                    if pd.notna(text):
                        cleaned = loader.clean_text(str(text))
                        if cleaned and len(cleaned) > 5:
                            texts.append(cleaned)
                            valid_indices.append(idx)

                if not texts:
                    st.error("No valid texts found for analysis after cleaning.")
                    return

                # Initialize components
                st.session_state.analyzer = TopicAnalyzer(
                    embedding_model_type=model_type,
                    min_topic_size=min_topic_size,
                    model=model_name,
                )
                st.session_state.results = st.session_state.analyzer.extract_topics(
                    texts
                )
                st.session_state.texts = texts
                st.session_state.valid_indices = valid_indices
                st.session_state.topic_tagger = TopicTagger(
                    st.session_state.analyzer.topic_model
                )
                st.session_state.feedback_manager = TopicFeedbackManager(
                    st.session_state.analyzer.topic_model
                )
                st.session_state.feedback_manager.active_topics = set(
                    t for t in st.session_state.results.topics if t != -1
                )

                # Initial topic assignments
                update_topic_assignments()
                st.session_state.analyzed = True

        # Display results if analysis has been performed
        if st.session_state.analyzed:
            # Display quality metrics
            st.subheader("Topic Model Quality")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    plot_topic_distribution(st.session_state.results.topic_info)
                )
            with col2:
                quality_scores = calculate_topic_quality_scores(
                    st.session_state.analyzer.topic_model, st.session_state.results
                )
                st.plotly_chart(plot_quality_metrics(quality_scores))

            # Display topics
            st.subheader("Topic Analysis Results")
            for topic_id in sorted(st.session_state.results.topic_labels.keys()):
                with st.expander(
                    f"Topic {topic_id}: {st.session_state.results.topic_labels[topic_id]}"
                ):
                    # Show keywords and examples
                    keywords = st.session_state.analyzer.get_topic_keywords(topic_id)
                    st.write("**Top Keywords:**", ", ".join(keywords))
                    st.write("**Example Responses:**")
                    topic_docs = [
                        text
                        for i, text in enumerate(st.session_state.texts)
                        if st.session_state.results.topics[i] == topic_id
                    ][:3]
                    for doc in topic_docs:
                        st.write(f"- {doc[:200]}...")

            # Display response analysis
            st.subheader("Response Analysis")
            st.write("Responses and their assigned topics:")
            st.dataframe(
                st.session_state.response_df.sort_values(
                    "Quality Score",
                    ascending=False,
                ),
                use_container_width=True,
            )

            # Display feedback section
            display_feedback_section()

            # Export results
            if st.button("Export Results"):
                # Récupérer df et selected_column depuis st.session_state
                df = st.session_state.df.copy()
                selected_column = selected_column

                for idx, text in zip(
                    st.session_state.valid_indices, st.session_state.texts
                ):
                    topic_assignments = (
                        st.session_state.tag_results.multi_label_matrix.loc[
                            st.session_state.texts.index(text), "multi_choice_format"
                        ]
                    )
                    df.loc[idx, f"{selected_column}_Topics"] = topic_assignments

                # Sauvegarder dans un fichier Excel en mémoire
                output = io.BytesIO()
                df.to_excel(output, index=False)
                output.seek(0)

                st.download_button(
                    label="Download Results",
                    data=output,
                    file_name="resultats_analyse_topics.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )


if __name__ == "__main__":
    main()
