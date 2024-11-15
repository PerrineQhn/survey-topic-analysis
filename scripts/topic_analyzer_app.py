"""
Streamlit app for topic extraction and analysis with interactive feedback management.

This script provides a Streamlit app that allows users to upload an Excel file containing text data, 
extract topics using BERTopic, display topic distribution, and quality metrics, and export the results. 
Users can also provide feedback on topics, update labels, and view feedback history.

Usage:
    streamlit run scripts/topic_analyzer_app.py
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import DataLoader
from topic_tagging_utils import TopicTaggingConverter
from enhanced_topic_tagger import TopicTaggerEnhanced
from feedback_manager import interactive_topic_feedback_streamlit
from topic_extraction import TopicExtractorBERTopic


@st.cache_data
def load_data(file: Union[str, bytes]) -> Optional[pd.DataFrame]:
    """
    Load and validate input data from an uploaded file.

    Args:
        file (Union[str, bytes]): The uploaded file containing the text data in Excel format.

    Returns:
        Optional[pd.DataFrame]: The loaded data as a pandas DataFrame or None if the file is invalid or empty.
    """
    try:
        df = pd.read_excel(file)
        if df.empty:
            st.error("The uploaded file is empty.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def plot_topic_distribution(topic_info: pd.DataFrame) -> None:
    """
    Creates an interactive plot of topic distribution.

    Args:
        topic_info (pd.DataFrame): DataFrame containing topic distribution information with 'Topic' and 'Count' columns.
    """
    fig = px.bar(
        topic_info,
        x="Topic",
        y="Count",
        title="Topic Distribution",
        labels={"Count": "Number of Documents", "Topic": "Topic ID"},
    )
    st.plotly_chart(fig)


def get_keywords_in_text(text: str, keywords: List[str]) -> str:
    """
    Helper function to find keywords present in the text.

    Args:
        text (str): The text in which to search for keywords.
        keywords (List[str]): A list of keywords to find in the text.

    Returns:
        str: A comma-separated string of keywords found in the text.
    """
    return ", ".join([keyword for keyword in keywords if keyword in text])


def extract_topics(
    texts: pd.Series, model_type: str, model_name: str, min_topic_size: int
) -> Tuple[Dict[str, Any], TopicExtractorBERTopic]:
    """
    Extract topics from the provided texts using a specified model.

    Args:
        texts (pd.Series): Series containing the text data.
        model_type (str): The type of model to use ('sentence-transformer' or 'hugging-face').
        model_name (str): The name of the model to use.
        min_topic_size (int): The minimum size for a topic.

    Returns:
        Tuple[Dict[str, Any], TopicExtractorBERTopic]: A tuple containing the extracted topic results and the topic extractor instance.
    """
    topic_extractor = TopicExtractorBERTopic(
        model_type=model_type, model_name=model_name, min_topic_size=min_topic_size
    )
    return topic_extractor.extract_topics(texts), topic_extractor


def display_quality_metrics(tagger: TopicTaggerEnhanced, processed_df: pd.DataFrame) -> None:
    """
    Display quality metrics based on topic tagging results.

    Args:
        tagger (TopicTaggerEnhanced): The topic tagger used to calculate quality metrics.
        processed_df (pd.DataFrame): The processed DataFrame containing the topic-tagged data.
    """
    metrics = tagger.calculate_quality_metrics(processed_df)
    metrics_display = {"Coverage Ratio (%)": f"{metrics['coverage_ratio']:.2f}"}

    if "avg_confidence" in metrics:
        metrics_display.update(
            {
                "Average Confidence": f"{metrics['avg_confidence']['value']:.3f}",
                "Confidence Interpretation": metrics["avg_confidence"][
                    "interpretation"
                ]["level"],
            }
        )

    for topic_name, count in metrics["topic_sizes"].items():
        percentage = (count / sum(metrics["topic_sizes"].values())) * 100
        metrics_display[f"Topic '{topic_name}' Documents"] = int(count)
        metrics_display[f"Topic '{topic_name}' Percentage"] = f"{percentage:.2f}%"

    # Create a DataFrame for display and show it
    metrics_df = pd.DataFrame.from_dict(
        metrics_display, orient="index", columns=["Value"]
    )
    st.subheader("Quality Metrics")
    st.dataframe(metrics_df, use_container_width=True)

    if "avg_confidence" in metrics:
        st.write(
            f"Confidence Threshold: {metrics['avg_confidence']['interpretation']['threshold']}"
        )
        st.write(
            f"Confidence Scale: {metrics['avg_confidence']['interpretation']['scale']}"
        )


def main() -> None:
    """
    Main function for Streamlit app.

    This function initializes the Streamlit app, handles user input, performs topic extraction, 
    and displays the results and metrics.
    """
    st.title("Topic Extraction and Analysis")

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

    min_topic_size = st.sidebar.slider("Minimum Topic Size", 2, 20, 3)

    # Reset results when model or settings change
    if "model_type" in st.session_state and st.session_state.model_type != model_type:
        st.session_state.results = None
    if "model_name" in st.session_state and st.session_state.model_name != model_name:
        st.session_state.results = None
    if (
        "min_topic_size" in st.session_state
        and st.session_state.min_topic_size != min_topic_size
    ):
        st.session_state.results = None

    # Save the current settings in session state
    st.session_state.model_type = model_type
    st.session_state.model_name = model_name
    st.session_state.min_topic_size = min_topic_size

    # File uploader
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

    # Load and display data
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.write("### Data Preview:")
            st.dataframe(df.head())
            text_column = st.selectbox("Select text column for analysis:", df.columns)

            # Process data when "Extract Topics" button is pressed
            if st.button("Extract Topics") or st.session_state.get("results"):
                if not st.session_state.get("results"):
                    with st.spinner("Extracting topics..."):
                        try:
                            loader = DataLoader()
                            texts = df[text_column].apply(loader.clean_text).dropna()
                            results, topic_extractor = extract_topics(
                                texts.tolist(), model_type, model_name, min_topic_size
                            )

                            # Save results in session state
                            st.session_state.results = {
                                "topics": results.topics,
                                "topic_info": results.topic_info,
                                "topic_labels": results.topic_labels,
                                "embeddings": results.embeddings,
                                "topic_embeddings": results.topic_embeddings,
                                "document_info": results.document_info,
                                "probabilities": results.probabilities,
                                "valid_texts": texts,
                                "topic_extractor": topic_extractor,
                            }
                        except Exception as e:
                            st.error(f"Error during topic extraction: {str(e)}")
                            return

                # Display results from session state
                results = st.session_state.results
                st.subheader("Topic Extraction Results")

                # Display model info
                st.write("#### Model Information")
                model_info = results["topic_extractor"].get_model_info()
                st.json(model_info)

                # Plot topic distribution
                plot_topic_distribution(results["topic_info"])

                # Display detailed topic information
                st.write("#### Topic Details")
                topic_details = [
                    {
                        "Topic ID": topic_id,
                        "Label": label,
                        "Keywords": ", ".join(
                            results["topic_extractor"].get_topic_keywords(topic_id)[:10]
                        ),
                        "Size": len([t for t in results["topics"] if t == topic_id]),
                        "Proportion (%)": round(
                            len([t for t in results["topics"] if t == topic_id])
                            / len(results["topics"])
                            * 100,
                            2,
                        ),
                    }
                    for topic_id, label in results["topic_labels"].items()
                ]
                st.dataframe(pd.DataFrame(topic_details))

                # Process dataset for metrics and tagging
                tagger = TopicTaggerEnhanced(model_name=model_name)
                tagger.set_topics(
                    results["topic_info"], results["topic_extractor"].topic_model
                )

                processed_df = tagger.process_dataset(
                    df=pd.DataFrame({"text": results["valid_texts"]}),
                    text_column="text",
                )

                # Display quality metrics
                display_quality_metrics(tagger, processed_df)

                # Prepare output DataFrame with tagged topics and keywords
                output_df = pd.DataFrame(
                    {
                        "Text": results["valid_texts"],
                        "Topic": results["topics"],
                        "Topic_Label": [
                            results["topic_labels"].get(t, "Other")
                            for t in results["topics"]
                        ],
                        "Keywords": [
                            (
                                ", ".join(
                                    results["topic_extractor"].get_topic_keywords(t)[:5]
                                )
                                if t != -1
                                else "Other"
                            )
                            for t in results["topics"]
                        ],
                        "Keywords_In_Text": [
                            get_keywords_in_text(
                                text,
                                results["topic_extractor"].get_topic_keywords(topic)[
                                    :5
                                ],
                            )
                            for text, topic in zip(
                                results["valid_texts"], results["topics"]
                            )
                        ],
                    }
                )

                # Export results
                st.download_button(
                    "Download Results",
                    output_df.to_csv(index=False).encode("utf-8"),
                    "topic_analysis_results.csv",
                    "text/csv",
                )
                st.write("### Tagged Data")
                st.dataframe(output_df)

                # Display results with multiple-choice columns
                st.write("### Results with Multiple Choice Columns")
                st.write(processed_df)


                # # Interactive feedback section (optional)
                # st.subheader("Interactive Topic Feedback")
                # interactive_topic_feedback_streamlit(
                #     results["topic_extractor"],
                #     results["topics"],
                #     results["valid_texts"],
                # )


if __name__ == "__main__":
    main()
