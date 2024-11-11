"""

Streamlit app for topic extraction and analysis with interactive feedback management.

This script provides a Streamlit app that allows users to upload an Excel file containing text data, 
extract topics using BERTopic, display topic distribution, and quality metrics, and export the results. 
Users can also provide feedback on topics, update labels, and view feedback history.

Usage:
    streamlit run topic_analyzer_app.py

"""

from typing import Any, Dict, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import DataLoader
from enhanced_topic_tagger import TopicTaggerEnhanced
from feedback_manager import interactive_topic_feedback_streamlit
from topic_extraction import TopicExtractorBERTopic


@st.cache_data
def load_data(file) -> Optional[pd.DataFrame]:
    """
    Load and validate input data from an uploaded file.

    Args:
        file: Uploaded file in Streamlit's file uploader format.

    Returns:
        pd.DataFrame or None: Loaded data as a DataFrame, or None if loading failed.
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


def plot_topic_distribution(topic_info: pd.DataFrame):
    """
    Creates an interactive plot of topic distribution.

    Args:
        topic_info (pd.DataFrame): DataFrame containing topic information with counts.
    """
    fig = px.bar(
        topic_info,
        x="Topic",
        y="Count",
        title="Topic Distribution",
        labels={"Count": "Number of Documents", "Topic": "Topic ID"},
    )
    st.plotly_chart(fig)


def extract_topics(
    texts: pd.Series, model_type: str, model_name: str, min_topic_size: int
) -> Tuple[Dict[str, Any], TopicExtractorBERTopic]:
    """
    Extract topics from the provided texts using a specified model.

    Args:
        texts (pd.Series): Series of text documents.
        model_type (str): Type of language model (e.g., 'sentence-transformer').
        model_name (str): Specific model name to use.
        min_topic_size (int): Minimum size of topics to consider.

    Returns:
        Tuple[Dict[str, Any], TopicExtractorBERTopic]: Results dictionary and the topic extractor instance.
    """
    topic_extractor = TopicExtractorBERTopic(
        model_type=model_type, model_name=model_name, min_topic_size=min_topic_size
    )
    return topic_extractor.extract_topics(texts), topic_extractor


def main() -> None:
    """
    Main function for Streamlit app, handling user input, topic extraction,
    displaying analysis results, and exporting feedback.
    """
    st.title("Topic Extraction and Analysis")

    # Initialize session state
    if "results" not in st.session_state:
        st.session_state.results = None

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    model_type = st.sidebar.selectbox(
        "Select LLM Model Type", ["sentence-transformer", "hugging-face"]
    )

    if model_type == "sentence-transformer":
        model_name = st.sidebar.selectbox(
            "Select Model",
            ["all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2"],
        )
    else:
        model_name = st.sidebar.selectbox(
            "Select Model", ["bert-base-uncased", "roberta-base"]
        )

    min_topic_size = st.sidebar.slider("Minimum Topic Size", 2, 20, 3)

    # File uploader
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

    if uploaded_file is not None:
        # Load and preview data
        df = load_data(uploaded_file)
        if df is not None:
            st.write("Data Preview:")
            st.dataframe(df.head())

            # Column selection
            text_column = st.selectbox("Select text column for analysis:", df.columns)

            if st.button("Extract Topics") or st.session_state.results is not None:
                if st.session_state.results is None:
                    with st.spinner("Extracting topics..."):
                        try:
                            loader = DataLoader()

                            # Clean and prepare texts
                            texts = df[text_column].apply(loader.clean_text)
                            valid_texts = texts.dropna()

                            # Extract topics
                            results, topic_extractor = extract_topics(
                                valid_texts.tolist(),
                                model_type,
                                model_name,
                                min_topic_size,
                            )

                            # Store results in session state
                            st.session_state.results = {
                                "topics": results.topics,
                                "topic_info": results.topic_info,
                                "topic_labels": results.topic_labels,
                                "embeddings": results.embeddings,
                                "topic_embeddings": results.topic_embeddings,
                                "document_info": results.document_info,
                                "probabilities": results.probabilities,
                                "valid_texts": valid_texts,
                                "topic_extractor": topic_extractor,
                            }

                        except Exception as e:
                            st.error(f"Error during topic extraction: {str(e)}")
                            st.exception(e)
                            return

                # Display results
                results = st.session_state.results

                st.subheader("Topic Analysis Results")

                # Model Info
                st.subheader("Model Information")
                model_info = results["topic_extractor"].get_model_info()
                st.json(model_info)

                # Topic distribution plot
                plot_topic_distribution(results["topic_info"])

                # Topic details with keywords
                st.subheader("Topic Details")
                topic_details = []
                for topic_id, label in results["topic_labels"].items():
                    size = len([t for t in results["topics"] if t == topic_id])
                    topic_details.append(
                        {
                            "Topic ID": topic_id,
                            "Label": label,
                            "Keywords": ", ".join(
                                results["topic_extractor"].get_topic_keywords(topic_id)[
                                    :10
                                ]
                            ),
                            "Size": size,
                            "Proportion (%)": round(
                                size / len(results["topics"]) * 100, 2
                            ),
                        }
                    )
                st.dataframe(pd.DataFrame(topic_details))

                # Process dataset for metrics
                tagger = TopicTaggerEnhanced(model_name=model_name)
                tagger.set_topics(
                    results["topic_info"], results["topic_extractor"].topic_model
                )
                processed_df = tagger.process_dataset(
                    df=pd.DataFrame({"text": results["valid_texts"]}),
                    text_column="text",
                )

                # Display quality metrics
                st.subheader("Quality Metrics")
                metrics = tagger.calculate_quality_metrics(processed_df)

                if "avg_confidence" in metrics:
                    metrics_display = {
                    "Coverage Ratio (%)": f"{metrics['coverage_ratio']:.2f}",
                    "Average Confidence": f"{metrics['avg_confidence']['value']:.3f}",
                    "Confidence Interpretation": metrics['avg_confidence']['interpretation']['level'],
                }

                else:
                    metrics_display = {
                        "Coverage Ratio (%)": f"{metrics['coverage_ratio']:.2f}",
                        "Average Confidence": "N/A",
                        "Confidence Interpretation": "N/A",
                    }
                

                for topic_name, count in metrics['topic_sizes'].items():
                    percentage = (count / sum(metrics['topic_sizes'].values())) * 100
                    metrics_display[f"Topic '{topic_name}' Documents"] = int(count)
                    metrics_display[f"Topic '{topic_name}' Percentage"] = f"{percentage:.2f}%"

                # Create a DataFrame for display
                metrics_df = pd.DataFrame.from_dict(metrics_display, orient='index', columns=['Value'])

                # Display the DataFrame
                st.dataframe(metrics_df, use_container_width=True)

                # Display additional details for average confidence
                if "avg_confidence" in metrics:
                    st.write(f"Confidence Threshold: {metrics['avg_confidence']['interpretation']['threshold']}")
                    st.write(f"Confidence Scale: {metrics['avg_confidence']['interpretation']['scale']}")

                # Export button
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
                    }
                )

                # Add confidence scores
                confidence_cols = [
                    col for col in processed_df.columns if col.startswith("confidence_")
                ]
                for col in confidence_cols:
                    output_df[col] = processed_df[col]

                st.download_button(
                    "Export Results",
                    output_df.to_csv(index=False).encode("utf-8"),
                    "topic_analysis_results.csv",
                    "text/csv",
                )

                # Interactive topic feedback
                st.subheader("Interactive Topic Feedback")
                interactive_topic_feedback_streamlit(
                    results["topic_extractor"],
                    results["topics"],
                    results["valid_texts"],
                )


if __name__ == "__main__":
    main()
