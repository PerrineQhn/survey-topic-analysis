# Topic Analyzer Documentation

## Overview

The Topic Analyzer is a sophisticated Python tool designed to extract, analyze, and manage topics from text data using advanced natural language processing techniques. It combines several powerful components including BERTopic, UMAP, HDBSCAN, and various embedding models to provide robust topic modeling capabilities.

## Key Components

### TopicAnalyzer Class

The main class that orchestrates the topic analysis process. It integrates:
- Embedding model (default: sentence_transformers)
- UMAP for dimensionality reduction
- HDBSCAN for clustering
- BERTopic for topic modeling

### TopicExtractionResults Class

A dataclass that contains all the results from topic extraction:
- Topics: List of assigned topic IDs
- Topic info: DataFrame with topic metadata
- Topic labels: Dictionary mapping topic IDs to descriptive labels
- Embeddings: Document embeddings array
- Topic embeddings: Dictionary of topic centroid embeddings
- Document info: DataFrame with document metadata
- Probabilities: Topic assignment probabilities

## Core Functionality

### 1. Text Column Analysis

The system automatically identifies suitable text columns for analysis by:
- Checking for minimum text length (>5 characters)
- Filtering out purely numeric content
- Requiring at least 5 meaningful text entries per column

### 2. Topic Extraction Process

1. **Embedding Generation**
   - Converts text to numerical vectors using the specified embedding model
   - Handles batch processing for large datasets

2. **Dimensionality Reduction**
   - Uses UMAP to reduce embedding dimensions while preserving relationships
   - Configured for optimal topic separation (n_neighbors=5, n_components=5)

3. **Clustering**
   - Applies HDBSCAN for robust cluster identification
   - Minimum cluster size configurable (default: 3)
   - Uses EOM (Excess of Mass) for cluster selection

4. **Topic Modeling**
   - Generates topic keywords and labels
   - Calculates topic probabilities for each document
   - Handles outlier detection (-1 topic ID)

### 3. Quality Metrics

The system calculates several quality metrics:

1. **Coherence**
   - Measures how well topic words co-occur
   - Based on word probability within topics

2. **Distinctiveness**
   - Evaluates topic separation
   - Uses cosine distance between topic embeddings

3. **Coverage**
   - Percentage of documents assigned to meaningful topics
   - Excludes outlier assignments

4. **Confidence**
   - Average probability of topic assignments
   - Indicates assignment reliability

### 4. Interactive Feedback System

The tool supports interactive refinement through:

1. **Topic Merging**
   - Combines similar or related topics
   - Maintains assignment history

2. **Keyword Management**
   - Adding relevant keywords
   - Removing irrelevant keywords
   - Tracking keyword modifications

3. **Topic Guidance**
   - User-provided descriptions
   - Influences future assignments

### 5. Multi-topic Assignment

The system supports assigning multiple topics to a single document:
- Uses confidence thresholds (default: 0.15)
- Considers relative scoring
- Implements fallback mechanisms for borderline cases

## Usage Example

```python
# Initialize analyzer
analyzer = TopicAnalyzer(
    min_topic_size=3,
    embedding_model_type="sentence-transformer"
)

# Process a specific column
results_df = process_column(
    file_path="NLP_LLM_survey_example_1.xlsx",
    column_name="Satisfaction (What did you like about the food/drinks?)",
    min_topic_size=3
)
```

## Output

The system generates:

1. **Excel File**
   - Topic assignment columns
   - Multi-choice format column
   - Preserves original data

2. **Console Output**
   - Topic summaries
   - Quality metrics
   - Assignment statistics
   - Example responses

## Best Practices

1. **Data Preparation**
   - Clean text data before analysis
   - Remove empty rows
   - Ensure sufficient text length

2. **Topic Refinement**
   - Use feedback system for unclear topics
   - Provide guidance for ambiguous cases
   - Review example responses

3. **Assignment Tuning**
   - Adjust confidence thresholds if needed
   - Monitor unassigned response rate
   - Check for topic overlap

## Technical Requirements

- Python 3.6+
- Required packages:
  - bertopic
  - hdbscan
  - umap-learn
  - pandas
  - numpy
  - scipy
  - sentence-transformers (default embedding model)

## Limitations and Considerations

1. **Performance**
   - Processing time increases with dataset size
   - Memory usage scales with embedding dimensions

2. **Topic Quality**
   - Minimum cluster size affects granularity
   - Small datasets may produce fewer topics
   - Outlier detection may vary with data distribution

3. **Assignment Accuracy**
   - Influenced by text quality and length
   - May require threshold adjustments
   - Multiple assignments can increase complexity

## Error Handling

The system includes comprehensive error handling for:
- File loading issues
- Invalid column selections
- Processing errors
- Empty or invalid text data

All errors are logged with appropriate messages to aid debugging and maintenance.