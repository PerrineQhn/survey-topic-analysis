# Detailed Guide: topic_extraction.py

This script implements the topic extraction process using BERTopic and language models. It forms the core of the analysis system.

## Data Classes (DataClasses)

### TopicExtractionResults
**Purpose**: Container for topic extraction results.

**Attributes**:
- `topics`: List of assigned topic IDs
- `topic_info`: DataFrame with topic information
- `topic_labels`: Dictionary mapping IDs to topic labels
- `embeddings`: Document embeddings
- `topic_embeddings`: Topic center embeddings
- `document_info`: Additional document information
- `probabilities`: Topic assignment probabilities

## TopicExtractorBERTopic Class

### __init__(model_type: str, model_name: str, min_topic_size: int)
**Purpose**: Initialize the topic extractor.

**Parameters**:
- `model_type`: LLM model type (default: "sentence-transformer")
- `model_name`: Model name (default: "all-MiniLM-L6-v2")
- `min_topic_size`: Minimum topic size (default: 3)

**Initialization**:
1. LLM model configuration
2. UMAP configuration for dimensionality reduction
3. HDBSCAN configuration for clustering
4. BERTopic initialization
5. Feedback manager setup

### _initialize_umap() -> umap.UMAP
**Purpose**: Initialize UMAP model with optimal parameters.

**Configuration**:
- n_neighbors: 5
- n_components: 5
- min_dist: 0.0
- metric: "cosine"
- low_memory: False

### _initialize_hdbscan(min_topic_size: int) -> hdbscan.HDBSCAN
**Purpose**: Initialize HDBSCAN model for clustering.

**Configuration**:
- min_cluster_size: min_topic_size
- metric: "euclidean"
- cluster_selection_method: "eom"
- prediction_data: True

### _embedding_function(texts) -> np.ndarray
**Purpose**: Generate embeddings for input texts.

**Usage**:
- Callback function for BERTopic
- Uses configured LLM model

### extract_topics(texts) -> TopicExtractionResults
**Purpose**: Extract topics from provided texts.

**Steps**:
1. Embedding generation
2. BERTopic application
3. Topic information retrieval
4. Label generation
5. Topic embedding calculation

### Auxiliary Methods

#### _generate_topic_labels(topic_info: pd.DataFrame) -> Dict[int, str]
**Purpose**: Generate topic labels.

#### _calculate_topic_embeddings(topics, embeddings) -> Dict[int, np.ndarray]
**Purpose**: Calculate average embeddings for each topic.

#### get_model_info() -> Dict[str, Any]
**Purpose**: Get LLM model information.

#### get_topic_keywords(topic_id: int, top_n: int = 10) -> list
**Purpose**: Retrieve keywords for a topic.

## Utility Functions

### clean_and_extract_texts(df: pd.DataFrame, column_name: str) -> Tuple[List[str], List[int]]
**Purpose**: Clean and extract valid texts from a column.

### display_extracted_topics(topic_extractor: TopicExtractorBERTopic, topics: List[int])
**Purpose**: Display extracted topics and their keywords.

### process_feedback(tagger, topic_extractor, topic_info, topic_labels)
**Purpose**: Handle user feedback and update labels.

### save_results(tagging_results, output_path: str)
**Purpose**: Save results to Excel file.

## Main Function

### topics_extraction_process(df: pd.DataFrame, column_name: str, model_name: str)
**Purpose**: Execute complete topic extraction process.

**Steps**:
1. Text cleaning and extraction
2. Extractor initialization
3. Topic extraction
4. Results display
5. User feedback handling
6. Quality metrics calculation
7. Results saving

## Important Points
1. Model Configuration
- Critical LLM model choice for quality
- Sensitive UMAP and HDBSCAN parameters
- Minimum topic size impacts granularity

2. Performance
- Embedding caching
- Possible batch processing
- Memory management for large datasets

3. Quality
- Topic coherence metrics
- Coverage ratio
- Confidence scores

4. Feedback
- Continuous feedback integration
- Modification persistence
- Change traceability

## Key Dependencies
- bertopic: For topic modeling
- sentence-transformers: For embeddings
- umap-learn: For dimensionality reduction
- hdbscan: For clustering
- pandas: For data manipulation
