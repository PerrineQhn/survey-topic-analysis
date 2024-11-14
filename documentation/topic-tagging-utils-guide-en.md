# Detailed Guide: topic_tagging_utils.py

This module provides utilities for converting and processing topic labels, focusing on format management and statistics.

## Data Classes

### TopicTaggingResults
```python
@dataclass
class TopicTaggingResults:
    """Stores topic tagging results"""
    tagged_df: pd.DataFrame
    topic_distribution: Dict[str, float]
    single_label_stats: Dict[str, float]
```

## TopicTaggingConverter Class

### Main Methods

#### create_topic_columns()
**Purpose**: Create columns for each topic in the DataFrame.

```python
def create_topic_columns(
    self,
    df: pd.DataFrame,
    topic_info: pd.DataFrame,
    prefix: str
) -> pd.DataFrame:
    """Creates columns for each topic and an 'other' column"""
    df_tagged = df.copy()
    
    # Create topic columns
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id != -1:  # Ignore outliers
            column_name = f"{prefix}_topic_{topic_id}"
            df_tagged[column_name] = 0
    
    # "Other" column
    df_tagged[f"{prefix}_topic_other"] = 0
    
    return df_tagged
```

#### assign_topics()
**Purpose**: Assign topics to responses based on probabilities.

```python
def assign_topics(
    self,
    df: pd.DataFrame,
    topics: List[int],
    probabilities: np.ndarray,
    indices: List[int],
    prefix: str
) -> pd.DataFrame:
    """Assigns most probable topic to each response"""
    
    # Assignment logic
    for idx, probs in enumerate(probabilities):
        original_idx = indices[idx]
        max_prob_idx = np.argmax(probs)
        max_prob = probs[max_prob_idx]
        
        # Confidence threshold
        if max_prob < 0.2:
            df.at[original_idx, f"{prefix}_topic_other"] = 1
        else:
            column_name = f"{prefix}_topic_{max_prob_idx}"
            df.at[original_idx, column_name] = 1
            df.at[original_idx, f"{prefix}_probability_{max_prob_idx}"] = max_prob
```

#### create_multiple_choice_format()
**Purpose**: Convert responses to MCQ format with probabilities.

```python
def create_multiple_choice_format(
    self,
    df: pd.DataFrame,
    prefix: str,
    topic_info: pd.DataFrame
) -> pd.DataFrame:
    """Converts to MCQ format with topics sorted by probability"""
    
    df_mcq = df.copy()
    df_mcq[f"{prefix}_selected_topics"] = ""
    
    # Processing
    for idx in df_mcq.index:
        selected_topics = []
        
        for col in topic_columns:
            if df_mcq.at[idx, col] == 1:
                topic_id = col.split('_')[-1]
                if topic_id == 'other':
                    selected_topics.append(("Other", 0.0))
                else:
                    topic_info_row = topic_info[topic_info['Topic'] == int(topic_id)]
                    if not topic_info_row.empty:
                        topic_name = topic_info_row.iloc[0].get('Name', f'Topic_{topic_id}')
                        prob = df_mcq.at[idx, f"{prefix}_probability_{topic_id}"]
                        selected_topics.append((f"{topic_name} ({prob:.2f})", prob))
```

#### calculate_statistics()
**Purpose**: Calculate statistics on topic distribution.

```python
def calculate_statistics(
    self,
    df: pd.DataFrame,
    prefix: str
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Calculates topic distribution statistics"""
    
    # Topic distribution
    topic_distribution = {}
    for col in topic_columns:
        topic_name = col.replace(f"{prefix}_topic_", "")
        topic_distribution[topic_name] = (df[col].sum() / len(df)) * 100
    
    # Single label statistics
    responses_with_topics = df[topic_columns].sum(axis=1)
    single_label_stats = {
        'single_topic_ratio': (responses_with_topics == 1).mean() * 100,
        'no_topic_ratio': (responses_with_topics == 0).mean() * 100,
        'accuracy': np.mean([row.max() for _, row in df[topic_columns].iterrows()])
    }
```

#### process_dataset()
**Purpose**: Complete process of tagging and conversion.

```python
def process_dataset(
    self,
    df: pd.DataFrame,
    topic_info: pd.DataFrame,
    topics: List[int],
    probabilities: np.ndarray,
    indices: List[int],
    column_prefix: str
) -> TopicTaggingResults:
    """Complete process of tagging and conversion to MCQ"""
    
    # Processing steps
    df_processed = self.create_topic_columns(df, topic_info, column_prefix)
    df_processed = self.assign_topics(df_processed, topics, probabilities, indices, column_prefix)
    df_processed = self.create_multiple_choice_format(df_processed, column_prefix, topic_info)
    
    # Calculate statistics
    topic_dist, single_label_stats = self.calculate_statistics(df_processed, column_prefix)
    
    return TopicTaggingResults(
        tagged_df=df_processed,
        topic_distribution=topic_dist,
        single_label_stats=single_label_stats
    )
```

## Usage Guide

### 1. Initialization and Configuration
```python
converter = TopicTaggingConverter()
```

### 2. Complete Processing
```python
results = converter.process_dataset(
    df=data,
    topic_info=topics,
    topics=topic_list,
    probabilities=probs,
    indices=idx_list,
    column_prefix="survey"
)
```

### 3. Accessing Results
```python
# Tagged DataFrame
tagged_data = results.tagged_df

# Topic distribution
distribution = results.topic_distribution

# Statistics
stats = results.single_label_stats
```

## Best Practices

### 1. Data Management
- DataFrame copying
- Input validation
- Edge case handling

### 2. Performance
- Vectorized operations
- Loop minimization
- Memory optimization

### 3. Readability
- Clear column naming
- Detailed documentation
- Modular code

## Important Points

### 1. Data Format
- Column structure
- Data types
- NaN handling

### 2. Thresholds and Parameters
- Confidence threshold (0.2)
- Probability format
- Column prefixes

### 3. Extensibility
- Multi-format support
- Metric customization
- Code evolution

## Improvement Suggestions

1. Additional Formats
   - JSON export
   - Specialized formats
   - API integration

2. Advanced Metrics
   - Inter-topic coherence
   - Temporal analyses
   - Custom metrics

3. Optimizations
   - Parallelization
   - Smart caching
   - Memory reduction

4. Visualizations
   - Integrated graphs
   - Dashboards
   - Automatic reports
