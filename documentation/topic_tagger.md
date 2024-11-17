# Topic Tagger Documentation

## Overview

The Topic Tagger is a sophisticated system designed to assign topics to text documents using a combination of embedding-based similarity calculations, keyword analysis, and user guidance. It provides robust multi-label topic assignment capabilities with quality scoring and detailed results tracking.

## Key Components

### 1. TopicTagResults Class

A dataclass that encapsulates the results of topic tagging:
- `assignments`: Dictionary mapping response indices to topic assignments and confidence scores
- `quality_scores`: Dictionary mapping response indices to quality scores
- `multi_label_matrix`: Pandas DataFrame containing the full assignment matrix

### 2. TopicTagger Class

The main class handling topic assignment functionality:
- Topic model integration
- Embedding-based similarity calculation
- Quality scoring
- Multi-label assignment
- User guidance integration

## Core Functionality

### 1. Topic Assignment Process

#### Embedding Generation
```python
def _initialize_topic_embeddings(self):
```
- Generates embeddings for topic representations
- Uses model's embedding function
- Handles both callable and method-based embedding models

#### Similarity Calculation
```python
def _calculate_topic_similarities(self, doc_embedding: np.ndarray):
```
- Computes cosine similarity between documents and topics
- Incorporates user guidance boosts
- Handles error cases gracefully

#### Document Tagging
```python
def tag_responses(self, texts: List[str], embeddings: Optional[np.ndarray] = None):
```
- Main tagging function
- Handles both pre-computed and new embeddings
- Generates comprehensive assignment results

### 2. Quality Scoring System

```python
def _calculate_quality_score(self, text: str, assignments: List[Tuple[int, float]]) -> float:
```

Components of quality scoring:

1. **Confidence Score (40%)**
   - Based on model confidence
   - Weighted combination of max and mean confidence
   - Scale: 0.0 to 1.0

2. **Keyword Overlap Score (30%)**
   - Measures keyword presence in text
   - Handles partial matches
   - Considers word stems

3. **Coverage Score (30%)**
   - Evaluates topic assignment coverage
   - Optimal range: 1-2 topics
   - Penalty for over-assignment

### 3. User Guidance Integration

```python
def add_user_guidance(self, topic_id: int, guidance: str):
def get_user_guidance(self, topic_id: int) -> str:
```

Features:
- Stores user guidance per topic
- Influences similarity calculations
- Provides guidance retrieval
- Affects quality scoring

## Usage Example

```python
# Initialize tagger
tagger = TopicTagger(
    topic_model,
    min_confidence=0.3,
    max_topics_per_response=3
)

# Add user guidance
tagger.add_user_guidance(1, "Technology and innovation related content")

# Tag documents
results = tagger.tag_responses(texts, embeddings)

# Display quality metrics
display_tagging_quality_scores(tagger, results)
```

## Result Formats

### 1. Multi-label Matrix
- Document-topic binary matrix
- Quality scores included
- Multiple-choice format
- Text content preserved

### 2. Quality Scores
- Per-document quality metrics
- Component breakdown
- Distribution statistics
- Range: 0.0 to 1.0

### 3. Topic Assignments
- Topic IDs with confidence scores
- Multiple assignments per document
- "Other" category handling
- User guidance integration

## Technical Features

### 1. Similarity Calculation
- Cosine similarity based
- Embedding space operations
- Confidence thresholding
- Error handling

### 2. Quality Scoring
- Multi-component evaluation
- Weighted scoring system
- User guidance bonus
- Score normalization

### 3. Topic Assignment
- Multi-label support
- Confidence thresholds
- Maximum topic limits
- Fallback handling

## Best Practices

### 1. Configuration
- Set appropriate confidence threshold
- Limit maximum topics per response
- Consider domain specifics
- Tune quality weights

### 2. User Guidance
- Provide clear, specific guidance
- Update guidance as needed
- Monitor impact on assignments
- Document guidance decisions

### 3. Quality Monitoring
- Review quality score distributions
- Check component scores
- Monitor assignment patterns
- Validate results

## Technical Requirements

- Python 3.6+
- Dependencies:
  - numpy
  - pandas
  - scikit-learn
  - Topic model integration
  - Embedding model support

## Limitations and Considerations

### 1. Performance
- Embedding computation intensive
- Scales with document count
- Memory requirements
- Processing time

### 2. Quality Scoring
- Subjective components
- Domain dependency
- Guidance influence
- Score interpretation

### 3. Topic Assignment
- Threshold sensitivity
- Multi-label complexity
- Error propagation
- Model dependencies

## Error Handling

The system includes robust error handling for:
- Embedding generation
- Similarity calculation
- Quality scoring
- Assignment processing

### Key Error Handling Features:
1. Graceful fallbacks
2. Error logging
3. Empty result handling
4. Invalid input protection

## Implementation Guidelines

### 1. Setup
```python
# Initialize with appropriate parameters
tagger = TopicTagger(
    topic_model,
    min_confidence=0.3,  # Adjust based on needs
    max_topics_per_response=3  # Limit multi-label assignments
)
```

### 2. User Guidance
```python
# Add specific guidance
tagger.add_user_guidance(
    topic_id=1,
    guidance="Detailed description of topic"
)
```

### 3. Processing
```python
# Process with error handling
try:
    results = tagger.tag_responses(texts)
except Exception as e:
    results = tagger._create_empty_results(len(texts))
```

## Quality Monitoring

The `display_tagging_quality_scores` function provides:
1. Overall statistics
2. Component breakdowns
3. Score distributions
4. Quality metrics

Monitor these metrics to:
- Validate assignments
- Identify issues
- Track performance
- Guide improvements