# Topic Feedback Manager Documentation

## Overview

The Topic Feedback Manager is an interactive system designed to collect, process, and apply user feedback for topic modeling refinement. It provides mechanisms for merging similar topics, managing keywords, and tracking topic modifications while maintaining data consistency and error handling throughout the process.

## Key Components

### 1. TopicFeedback Class

A dataclass that stores different types of user feedback:
- `merge_suggestions`: List of topic pairs to be merged
- `rename_suggestions`: Dictionary of topic ID to new name mappings
- `irrelevant_keywords`: Dictionary mapping topics to sets of keywords to remove
- `additional_keywords`: Dictionary mapping topics to sets of keywords to add

### 2. TopicFeedbackManager Class

Main class managing feedback collection and application:
- Topic model integration
- Feedback collection and storage
- Topic merging operations
- Keyword management
- Change tracking and reporting

## Core Functionality

### 1. Feedback Collection

#### Interactive Feedback Interface
```python
def get_feedback(self, topics: List[int]) -> TopicFeedback:
```
Provides four main feedback options:
1. Merge similar topics
2. Add keywords to topics
3. Remove irrelevant keywords
4. Continue with current topics

### 2. Topic Merging

```python
def _handle_merge_feedback(self):
```

Features:
- Interactive topic selection
- Keyword combination logic
- Weight normalization
- Merge tracking
- Error handling

#### Merge Process:
1. Display available topics
2. Collect user input
3. Validate selections
4. Combine keywords with weights
5. Update topic model
6. Track changes

### 3. Keyword Management

#### Adding Keywords
```python
def _handle_keyword_addition(self):
```
- User input collection
- Keyword validation
- Weight assignment
- Topic update
- Change tracking

#### Removing Keywords
```python
def _handle_keyword_removal(self):
```
- Current keyword display
- Removal validation
- Topic update
- Change tracking

## Usage Example

```python
# Initialize manager
feedback_manager = TopicFeedbackManager(topic_model)

# Collect feedback
feedback = feedback_manager.get_feedback(topics)

# Apply feedback
new_topics, new_probs = feedback_manager.apply_feedback(texts, topics)

# Summarize changes
feedback_manager.summarize_changes(topics, new_topics)
```

## Key Features

### 1. Interactive Interface

- Clear option presentation
- Guided user input
- Immediate feedback
- Error handling

### 2. Topic Management

- Topic merging
- Keyword addition/removal
- Active topic tracking
- Change validation

### 3. Data Consistency

- Input validation
- Error handling
- State tracking
- Data integrity checks

## Implementation Details

### 1. Topic Word Management
```python
def get_topic_words(self, topic_id: int) -> List[Tuple[str, float]]:
def _update_topic_words(self, topic_id: int, word_scores: List[Tuple[str, float]]):
```

Features:
- Local storage
- Model synchronization
- Error handling
- Validation checks

### 2. Change Tracking

The system maintains:
- Merged topics mapping
- Active topics set
- Custom topic words
- Feedback history

### 3. Validation

```python
def _validate_topic_id(self, topic_id: int, available_topics: list) -> bool:
```
Ensures:
- Valid topic IDs
- Available topics
- Input consistency
- Data integrity

## Best Practices

### 1. Topic Merging
- Review keywords before merging
- Consider topic sizes
- Validate results
- Document changes

### 2. Keyword Management
- Use specific keywords
- Validate additions
- Check removals
- Monitor impact

### 3. Change Tracking
- Review summaries
- Monitor distributions
- Document modifications
- Validate results

## Technical Requirements

- Python 3.10+
- Dependencies:
  - numpy
  - pandas
  - Topic model integration
  - Data structures support

## Advanced Features

### 1. Change Summary Generation
```python
def summarize_changes(self, original_topics: List[int], new_topics: List[int]):
```

Provides:
- Merge history
- Keyword changes
- Size changes
- Distribution analysis

### 2. Feedback Application
```python
def apply_feedback(self, texts: List[str], topics: List[int]) -> Tuple[List[int], np.ndarray]:
```

Features:
- Topic reassignment
- Probability recalculation
- Error handling
- State restoration

## Limitations and Considerations

### 1. Performance
- Interactive nature
- Memory requirements
- Processing overhead
- Scale limitations

### 2. Data Consistency
- State management
- Error propagation
- Model synchronization
- Change tracking

### 3. User Input
- Validation requirements
- Error handling
- Input formatting
- Response timing

## Error Handling

The system includes comprehensive error handling for:
- User input validation
- Topic operations
- Data consistency
- State management

### Key Error Handling Features:
1. Input validation
2. Operation verification
3. State recovery
4. Error reporting

## Implementation Guidelines

### 1. Initialization
```python
feedback_manager = TopicFeedbackManager(
    topic_model=your_topic_model
)
```

### 2. Feedback Collection
```python
# Collect with error handling
try:
    feedback = feedback_manager.get_feedback(topics)
except Exception as e:
    print(f"Error collecting feedback: {e}")
```

### 3. Feedback Application
```python
# Apply with verification
new_topics, new_probs = feedback_manager.apply_feedback(
    texts=documents,
    topics=current_topics
)
```

## Quality Monitoring

Monitor:
- Topic distributions
- Keyword changes
- Merge impact
- User feedback patterns

Use the change summary to:
- Validate modifications
- Track improvements
- Document changes
- Guide refinements