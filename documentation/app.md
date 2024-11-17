# Topic Analysis Dashboard Documentation

## Overview

The Topic Analysis Dashboard is a Streamlit-based web application for interactive text topic analysis, featuring topic modeling, visualization, feedback management, and result export capabilities.

## Key Features

### 1. Session State Management
```python
def init_session_state():
```
Manages:
- Analysis status (`analyzed`)
- Topic model results (`results`)
- Text data (`texts`)
- Valid indices tracking (`valid_indices`)
- Topic tagger (`topic_tagger`)
- Topic analyzer (`analyzer`)
- Feedback management (`feedback_manager`)
- Topic assignments (`tag_results`)
- Response data (`response_df`)

### 2. Data Visualization

#### Topic Distribution
```python
@st.cache_data
def plot_topic_distribution(topic_info):
```
- Interactive bar chart of topic frequencies
- Excludes outlier topics (-1)
- Cached for performance optimization

#### Quality Metrics
```python
def plot_quality_metrics(quality_scores):
```
- Bar chart of topic model quality metrics
- Score normalization (0-1 range)
- Interactive tooltips with precise values

### 3. Topic Management System

#### Topic Assignment Updates
```python
def update_topic_assignments():
```
- Recalculates topic assignments after changes
- Updates response DataFrame
- Maintains topic quality scores
- Handles multi-label format

#### Interactive Feedback System
```python
def display_feedback_section():
```
Features:
- Topic merging interface
- Keyword management (add/remove)
- Change review and summary
- Topic modification history

## Interface Components

### 1. Data Loading
- Excel file upload support
- Automatic column type detection
- Text preprocessing
- Data validation

### 2. Analysis Controls
- Text column selection
- Minimum topic size adjustment (2-10)
- Analysis initialization
- Progress tracking

### 3. Results Display
- Topic distribution visualization
- Quality metrics display
- Topic details in expandable sections
- Comprehensive response analysis table

### 4. Feedback Management
- Topic merging interface
- Keyword addition and removal
- Change tracking and review
- Results update after modifications

### 5. Export Functionality
- Excel export with topic assignments
- Quality scores inclusion
- Original data preservation
- Multi-sheet results

## Usage Flow

### 1. Initial Setup
```python
# File Upload
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

# Configuration
selected_column = st.selectbox("Select text column to analyze", text_columns)
min_topic_size = st.slider("Minimum topic size", 2, 10, 3)
```

### 2. Topic Analysis
```python
if st.button("Analyze Topics"):
    # Initialize components
    st.session_state.analyzer = TopicAnalyzer(min_topic_size=min_topic_size)
    st.session_state.results = st.session_state.analyzer.extract_topics(texts)
```

### 3. Feedback Management
```python
# Topic Merging
topics_to_merge = st.multiselect("Select topics to merge", available_topics)

# Keyword Management
new_keywords = st.text_input("Add keywords (comma-separated)")
keywords_to_remove = st.multiselect("Select keywords to remove", current_keywords)
```

### 4. Result Export
```python
if st.button("Export Results"):
    # Create Excel file with results
    df.to_excel(output, index=False)
```

## Technical Details

### Data Processing Pipeline
1. File upload and validation
2. Text preprocessing and cleaning
3. Topic modeling and analysis
4. Quality assessment
5. Interactive feedback
6. Result export

### Visualization Components
- Plotly for interactive charts
- Streamlit components for UI
- Pandas for data management
- Custom feedback visualization

### State Management
- Session-based state persistence
- Component synchronization
- Result caching
- Error recovery

## Best Practices

### 1. Data Preparation
- Clean and validate input data
- Check for minimum text length
- Remove invalid entries
- Preprocess text appropriately

### 2. Topic Analysis
- Start with moderate topic sizes
- Monitor quality metrics
- Review topic distributions
- Validate topic assignments

### 3. Feedback Process
- Review topics before merging
- Validate keyword additions
- Monitor changes in review tab
- Apply changes incrementally

### 4. Result Export
- Include all relevant metrics
- Maintain data structure
- Add quality indicators
- Document topic assignments

## Dependencies
```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from topic_analyzer import TopicAnalyzer
from topic_tagger import TopicTagger
from feedback_manager import TopicFeedbackManager
```

## Future Improvements

### Planned Enhancements
1. **Interface**
   - Real-time topic updates
   - Advanced visualization options
   - Interactive topic exploration
   - Custom plot configurations

2. **Functionality**
   - Multiple file processing
   - Additional export formats
   - Advanced topic refinement
   - Result comparison tools

3. **Performance**
   - Enhanced caching
   - Parallel processing
   - Memory optimization
   - Asynchronous operations

## Known Limitations

### Current Constraints
- Single file processing only
- Limited file format support
- Limited to one model
- Manual refresh requirements
- Basic export options

### Performance Considerations
- Memory usage with large datasets
- Processing time for complex analysis
- Visualization limits
- Cache size restrictions

## Error Handling

### Common Issues
1. **File Upload**
   - Invalid format errors
   - Size limitations
   - Encoding issues
   - Missing data

2. **Analysis**
   - Insufficient data
   - Invalid text content
   - Processing failures
   - Memory constraints

3. **Feedback**
   - Invalid topic merges
   - Keyword conflicts
   - Update failures
   - State synchronization

### Resolution Steps
1. Validate input data
2. Monitor error messages
3. Check processing status
4. Review error logs