# Detailed Guide: feedback_manager.py

This module manages user feedback on extracted topics, with a Streamlit interface for interactive management and a JSON persistence system.

## TopicFeedbackManager Class

### __init__(feedback_file: str = "topic_feedback.json")
**Purpose**: Initialize the feedback manager.

**Parameters**:
- `feedback_file`: Path to JSON storage file

**Initialization**:
1. Creation of data directory
2. Loading of history
3. Creation of unique session ID

```python
def __init__(self, feedback_file: str = "topic_feedback.json"):
    os.makedirs("data", exist_ok=True)
    self.feedback_file = os.path.join("data", feedback_file)
    self.feedback_history = self.load_feedback_history()
    self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
```

### load_feedback_history() -> dict
**Purpose**: Load feedback history from JSON file.

**Return Structure**:
```json
{
    "sessions": {},
    "topic_updates": {}
}
```

**Error Handling**:
- Creation of new structure if file is missing
- JSON structure validation

### save_feedback()
**Purpose**: Save feedback history to JSON file.

**Operations**:
1. JSON formatting with indentation
2. UTF-8 encoding
3. Write error handling

### add_topic_feedback(topic_id: int, feedback_type: str, feedback_content: str)
**Purpose**: Add a new feedback entry.

**Supported feedback types**:
- comment: General comments
- edit: Topic modifications
- guidance: User instructions

**Entry Structure**:
```python
feedback_entry = {
    "topic_id": topic_id,
    "type": feedback_type,
    "content": feedback_content,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}
```

### update_topic_label(topic_id: int, new_label: str)
**Purpose**: Update a topic's label.

**Operations**:
1. Record new label
2. Update JSON file
3. Maintain data consistency

## Streamlit Interface

### interactive_topic_feedback_streamlit(topic_extractor: Any, topics: list, texts: list)
**Purpose**: Graphical interface for feedback management.

**Features**:

#### 1. Session Initialization
```python
if "topic_extractor" not in st.session_state:
    st.session_state.topic_extractor = topic_extractor
```

#### 2. Current Topics Display
- Topics list
- Associated keywords
- Interactive DataFrames

#### 3. Management Tabs
**Tab 1: Label Modification**
- Topic selection
- New label input
- Form validation

**Tab 2: Comment Addition**
- Topic selection
- Comment text area
- Feedback recording

**Tab 3: History**
- JSON history display
- Download button
- Structured visualization

### State Management
1. Session Variables
```python
st.session_state.topics = topics
st.session_state.texts = texts
```

2. Forms and Validations
```python
with st.form("update_label_form"):
    submit_button = st.form_submit_button("Update Label")
```

3. User Feedback
```python
st.success("Comment recorded")
```

## Best Practices

### 1. Data Persistence
- Automatic saving
- Structured JSON format
- Error handling

### 2. User Interface
- Tab organization
- Validation forms
- Confirmation messages

### 3. Traceability
- Modification timestamps
- Unique sessions
- Complete history

### 4. Performance
- Optimized loading
- Streamlit cache
- Efficient updates

## Important Points

### File Management
1. Permissions
```python
os.makedirs("data", exist_ok=True)
```

2. Encoding
```python
with open(self.feedback_file, "w", encoding="utf-8") as f:
```

### Data Structure
1. Sessions
```python
"sessions": {
    "20240411_123456": [
        {
            "topic_id": 1,
            "type": "edit",
            "content": "Updated label",
            "timestamp": "2024-04-11 12:34:56"
        }
    ]
}
```

2. Topic Updates
```python
"topic_updates": {
    "1": "New Label",
    "2": "Updated Topic"
}
```

## Typical Usage

### Initialization
```python
feedback_manager = TopicFeedbackManager()
```

### Adding Feedback
```python
feedback_manager.add_topic_feedback(
    topic_id=1,
    feedback_type="comment",
    feedback_content="This topic needs more precision"
)
```

### Label Update
```python
feedback_manager.update_topic_label(
    topic_id=1,
    new_label="New topic name"
)
```

### Streamlit Interface
```python
interactive_topic_feedback_streamlit(
    topic_extractor=extractor,
    topics=topic_list,
    texts=text_list
)
```

## Improvement Suggestions

1. Data Export
   - Additional formats (CSV, Excel)
   - Export filters

2. Visualizations
   - Evolution graphs
   - Dashboards

3. Validation
   - Business rules
   - Integrity constraints

4. Collaboration
   - Multi-user support
   - Conflict management
