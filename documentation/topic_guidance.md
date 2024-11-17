# Topic Guidance Manager Documentation

## Overview

The Topic Guidance Manager is a specialized component designed to handle user interaction and guidance in the topic modeling process. It facilitates the collection, management, and application of human expertise to improve topic interpretations and assignments. This system bridges the gap between automated topic modeling and human understanding by incorporating expert knowledge into the topic assignment process.

## Key Components

### TopicGuidanceManager Class

The main class that manages all guidance-related functionality:
- Integrates with the topic model
- Works with the topic tagger
- Handles user interactions
- Manages documentation and reporting

## Core Functionality

### 1. Guidance Collection

#### Initial Guidance Collection
```python
collect_initial_guidance()
```
- Collects guidance for each topic before any modifications
- Displays topic keywords and example responses
- Allows users to provide interpretation guidelines
- Stores guidance in the topic tagger

#### Post-Feedback Guidance Collection
```python
collect_guidance_after_feedback(feedback_manager)
```
- Updates guidance after topic modifications
- Handles merged topics
- Shows updated keywords and examples
- Maintains guidance consistency

### 2. Topic Summary Display

```python
display_topic_summary(tag_results: pd.DataFrame)
```

Provides comprehensive topic analysis including:
1. **Overview Statistics**
   - Total responses analyzed
   - Assignment distributions
   - Coverage metrics

2. **Per-Topic Information**
   - Keywords
   - User guidance
   - Assignment counts and percentages
   - Example responses

3. **Unassigned Responses**
   - Count and percentage
   - Example texts
   - Reasons for non-assignment

### 3. Documentation Export

```python
export_topic_documentation(output_path: str)
```

Generates detailed documentation including:
- Topic definitions
- Keywords lists
- Assignment guidelines
- Representative examples
- Usage instructions

## Key Features

### 1. Interactive Guidance Collection

The system provides:
- Clear topic presentation
- Keyword visibility
- Example response display
- User input prompts
- Guidance storage

### 2. Topic Management

Handles:
- Active topics tracking
- Merged topics history
- Keyword updates
- Example management

### 3. Reporting and Documentation

Generates:
- Topic summaries
- Assignment statistics
- Example collections
- Exportable documentation

## Usage Example

```python
# Initialize manager
guidance_manager = TopicGuidanceManager(topic_model, topic_tagger)

# Collect initial guidance
guidance_manager.collect_initial_guidance()

# After feedback modifications
guidance_manager.collect_guidance_after_feedback(feedback_manager)

# Display results
guidance_manager.display_topic_summary(tag_results)

# Export documentation
guidance_manager.export_topic_documentation("topic_documentation.txt")
```

## Best Practices

### 1. Guidance Collection
- Provide clear, concise guidelines
- Include specific examples
- Reference key indicators
- Consider edge cases

### 2. Topic Management
- Review merged topics carefully
- Update guidance after modifications
- Maintain consistency across topics
- Document changes

### 3. Documentation
- Keep guidelines updated
- Include representative examples
- Document special cases
- Maintain version history

## Integration Points

### 1. Topic Model Integration
- Access to topic keywords
- Representative document retrieval
- Topic structure understanding

### 2. Topic Tagger Integration
- Guidance storage
- Assignment influence
- Response classification

### 3. Feedback Manager Integration
- Topic modification tracking
- Merge history management
- Keyword updates

## Output Formats

### 1. Console Output
- Interactive prompts
- Topic summaries
- Assignment statistics
- Example displays

### 2. Documentation Files
- Structured text format
- Topic definitions
- Guidelines
- Examples

## Technical Requirements

- Python 3.10+
- Dependencies:
  - pandas
  - typing support
  - Topic model integration
  - Topic tagger integration

## Limitations and Considerations

### 1. User Interaction
- Requires manual input
- Quality depends on expertise
- Time-consuming process

### 2. Topic Management
- Limited by topic model capabilities
- Merge operations are permanent
- Guidance updates needed after changes

### 3. Documentation
- Text-based output only
- Manual format changes needed
- Storage requirements

## Error Handling

The system handles:
- Invalid topic IDs
- Missing feedback manager
- File writing errors
- Data consistency issues

## Best Practices for Implementation

1. **Initial Setup**
   - Configure topic model first
   - Initialize topic tagger
   - Prepare feedback manager

2. **Guidance Collection**
   - Collect before assignments
   - Update after modifications
   - Document changes

3. **Documentation Management**
   - Regular exports
   - Version control
   - Backup procedures

4. **Integration**
   - Consistent topic IDs
   - Synchronized updates
   - Error handling