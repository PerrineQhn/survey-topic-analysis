# Topic Analysis Tool Documentation

## Overview
This script implements a tool for analyzing topics in text data using various embedding models. It supports multiple types of embedding models including custom implementations, Hugging Face models, and sentence transformers.

## Dependencies
```python
from pathlib import Path
import nltk
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from topic_analyzer import TopicAnalyzer, process_column
```

## Classes

### MyCustomEmbeddingModel
A customizable embedding model class that processes text data using sentence transformers.

#### Parameters
- `base_model_name` (str, optional): Name of the base model to use. Defaults to "YourModel".

#### Methods
##### `__init__(self, base_model_name: str = "YourModel")`
Initializes the custom embedding model with a specified base model.

##### `embed(self, documents: List[str]) -> np.ndarray`
Processes and embeds a list of documents.

**Parameters:**
- `documents` (List[str]): List of text documents to embed

**Returns:**
- `np.ndarray`: Normalized document embeddings

**Processing Steps:**
1. Text preprocessing (lowercase and whitespace removal)
2. Base embedding generation using SentenceTransformer
3. L2 normalization of embeddings

## Main Function

### `main()`
The primary function that orchestrates the topic analysis process.

#### Workflow
1. Sets up the analysis environment
2. Loads and validates the input file
3. Configures the embedding model
4. Processes the specified column
5. Offers interactive column analysis

#### Configuration Options
```python
# Custom Model Configuration
model = MyCustomEmbeddingModel()
embedding_model_type = "custom"

# Hugging Face Model Configuration
model = "bert-base-uncased"
embedding_model_type = "hugging-face"

# Available embedding_model_type options:
# - "custom"
# - "hugging-face"
# - "sentence-transformer"
```

## Usage

### Basic Usage
```python
python main.py
```

### Input File Requirements
- Format: Excel (.xlsx) or CSV
- Required Structure: Must contain text columns for analysis
- Default Path: './data/NLP_LLM_survey_example_1.xlsx'

### Example Configuration
```python
file_path = './data/NLP_LLM_survey_example_1.xlsx'
col = 'Satisfaction (What did you like about the food/drinks?)'
model = "bert-base-uncased"
embedding_model_type = "hugging-face"

df = process_column(
    file_path=file_path,
    column_name=col,
    embedding_model_type=embedding_model_type,
    model=model,
    model_name=model
)
```

## Error Handling
- File existence validation
- Model initialization error handling
- Processing error handling

## Interactive Features
- Column selection
- Multiple column analysis option
- Process continuation prompts

## Performance Considerations
- Embeddings are normalized for consistency
- Custom preprocessing steps available
- Supports batch processing through the embedding model

## Integration
The script integrates with the broader topic analysis system through:
- `topic_analyzer.TopicAnalyzer`: Main analysis engine
- `topic_analyzer.process_column`: Column processing utility

## Notes
- The script is designed to be modular and extensible
- Custom embedding models can be easily integrated
- Supports multiple embedding model types
- Interactive command-line interface for ease of use

## Limitations
- Currently supports single column analysis at a time
- Requires pre-formatted input files
- Model dependencies must be installed separately

## Future Improvements
- Multi-column parallel processing
- Additional embedding model support
- Enhanced error reporting
- Configuration file support
- Memory optimization for large datasets