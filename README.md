# Survey Topic Analysis System

A comprehensive system for analyzing survey responses using topic modeling, with support for interactive feedback and multiple user interfaces. The system combines BERTopic for topic extraction with modern language models for enhanced analysis.

## Features

- Topic extraction using BERTopic and transformer-based language models
- Interactive topic management and feedback collection
- Multiple user interfaces (CLI and Streamlit web app)
- Support for custom topic labels and user guidance
- Quality metrics and visualization of topic distribution
- Export functionality for analysis results

## Technical Approach

### Overview

The system implements a hybrid approach to topic modeling that combines statistical methods with modern language models:

1. Text Preprocessing Pipeline
   - Removal of special characters and punctuation
   - Stopword filtering
   - Text normalization and validation
   - Handling of missing values and duplicates
2. Topic Extraction Strategy
   - Uses BERTopic as the core topic modeling framework
   - Enhances topic detection with transformer-based embeddings
   - Implements a two-stage process:
     1. Document embedding generation using sentence transformers
     2. Clustering using UMAP dimensionality reduction and HDBSCAN
3. Topic Management and Refinement
   - Interactive feedback loop for topic refinement
   - Support for custom topic labels and keywords
   - User guidance integration to improve topic assignments
   - Continuous topic quality monitoring
4. Quality Assurance
   - Calculation of topic coherence scores
   - Coverage ratio measurement
   - Topic size distribution analysis
   - Confidence scoring for assignments

### Key Design Decisions

1. Modular Architecture
   - Separation of concerns between data loading, processing, and visualization
   - Abstract interfaces for LLM models allowing easy swapping of backends
   - Independent feedback management system
2. Scalability Considerations
   - Batch processing for large datasets
   - Caching of embeddings to improve performance
   - Efficient storage of topic feedback
3. User Experience
   - Dual interface (CLI and Web) to accommodate different use cases
   - Interactive visualization of results
   - Flexible export options
4. Model Selection
   - Default to all-MiniLM-L6-v2 for optimal performance/speed trade-off
   - Support for multiple language models to handle different requirements
   - Configurable parameters for fine-tuning

## Architecture

The system consists of several key components:

1. **Core Processing**
   - Topic extraction using BERTopic and LLM models
   - Topic tagging and classification
   - User feedback management

2. **User Interfaces**
   - Streamlit web application for interactive analysis
   - Command-line interface for batch processing

3. **Utilities**
   - Data loading and preprocessing
   - Topic tagging utilities
   - LLM model interfaces

## Dependencies

```plaintext
Python 3.12
transformers
pandas
numpy
streamlit
bertopic
torch
plotly
nltk
openpyxl
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/PerrineQhn/survey-topic-analysis.git
cd survey-topic-analysis
```

2. Create and activate a virtual environment:
```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data:
```python
import nltk
nltk.download('stopwords')
```

## Usage

### Web Interface

1. Start the Streamlit app at the root :
```bash
streamlit run scripts/topic_analyzer_app.py
```

2. Upload your Excel file containing survey data
3. Select the text column to analyze
4. Configure model parameters in the sidebar
5. Extract topics and explore the results

### Command Line Interface

Run the analysis through the command line at the root :
```bash
python scripts/main.py --file_path data/NLP_LLM_survey_example_1.xlsx \
               --column_name "Satisfaction (What did you like about the food/drinks?)" \
               --num_topics 5 \
               --model_name "all-MiniLM-L6-v2"
```

## Project Structure

```plaintext
survey-topic-analysis/
├── data/                        # Data directory
├── output/                      # Output directory
├── scripts/                     # Script directory
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── enhanced_topic_tagger.py # Topic tagging and management
│   ├── feedback_manager.py      # User feedback handling
│   ├── llm_module.py            # LLM model interfaces
│   ├── topic_extraction.py      # Core topic extraction functionality
│   ├── topic_tagging_utils.py   # Topic tagging utilities
│   ├── main.py                  # Command line interface
│   └── topic_analyzer_app.py    # Streamlit web interface
├── documentation/               # Documentation directory
│   ├── guide.md				 # Guide about scripts fonctionnalities
├── README.md					 # Project summary
└── requirements.txt             # Project dependencies
```

## Input Data Format

The system expects an Excel file (.xlsx) with:
- One text column containing survey responses
- Optional metadata columns
- First column should be "Respondent ID"

## Output

The system generates:
- Topic assignments for each response
- Topic distribution visualization
- Quality metrics
- Interactive feedback management
- Exportable results in CSV format

## Model Configuration

Available LLM models:
- Sentence Transformers:
  - all-MiniLM-L6-v2 (default)
  - paraphrase-multilingual-MiniLM-L12-v2
- Hugging Face:
  - bert-base-uncased
  - roberta-base

Configurable parameters:
- Minimum topic size
- Similarity threshold
- UMAP and HDBSCAN parameters

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BERTopic for topic modeling
- Sentence Transformers for embeddings
- Streamlit for the web interface
