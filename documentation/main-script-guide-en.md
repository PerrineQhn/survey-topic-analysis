# Detailed Guide: main.py

This script is the command-line entry point of the application, enabling topic analysis via a CLI interface.

## Script Structure

### Imports and Documentation
```python
"""
NLP Survey Analysis Application

Usage:
    python main.py --file_path data/survey_data.csv 
                  --column_name "Satisfaction (What did you like about the food/drinks?)" 
                  --num_topics 5 
                  --model_name "all-MiniLM-L6-v2"
"""

import argparse
import os
from typing import Tuple, Dict, Any
import pandas as pd
from dataclasses import dataclass
```

## Data Classes

### AnalysisResults
```python
@dataclass
class AnalysisResults:
    """Container for analysis results

    Attributes:
        tagged_data: DataFrame containing tagged data
        topics: List of extracted topics
        topic_info: DataFrame of topic information
        embeddings: Embeddings for each document
        topic_embeddings: Dictionary of topic embeddings
        document_info: DataFrame of document information
        probabilities: List of topic probabilities
    """
    tagged_data: pd.DataFrame
    topics: list
    topic_info: pd.DataFrame
    embeddings: Any
    topic_embeddings: Dict
    document_info: pd.DataFrame
    probabilities: Any
```

## Main Class

### SurveyAnalyzer
```python
class SurveyAnalyzer:
    """Main class for survey analysis operations"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.topic_extractor = None
        self.tagger = None
        self.feedback_manager = None
```

### Main Methods

#### load_data()
```python
def load_data(self, file_path: str) -> pd.DataFrame:
    """Load and preprocess survey data"""
    try:
        data_loader = DataLoader(file_path)
        df = data_loader.load_data()
        df = data_loader.preprocess_data(df)
        print(f"Successfully loaded data from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise
```

#### extract_topics()
```python
def extract_topics(
    self,
    data: pd.DataFrame,
    column_name: str,
    min_topic_size: int = 3
) -> Tuple:
    """Extract topics from specified column"""
    try:
        self.topic_extractor = TopicExtractorBERTopic(
            model_name=self.model_name,
            min_topic_size=min_topic_size
        )

        texts = data[column_name].dropna().tolist()
        results = self.topic_extractor.extract_topics(texts)

        print(f"Successfully extracted topics from column: {column_name}")
        return results
    except Exception as e:
        print(f"Error extracting topics: {str(e)}")
        raise
```

#### tag_topics()
```python
def tag_topics(
    self,
    data: pd.DataFrame,
    topics: list,
    topic_info: pd.DataFrame,
    probabilities: list,
    column_name: str,
) -> pd.DataFrame:
    """Tag data with extracted topics"""
    try:
        converter = TopicTaggingConverter()
        indices = data.index.tolist()
        tagged_data = converter.process_dataset(
            df=data,
            topic_info=topic_info,
            topics=topics,
            probabilities=probabilities,
            indices=indices,
            column_prefix=column_name,
        )
        print("Successfully tagged data with topics")
        return tagged_data
    except Exception as e:
        print(f"Error tagging topics: {str(e)}")
        raise
```

#### process_topics_and_feedback()
```python
def process_topics_and_feedback(
    self,
    topic_info: pd.DataFrame,
    texts: list
) -> None:
    """Process topics and manage user feedback"""
    try:
        self.tagger = TopicTaggerEnhanced(model_name=self.model_name)
        self.tagger.set_topics(topic_info, self.topic_extractor.topic_model)

        if input("\nWould you like to manage topics? (yes/no)").lower() == "yes":
            interactive_topic_editing(self.tagger)
            get_stats_summary(self.tagger)

        print("Successfully processed topics and feedback")
    except Exception as e:
        print(f"Error processing topics and feedback: {str(e)}")
        raise
```

#### enhance_with_llm()
```python
def enhance_with_llm(self, texts: list) -> Any:
    """Enhance analysis using LLM"""
    try:
        llm = get_llm_model("sentence-transformer", self.model_name)
        enhanced_data = llm.generate_embeddings(texts)
        print("Successfully enhanced data with LLM")
        return enhanced_data
    except Exception as e:
        print(f"Error enhancing with LLM: {str(e)}")
        raise
```

## Main Function

### main()
```python
def main(args: argparse.Namespace) -> None:
    """Main execution function"""
    try:
        # Initialization
        analyzer = SurveyAnalyzer(model_name=args.model_name)

        # Load data
        df = analyzer.load_data(args.file_path)
        df_with_topics = df.copy()

        # Process topics
        results = topics_extraction_process(
            df=df_with_topics,
            column_name=args.column_name,
            model_name=args.model_name
        )

        # Save results
        output_path = os.path.join("output", "survey_analysis_results.xlsx")
        results[0].to_excel(output_path, index=False)
        print(f"Results saved to {output_path}")

    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        raise
```

## Argument Parser

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NLP Survey Analysis Application",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="Path to survey data file"
    )

    parser.add_argument(
        "--column_name",
        type=str,
        required=True,
        help="Name of column to analyze"
    )

    parser.add_argument(
        "--num_topics",
        type=int,
        default=5,
        help="Number of topics to extract"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Name of LLM model to use"
    )

    args = parser.parse_args()
    main(args)
```

## Best Practices

### 1. Code Structure
- Modular organization
- Robust error handling
- Clear documentation

### 2. CLI Interface
- Explicit arguments
- Default values
- Help messages

### 3. User Feedback
- Progress messages
- Clear error handling
- Success confirmations

## Important Points

### 1. Error Handling
- Specific try/except blocks
- Detailed error messages
- Resource cleanup

### 2. Configuration
- Required arguments
- Input validation
- File paths

### 3. Performance
- Data copying
- Memory management
- Result saving

## Improvement Suggestions

1. Additional Options
   - Advanced configuration
   - Processing modes
   - Export formats

2. User Feedback
   - Progress bar
   - Detailed logs
   - Error reports

3. Automation
   - Batch processing
   - Auxiliary scripts
   - Automated tests
