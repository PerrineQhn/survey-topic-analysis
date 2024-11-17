# DataLoader Module Documentation

## Overview
The `data_loader.py` module is designed to handle loading, validation, and preprocessing of survey data stored in Excel files. It provides text cleaning capabilities, including removal of special characters, punctuation, and stopwords.

## Class `DataLoader`

### Description
This class provides the necessary methods for loading and processing survey data with text cleaning capabilities.

### Initialization
```python
loader = DataLoader()
```
Upon initialization, the class automatically loads a list of English stopwords from NLTK.

### Methods

#### `load_data(file_path: str) -> pd.DataFrame`
Loads data from an Excel file.

**Parameters:**
- `file_path` (str): Path to the Excel file

**Returns:**
- `pd.DataFrame`: Loaded data as a pandas DataFrame

**Raises:**
- `FileNotFoundError`: If the file doesn't exist
- `ValueError`: If the file is not in Excel format (.xlsx)

**Example:**
```python
df = loader.load_data('data/NLP_LLM_survey_example_1.xlsx')
```

#### `valid_file(df: pd.DataFrame) -> bool`
Verifies if the loaded file contains the expected columns.

**Parameters:**
- `df` (pd.DataFrame): Loaded data

**Returns:**
- `bool`: True if the file contains a "Respondent ID" column, False otherwise

**Example:**
```python
if loader.valid_file(df):
    # Process the data
```

#### `preprocess_data(df: pd.DataFrame) -> pd.DataFrame`
Preprocesses the data by removing duplicates and missing values.

**Parameters:**
- `df` (pd.DataFrame): Data to preprocess

**Returns:**
- `pd.DataFrame`: Processed data with duplicates and missing values removed

**Example:**
```python
clean_df = loader.preprocess_data(df)
```

#### `clean_text(text: str) -> str`
Cleans and preprocesses text data by removing special characters, punctuation, and stopwords.

**Parameters:**
- `text` (str): Input text to clean

**Returns:**
- `str`: Cleaned text with special characters, punctuation, and stopwords removed
- `None`: If input is not a string, empty, or results in empty text after cleaning

**Example:**
```python
cleaned_text = loader.clean_text("This is an example text!")
```

## Main Function

### Description
The `main()` function demonstrates the typical usage of the DataLoader class.

### Flow:
1. Creates a DataLoader instance
2. Loads data from a specified Excel file
3. Validates the file structure
4. Preprocesses the data
5. Cleans text in all columns
6. Displays a sample of the processed data

### Error Handling
- Catches and reports any exceptions during execution
- Provides specific error messages for different failure scenarios

## Usage Example

```python
from data_loader import DataLoader

def process_survey_data():
    loader = DataLoader()
    
    # Load the data
    df = loader.load_data('data/NLP_LLM_survey_example_1.xlsx')
    
    # Validate and process
    if loader.valid_file(df):
        # Preprocess the data
        df = loader.preprocess_data(df)
        
        # Clean text in all columns
        for col in df.columns:
            df[col] = df[col].apply(loader.clean_text)
            
        return df
    return None

```

## Dependencies
- pandas: For data manipulation and Excel file handling
- nltk: For stopwords and text processing
- os: For file path operations
- string: For punctuation handling

## Notes
- The module expects Excel files (.xlsx format)
- Text cleaning removes all punctuation except hyphens
- Only processes English text (uses English stopwords)
- Returns None for invalid or empty text inputs
- Requires the NLTK stopwords dataset to be installed