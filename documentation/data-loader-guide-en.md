# Detailed Guide: data_loader.py

This script handles loading and preprocessing survey data. It contains a main `DataLoader` class with static methods for different stages of the data loading process.

## DataLoader Class

### load_data(file_path: str) -> pd.DataFrame
**Purpose**: Load survey data from an Excel file.

**Parameters**:
- `file_path`: Path to Excel file (string or Path object)

**Return**: 
- Pandas DataFrame containing survey data

**Behavior**:
1. Verify file exists
2. Check if it's an Excel file (.xlsx)
3. Load file into DataFrame
4. Raise appropriate exceptions if errors occur

**Usage Example**:
```python
df = DataLoader.load_data("data/survey_data.xlsx")
```

### valid_file(df: pd.DataFrame) -> bool
**Purpose**: Validate the loaded DataFrame structure.

**Parameters**:
- `df`: DataFrame to validate

**Return**: 
- `True` if structure is valid
- `False` otherwise

**Validation performed**:
- Checks if first column is "Respondent ID"

**Usage Example**:
```python
if DataLoader.valid_file(df):
    # Continue processing
```

### preprocess_data(df: pd.DataFrame) -> pd.DataFrame
**Purpose**: Preprocess data by handling missing values and duplicates.

**Parameters**:
- `df`: DataFrame to preprocess

**Return**: 
- Preprocessed DataFrame

**Operations**:
1. Remove rows with missing values
2. Remove duplicates

**Usage Example**:
```python
df_clean = DataLoader.preprocess_data(df)
```

### clean_text(text: str) -> Optional[str]
**Purpose**: Clean and normalize survey response text.

**Parameters**:
- `text`: Text to clean

**Return**: 
- Cleaned text or None if text invalid

**Cleaning Operations**:
1. Convert to lowercase
2. Remove special characters (except hyphens)
3. Remove stopwords
4. Validate word count

**Conditions for returning None**:
- Non-string text
- Empty text
- Single word after cleaning

**Usage Example**:
```python
cleaned_text = DataLoader.clean_text("Here's some sample text!")
```

## main() Function
**Purpose**: Entry point for testing data loading and preprocessing.

**Operations**:
1. Load data from example file
2. Validate structure
3. Preprocess data
4. Clean text in each column
5. Display processed data preview

**Usage Example**:
```python
if __name__ == "__main__":
    main()
```

## Implemented Best Practices
1. Use of static methods for easy usage
2. Robust error handling with try/except
3. Input data validation
4. Detailed function documentation
5. Parameter and return type hinting

## Dependencies
- pandas: For data manipulation
- nltk: For stopword management
- os: For file path management
- typing: For function typing

## Important Points
1. Requires NLTK properly configured with stopwords
2. Assumes specific input file format (Excel with ID)
3. Text cleaning optimized for English
4. Fixed cleaning parameters (not configurable)

## Points of Attention

### File Handling
1. Permissions
```python
os.makedirs("data", exist_ok=True)
```

2. Encoding
```python
with open(self.feedback_file, "w", encoding="utf-8") as f:
```

### Data Structure Validation
1. Required Columns
- "Respondent ID"
- Response columns

2. Data Types
- Numeric IDs
- Text responses

### Text Processing
1. Cleaning Steps
- Case normalization
- Special character removal
- Stopword removal

2. Validation Rules
- Minimum length
- Required fields
- Format consistency

## Usage Examples

### Basic Data Loading
```python
# Initialize loader
loader = DataLoader()

# Load and validate data
df = loader.load_data("survey_responses.xlsx")
if loader.valid_file(df):
    df_clean = loader.preprocess_data(df)
```

### Text Cleaning
```python
# Clean single response
cleaned = loader.clean_text("Sample response text!")

# Clean entire column
df['cleaned_responses'] = df['responses'].apply(loader.clean_text)
```

## Improvement Suggestions

1. File Support
   - Additional formats (CSV, JSON)
   - Custom delimiters
   - Compression support

2. Preprocessing Options
   - Configurable cleaning
   - Custom validation rules
   - Language options

3. Performance
   - Batch processing
   - Memory optimization
   - Progress tracking

4. Error Handling
   - Custom exceptions
   - Detailed logging
   - Recovery options

5. Documentation
   - API reference
   - Usage examples
   - Error guides
