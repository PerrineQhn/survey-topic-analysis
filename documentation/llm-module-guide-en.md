# Detailed Guide: llm_module.py

This module provides an abstract interface for different language models and their specific implementations.

## Abstract LLMInterface

### Abstract Base Class
```python
class LLMInterface(ABC):
    """Abstract interface defining the contract for LLM models."""
```

### Required Abstract Methods

#### generate_embeddings(texts: List[str]) -> np.ndarray
**Purpose**: Generate embeddings for input texts.

**Parameters**:
- `texts`: List of texts to encode

**Return**:
- NumPy array of embeddings

#### get_model_info() -> Dict[str, Any]
**Purpose**: Provide model metadata.

**Return**:
- Dictionary containing model information

## Concrete Implementations

### 1. SentenceTransformerLLM

**Available Models**:
- "all-MiniLM-L6-v2"
- "paraphrase-MiniLM-L6-v2"
- "distiluse-base-multilingual-cased-v1"

#### Initialization
```python
def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
    self.model = SentenceTransformer(model_name)
    self.model_name = model_name
```

#### Embedding Generation
```python
def generate_embeddings(self, texts: List[str]) -> np.ndarray:
    return self.model.encode(texts, convert_to_numpy=True)
```

#### Model Information
```python
def get_model_info(self) -> Dict[str, Any]:
    return {
        "model_name": self.model_name,
        "embedding_dimension": self.model.get_sentence_embedding_dimension(),
        "max_seq_length": self.model.max_seq_length,
        "type": "sentence-transformer"
    }
```

### 2. HuggingFaceLLM

**Available Models**:
- "bert-base-uncased"
- "roberta-base"

#### Initialization
```python
def __init__(self, model_name: str = "bert-base-uncased"):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name).to(self.device)
    self.model_name = model_name
```

#### Embedding Generation
```python
def generate_embeddings(self, texts: List[str]) -> np.ndarray:
    inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(self.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
    
    return embeddings.cpu().numpy()
```

#### Model Information
```python
def get_model_info(self) -> Dict[str, Any]:
    return {
        "model_name": self.model_name,
        "embedding_dimension": self.model.config.hidden_size,
        "max_seq_length": self.tokenizer.model_max_length,
        "type": "hugging-face"
    }
```

## Factory Function

### get_llm_model()
**Purpose**: Create LLM instances according to specified type.

```python
def get_llm_model(
    model_type: str = "sentence-transformer",
    model_name: str = None
) -> LLMInterface:
```

**Available Configurations**:
1. sentence-transformer:
   - "all-MiniLM-L6-v2" (default)
   - "paraphrase-MiniLM-L6-v2"
   - "distiluse-base-multilingual-cased-v1"

2. hugging-face:
   - "bert-base-uncased" (default)
   - "roberta-base"

## Usage Guide

### 1. Basic Usage
```python
# Creating an instance with default parameters
model = get_llm_model()

# Generating embeddings
texts = ["First text", "Second text"]
embeddings = model.generate_embeddings(texts)

# Getting model information
info = model.get_model_info()
```

### 2. Specific Configuration
```python
# Custom Sentence Transformer
st_model = get_llm_model(
    model_type="sentence-transformer",
    model_name="paraphrase-MiniLM-L6-v2"
)

# Custom Hugging Face model
hf_model = get_llm_model(
    model_type="hugging-face",
    model_name="roberta-base"
)
```

## Best Practices

### 1. Resource Management
- Using CUDA if available
- Memory management with torch.no_grad()
- Proper tensor cleanup

### 2. Performance
- Optimized padding and truncation
- Implicit batch processing
- Efficient numpy/tensor conversion

### 3. Extensibility
- Clear abstract interface
- Factory pattern for creation
- Documentation of supported models

## Important Points

### Dependency Installation
```plaintext
sentence-transformers
transformers
torch
```

### Memory Management
- Appropriate batch size
- GPU resource release
- Tokenizer cache

### Model Compatibility
- Framework versions
- Embedding sizes
- Sequence limitations

## Extension Example

```python
class CustomLLM(LLMInterface):
    def __init__(self, **kwargs):
        # Custom initialization
        pass
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        # Custom implementation
        pass
        
    def get_model_info(self) -> Dict[str, Any]:
        # Custom information
        pass
```
