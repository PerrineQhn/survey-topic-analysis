# Detailed Guide: embedding_model.py

This module provides an abstract interface for different embedding models and their specific implementations.

## Abstract BaseEmbeddingModel

### Abstract Base Class
```python
class BaseEmbeddingModel(ABC):
    """Abstract interface defining the contract for embedding models."""
```

### Required Abstract Methods

#### initialize(**kwargs)
**Purpose**: Initialize the model with specific parameters.

#### embed(documents: list) -> np.ndarray
**Purpose**: Generate embeddings for input documents.

**Parameters**:
- `documents`: List of texts to encode

**Return**:
- NumPy array of embeddings

## Concrete Implementations

### 1. SentenceTransformerEmbedding

**Available Models**:
- "all-MiniLM-L6-v2"
- "paraphrase-MiniLM-L6-v2"
- "distiluse-base-multilingual-cased-v1"

#### Initialization
```python
def initialize(self, model_name: str = "all-MiniLM-L6-v2"):
    self.model = SentenceTransformer(model_name)
```

#### Embedding Generation
```python
def embed(self, documents: list) -> np.ndarray:
    return self.model.encode(documents, show_progress_bar=True, convert_to_numpy=True)
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

### 2. HuggingFaceEmbedding

**Available Models**:
- "bert-base-uncased"
- "roberta-base"

#### Initialization
```python
def initialize(self, model_name: str = "bert-base-uncased", **kwargs):
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name)
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model.to(self.device)
```

#### Embedding Generation
```python
def embed(self, documents: list) -> np.ndarray:
    inputs = self.tokenizer(documents, padding=True, truncation=True, max_length=512, return_tensors="pt")
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

### 3. CustomEmbedding

A flexible class for implementing custom embedding models.

#### Initialization
```python
def initialize(self, model, **kwargs):
    self.model = model
```

#### Embedding Generation
```python
def embed(self, documents: list) -> np.ndarray:
    return self.model.embed(documents)
```

## Factory Function

### get_embedding_model()
**Purpose**: Create embedding model instances according to specified type.

```python
def get_embedding_model(
    model_type: str,
    **kwargs
) -> BaseEmbeddingModel
```

**Available Configurations**:
1. sentence_transformers:
   - "all-MiniLM-L6-v2" (default)
   - "paraphrase-MiniLM-L6-v2"
   - "distiluse-base-multilingual-cased-v1"

2. hugging-face:
   - "bert-base-uncased" (default)
   - "roberta-base"

3. custom:
   - Accepts custom model implementation

## Usage Example

The module includes a main function demonstrating usage with topic modeling:

```python
# Example documents with distinct topics
documents = [
    # AI/ML Topic
    "Machine learning is great for data analysis and prediction.",
    "Deep learning uses neural networks for complex pattern recognition.",
    # ...

    # Web Development Topic
    "HTML and CSS are fundamental to web development.",
    "JavaScript enables interactive web applications.",
    # ...

    # Database Topic
    "SQL is used for managing relational databases.",
    "NoSQL databases provide flexible data storage.",
    # ...
]

# Initialize embedding model
embedding_model = get_embedding_model("hugging-face", model_name="bert-base-uncased")

# Generate embeddings and perform topic modeling using BERTopic
topic_model = BERTopic(
    embedding_model=embedding_model.embed,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    min_topic_size=2,
    nr_topics="auto"
)
```

## Dependencies
- sentence-transformers
- transformers
- torch
- numpy
- umap-learn
- hdbscan
- bertopic

## Important Notes

### Resource Management
- Automatically uses CUDA if available
- Proper memory management with torch.no_grad()
- Automatic device selection (CPU/GPU)

### Performance Features
- Progress bar for long operations
- Efficient batch processing
- Automatic tensor management

### Extensibility
- Clean abstract interface
- Factory pattern implementation
- Support for custom models