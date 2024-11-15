"""
Script for implementing a unified interface for Large Language Models (LLMs).

This script defines an abstract base class `LLMInterface` to standardize interaction with 
various LLM models for generating embeddings and retrieving model information. It provides 
implementations for:

1. Sentence Transformer models using the `sentence-transformers` library.
2. Hugging Face Transformers models using the `transformers` library.

"""

from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Any, Dict, List, Union

class LLMInterface(ABC):
    """Abstract base class defining the interface for LLM models."""
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for input texts.
        
        Args:
            texts (List[str]): List of input texts to embed
            
        Returns:
            np.ndarray: Array of embeddings
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model information and parameters.
        
        Returns:
            Dict[str, Any]: Model metadata including name, dimensions, etc.
        """
        pass

class SentenceTransformerLLM(LLMInterface):
    """Sentence Transformer implementation of LLM interface.
    
    Available models:
    - "all-MiniLM-L6-v2"
    - "paraphrase-MiniLM-L6-v2"
    - "distiluse-base-multilingual-cased-v1"
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the model.
        
        Args:
            model_name (str): Name of the sentence-transformer model
        """
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "max_seq_length": self.model.max_seq_length,
            "type": "sentence-transformer"
        }

class HuggingFaceLLM(LLMInterface):
    """Hugging Face Transformers implementation of LLM interface.
    
    Available models:
    - "bert-base-uncased"
    - "roberta-base"
    """
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        """Initialize the model.
        
        Args:
            model_name (str): Name of the Hugging Face model
        """
        from transformers import AutoTokenizer, AutoModel
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model_name = model_name
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        # Tokenize and move to device
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings.cpu().numpy()
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.model.config.hidden_size,
            "max_seq_length": self.tokenizer.model_max_length,
            "type": "hugging-face"
        }

def get_llm_model(model_type: str = "sentence-transformer", model_name: str = None) -> LLMInterface:
    """Factory function to create LLM instances.
    
    Args:
        model_type (str): Type of model to create ("sentence-transformer" or "hugging-face")
        model_name (str, optional): Specific model name to use
        
    Returns:
        LLMInterface: Configured LLM model instance
        
    Available configurations:
    - sentence-transformer:
        - "all-MiniLM-L6-v2" (default)
        - "paraphrase-MiniLM-L6-v2"
        - "distiluse-base-multilingual-cased-v1"
    - hugging-face:
        - "bert-base-uncased" (default)
        - "roberta-base"
    """
    if model_type == "sentence-transformer":
        return SentenceTransformerLLM(model_name or "all-MiniLM-L6-v2")
    elif model_type == "hugging-face":
        return HuggingFaceLLM(model_name or "bert-base-uncased")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

