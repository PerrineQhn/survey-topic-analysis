from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from bertopic import BERTopic
import numpy as np
import torch
import umap
import hdbscan
from typing import Dict, Any


# Abstract Base Class for Embedding Models
class BaseEmbeddingModel(ABC):
    @abstractmethod
    def initialize(self, **kwargs):
        pass

    @abstractmethod
    def embed(self, documents: list) -> np.ndarray:
        pass


# Using Sentence-Transformers Models
class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """Sentence Transformer implementation of LLM interface.

    Available models:
    - "all-MiniLM-L6-v2"
    - "paraphrase-MiniLM-L6-v2"
    - "distiluse-base-multilingual-cased-v1"
    """

    def initialize(self, model_name: str = "all-MiniLM-L6-v2", **kwargs):
        self.model = SentenceTransformer(model_name)
        print(f"Model: SentenceTransformerEmbedding: {model_name}")

    def embed(self, documents: list) -> np.ndarray:
        return self.model.encode(
            documents, show_progress_bar=True, convert_to_numpy=True
        )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "max_seq_length": self.model.max_seq_length,
            "type": "sentence-transformer",
        }


# Using Hugging Face Models
class HuggingFaceEmbedding(BaseEmbeddingModel):
    """Hugging Face Transformers implementation of LLM interface.

    Available models:
    - "bert-base-uncased"
    - "roberta-base"
    """

    def initialize(self, model_name: str = "bert-base-uncased", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model: HuggingFaceEmbedding: {model_name}")

    def embed(self, documents: list) -> np.ndarray:
        inputs = self.tokenizer(
            documents,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
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
            "type": "hugging-face",
        }


# Custom Embeddings
class CustomEmbedding(BaseEmbeddingModel):
    def initialize(self, model, **kwargs):
        self.model = model
        print(f"Model: CustomEmbedding:{model}")

    def embed(self, documents: list) -> np.ndarray:
        return self.model.embed(documents)


# A Factory Function to Get Embedding Models
def get_embedding_model(model_type: str, **kwargs) -> BaseEmbeddingModel:
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
        model = SentenceTransformerEmbedding()
    elif model_type == "hugging-face":
        model = HuggingFaceEmbedding()
    elif model_type == "custom":
        model = CustomEmbedding()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.initialize(**kwargs)
    return model


def main():
    # Example documents - Added more documents with clearer topic distinctions
    documents = [
        # AI/ML Topic
        "Machine learning is great for data analysis and prediction.",
        "Deep learning uses neural networks for complex pattern recognition.",
        "Artificial intelligence systems can learn from experience.",
        "Neural networks are inspired by biological brain structures.",
        # Web Development Topic
        "HTML and CSS are fundamental to web development.",
        "JavaScript enables interactive web applications.",
        "Web developers use frameworks like React and Angular.",
        "Responsive design ensures websites work on all devices.",
        # Database Topic
        "SQL is used for managing relational databases.",
        "NoSQL databases provide flexible data storage.",
        "Database indexing improves query performance.",
        "Data normalization reduces redundancy in databases.",
    ]

    # Initialize the embedding model
    embedding_model = get_embedding_model("sentence-transformer", model_name="all-MiniLM-L6-v2")

    # Generate embeddings
    embeddings = embedding_model.embed(documents)
    print(f"Embeddings shape: {embeddings.shape}")

    # Create UMAP model with parameters optimized for small datasets
    umap_model = umap.UMAP(
        n_neighbors=3,  # Reduced for small dataset
        n_components=5,  # Increased dimensions for better separation
        min_dist=0.0,  # Minimum distance between points
        metric="cosine",
        random_state=42,  # For reproducibility
    )

    # Create HDBSCAN model with very lenient parameters
    hdbscan_model = hdbscan.HDBSCAN(
        min_samples=2,  # Minimum points required to form a cluster
        min_cluster_size=2,  # Minimum cluster size
        cluster_selection_epsilon=0.3,  # More lenient cluster selection
        metric="euclidean",
        cluster_selection_method="eom",  # Excess of Mass for small datasets
        prediction_data=True,
    )

    # Create BERTopic model with adjusted parameters
    topic_model = BERTopic(
        embedding_model=embedding_model.embed,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=2,  # Minimum size of topics
        nr_topics="auto",  # Let the model decide the number of topics
        verbose=True,
    )

    # Fit the model and get topics
    topics, probs = topic_model.fit_transform(documents)

    # Print topic assignments
    print("\nTopic Assignments:")
    for doc, topic in zip(documents, topics):
        print(f"Topic {topic}: {doc}")

    # Print topic information
    if len(set(topics)) > 1 or (len(set(topics)) == 1 and -1 not in topics):
        print("\nTop Terms per Topic:")
        for topic in sorted(set(topics)):
            if topic != -1:  # Skip outlier topic
                terms = topic_model.get_topic(topic)
                print(f"\nTopic {topic}:")
                for term, score in terms[:5]:  # Show top 5 terms
                    print(f"  - {term}: {score:.4f}")

        # Print topic info
        print("\nTopic Information:")
        topic_info = topic_model.get_topic_info()
        print(topic_info)


if __name__ == "__main__":
    main()
