"""LLM abstraction module for topic analysis"""
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class LLMInterface:
    """Interface for LLM models that can be swapped"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def generate_embeddings(self, texts: List[str]) -> List[float]:
        """Generate embeddings for input texts"""
        return self.model.encode(texts, convert_to_tensor=True).tolist()
        
    def get_model_info(self) -> Dict[str, Any]:
        """Return model information"""
        return {
            "name": self.model.get_config_dict()["model_name"],
            "type": "sentence-transformer"
        }

# Pour utiliser un autre modèle, créez une nouvelle classe qui implémente ces méthodes
# Exemple:
# class CustomLLM(LLMInterface):
#     def __init__(self, model_path):
#         self.model = load_your_model(model_path)
