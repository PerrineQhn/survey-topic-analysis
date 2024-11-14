"""Topic analysis and management module"""
from typing import List, Dict, Tuple
import pandas as pd
from bertopic import BERTopic
from llm_module import LLMInterface

class TopicAnalyzer:
    def __init__(self, llm_model: LLMInterface):
        self.llm = llm_model
        self.topic_model = BERTopic(
            embedding_model=self.llm.model,
            calculate_probabilities=True
        )
        self.topics = {}
        self.topic_stats = {}

    def extract_topics(self, texts: List[str]) -> Tuple[List[int], List[float]]:
        """Extract topics from texts"""
        topics, probs = self.topic_model.fit_transform(texts)
        self._update_topic_stats(topics)
        return topics, probs

    def _update_topic_stats(self, topics: List[int]):
        """Update topic statistics"""
        for topic_id in set(topics):
            if topic_id != -1:
                topic_size = sum(1 for t in topics if t == topic_id)
                keywords = [word for word, _ in self.topic_model.get_topic(topic_id)]
                self.topics[topic_id] = {
                    "keywords": keywords,
                    "size": topic_size,
                    "name": f"Topic {topic_id}"
                }

    def edit_topic(self, topic_id: int, new_name: str = None, new_keywords: List[str] = None):
        """Edit topic name or keywords"""
        if topic_id in self.topics:
            if new_name:
                self.topics[topic_id]["name"] = new_name
            if new_keywords:
                self.topics[topic_id]["keywords"] = new_keywords

    def tag_responses(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Tag responses with topics"""
        texts = df[text_column].fillna('').tolist()
        topics, probs = self.extract_topics(texts)
        
        # Create topic columns
        result = df.copy()
        for topic_id in self.topics:
            result[f"topic_{self.topics[topic_id]['name']}"] = 0
        result["topic_other"] = 0
        
        # Assign topics
        for idx, (topic, prob) in enumerate(zip(topics, probs)):
            if topic != -1:
                topic_name = self.topics[topic]["name"]
                result.loc[idx, f"topic_{topic_name}"] = 1
            else:
                result.loc[idx, "topic_other"] = 1
            
        # Add confidence scores (numpy.float32 is not JSON serializable)
        result["confidence_score"] = [float(score) for score in probs]
        
        return result

    def get_quality_metrics(self) -> Dict[str, float]:
        """Calculate quality metrics"""
        return {
            "topic_sizes": {tid: info["size"] for tid, info in self.topics.items()},
            "average_topic_size": sum(t["size"] for t in self.topics.values()) / len(self.topics) if self.topics else 0,
            "total_topics": len(self.topics)
        }
