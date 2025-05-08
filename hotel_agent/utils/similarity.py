"""
Similarity utility for the Hotel Agent System.
Provides embedding and similarity calculation functions.
"""

from typing import List, Dict, Any
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from config import settings
from utils.logger import logger

class SimilarityCalculator:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL
        )
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text"""
        try:
            return await self.embeddings.aembed_query(text)
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            return []
    
    def calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            if not vec1 or not vec2:
                return 0.0
            
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            # Normalize vectors
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            
            # Calculate cosine similarity
            similarity = np.dot(vec1_norm, vec2_norm)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    async def find_most_similar(self, query: str, candidates: List[str]) -> Dict[str, Any]:
        """Find most similar text from candidates"""
        try:
            query_embedding = await self.get_embedding(query)
            if not query_embedding:
                return {"text": "", "similarity": 0.0}
            
            max_similarity = 0.0
            most_similar = ""
            
            for candidate in candidates:
                candidate_embedding = await self.get_embedding(candidate)
                if not candidate_embedding:
                    continue
                
                similarity = self.calculate_similarity(query_embedding, candidate_embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar = candidate
            
            return {
                "text": most_similar,
                "similarity": max_similarity
            }
            
        except Exception as e:
            logger.error(f"Error finding most similar: {str(e)}")
            return {"text": "", "similarity": 0.0}

similarity_calculator = SimilarityCalculator() 