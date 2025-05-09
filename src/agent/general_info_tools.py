from typing import List, Dict
from langchain_core.runnables import RunnableConfig
from .general_info_rag import GeneralInfoRAG
from utils.logger import logger

def search_facilities(query: str, config: RunnableConfig) -> List[Dict]:
    """Search hotel facilities and general information.
    
    This function uses the GeneralInfoRAG system to perform semantic search for facility information.
    It can handle natural language queries about hotel facilities, amenities, and general information.
    
    Args:
        query: The search query to find relevant facility information
              (e.g., "swimming pool hours", "fitness center location", "business center features")
        config: RunnableConfig containing user configuration
    
    Returns:
        List of relevant information with their details, including:
        - id: Unique identifier for the information
        - question: The question or topic this information addresses
        - answer: Detailed answer or information about the facility/service
    
    Example:
        >>> search_facilities("swimming pool hours", config)
        [{
            "id": "1",
            "question": "What are the swimming pool hours?",
            "answer": "The swimming pool is open daily from 7:00 AM to 10:00 PM. It is located on Level 5 and features an indoor heated pool with jacuzzi."
        }]
    """
    try:
        # Initialize GeneralInfoRAG
        general_info_rag = GeneralInfoRAG()
        
        # Search for relevant facility information
        results = general_info_rag.query(query)
        
        if not results:
            logger.info(f"No information found for query: {query}")
            return []
            
        return results
    
    except Exception as e:
        logger.error(f"Error searching facilities: {str(e)}")
        return []

