"""
LLM Manager for the Hotel Agent System.
Centralizes all LLM interactions using GPT-4-mini.
"""

from typing import List, Dict, Any
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, SystemMessage, HumanMessage
from config import settings
from utils.logger import logger

class LLMManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        print(settings.OPENAI_API_KEY)
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",  # Using GPT-4-mini as specified
            temperature=0,
            api_key=settings.OPENAI_API_KEY
        )
        self._initialized = True
    
    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        context: str = None
    ) -> str:
        """Generate a response using the LLM"""
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt if not context else f"Context: {context}\n\nUser Query: {user_prompt}")
            ]
            
            prompt = ChatPromptTemplate.from_messages(messages)
            logger.debug(f"Generated prompt with {len(messages)} messages")

            formatted_messages = prompt.format_messages()
            
            response = await self.llm.ainvoke(formatted_messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            raise
    
    async def classify_intent(self, query: str, format_instructions: str) -> Dict[str, Any]:
        """Classify user intent using the LLM"""
        try:
            system_prompt = """You are an intent classifier for a hotel assistant system.
            Your task is to classify the user's intent based on both the current query and the conversation context.
            
            Important guidelines:
            1. Always consider the full conversation context when available
            2. Look for references to previous messages in the current query
            3. Consider the overall conversation flow and topic
            4. Pay attention to context tags and key points from previous interactions
            
            Classify the intent into one of these categories:
            - dining (restaurant, food, menu related)
            - service (housekeeping, room service, maintenance)
            - ambiguous (if the query could be interpreted in multiple ways)
            
            You must respond with a valid JSON object in the following format:
            {
                "intent": "dining", "service", or "ambiguous",
                "confidence": float between 0 and 1,
                "requires_rag": boolean indicating if RAG is needed,
                "target_agent": "dining" or "service",
                "explanation": "brief explanation of the classification, including how context influenced the decision"
            }
            
            {format_instructions}"""
            
            response = await self.generate_response(system_prompt, query)
            
            # Parse the response as JSON
            import json
            try:
                # First ensure we have a string response
                if isinstance(response, dict):
                    return response
                elif isinstance(response, str):
                    return json.loads(response)
                else:
                    raise ValueError(f"Unexpected response type: {type(response)}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {response}")
                # Return a default response if parsing fails
                return {
                    "intent": "service",
                    "confidence": 0.0,
                    "explanation": "Failed to parse response",
                    "requires_rag": False,
                    "target_agent": "service"
                }
            
        except Exception as e:
            logger.error(f"Error in intent classification: {str(e)}")
            raise

# Create a singleton instance
llm_manager = LLMManager() 