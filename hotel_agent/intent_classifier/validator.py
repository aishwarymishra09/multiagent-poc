"""
Intent classification validator for the Hotel Agent System.
Uses LLM to validate classifications and falls back to similarity if needed.
"""

from typing import Dict, Any, Tuple
from pydantic import BaseModel, Field
from config import settings
from utils.logger import logger
from utils.similarity import similarity_calculator
from llm.llm_manager import llm_manager
from prompts.classifier_prompts import DINING_EXAMPLES, SERVICE_EXAMPLES

class ValidationResult(BaseModel):
    is_valid: bool = Field(description="Whether the classification is valid")
    confidence: float = Field(description="Confidence score of the validation")
    suggested_intent: str = Field(description="Suggested intent if different from original")
    explanation: str = Field(description="Explanation for the validation decision")

class IntentValidator:
    def __init__(self):
        self.intent_examples = {
            "dining": DINING_EXAMPLES,
            "service": SERVICE_EXAMPLES
        }
    
    async def validate_classification(
        self,
        query: str,
        original_intent: str,
        original_confidence: float
    ) -> Tuple[bool, str, float]:
        """Validate the classification using LLM and fall back to similarity if needed"""
        try:
            # First validation attempt with LLM
            validation_result = await self._validate_with_llm(query, original_intent)
            
            if validation_result.is_valid:
                return True, original_intent, original_confidence
            
            # If LLM suggests different intent, try reclassification
            if validation_result.suggested_intent != original_intent:
                logger.info(f"LLM suggested different intent: {validation_result.suggested_intent}")
                return False, validation_result.suggested_intent, validation_result.confidence
            
            # If LLM validation fails, try similarity-based validation
            return await self._validate_with_similarity(query, original_intent)
            
        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
            return await self._validate_with_similarity(query, original_intent)
    
    async def _validate_with_llm(
        self,
        query: str,
        original_intent: str
    ) -> ValidationResult:
        """Validate classification using LLM"""
        try:
            system_prompt = """You are an intent validation expert for a hotel assistant system.
            Your task is to validate if the given classification of a user query is correct.
            Consider the following:
            1. The query's main topic and intent
            2. The context and domain of the query
            3. The specific requirements of each intent category
            
            IMPORTANT: You can only suggest one of these valid intents:
            - "dining" (for restaurant, food, menu related queries)
            - "service" (for housekeeping, room service, maintenance)
            - "ambiguous" (if the query could be interpreted in multiple ways)
            
            Return a JSON object with:
            - is_valid: boolean indicating if the classification is correct
            - confidence: float between 0 and 1
            - suggested_intent: must be one of: "dining", "service", or "ambiguous"
            - explanation: brief explanation of your decision"""
            
            user_prompt = f"""Query: {query}
            Original Classification: {original_intent}
            
            Please validate this classification and provide your analysis."""
            
            response = await llm_manager.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            # Parse and validate the response
            result = ValidationResult.parse_raw(response)
            
            # Ensure suggested_intent is valid
            if result.suggested_intent not in ["dining", "service", "ambiguous"]:
                logger.warning(f"Invalid suggested intent: {result.suggested_intent}, defaulting to original")
                result.suggested_intent = original_intent
                result.is_valid = True
                result.confidence = 0.5
                result.explanation = "Invalid suggested intent received, keeping original classification"
            
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM validation: {str(e)}")
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                suggested_intent=original_intent,
                explanation="Error in LLM validation"
            )
    
    async def _validate_with_similarity(
        self,
        query: str,
        original_intent: str
    ) -> Tuple[bool, str, float]:
        """Validate classification using similarity matching"""
        try:
            # Compare query with examples for both intents
            dining_similarity = await similarity_calculator.find_most_similar(
                query,
                self.intent_examples["dining"]
            )
            
            service_similarity = await similarity_calculator.find_most_similar(
                query,
                self.intent_examples["service"]
            )
            
            # Determine the best matching intent
            if dining_similarity["similarity"] > service_similarity["similarity"]:
                best_intent = "dining"
                best_similarity = dining_similarity["similarity"]
            else:
                best_intent = "service"
                best_similarity = service_similarity["similarity"]
            
            # Check if the original intent matches the similarity-based intent
            is_valid = best_intent == original_intent
            
            return is_valid, best_intent, best_similarity
            
        except Exception as e:
            logger.error(f"Error in similarity validation: {str(e)}")
            return False, original_intent, 0.0 