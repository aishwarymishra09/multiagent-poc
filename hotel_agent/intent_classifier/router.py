"""
Intent classifier router for the Hotel Agent System.
Handles intent classification, validation, and fallback.
"""

from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field, ValidationError
from config import settings
from utils.logger import logger
from utils.similarity import similarity_calculator
from llm.llm_manager import llm_manager
from prompts.classifier_prompts import (
    CLASSIFIER_SYSTEM_PROMPT,
    CLASSIFIER_FORMAT_INSTRUCTIONS,
    DINING_EXAMPLES,
    SERVICE_EXAMPLES
)
from intent_classifier.validator import IntentValidator

class IntentClassification(BaseModel):
    intent: str = Field(description="The classified intent of the user query")
    confidence: float = Field(description="Confidence score of the classification")
    requires_rag: bool = Field(description="Whether this query requires RAG system")
    target_agent: str = Field(description="The agent that should handle this query")
    explanation: str = Field(description="Explanation for the classification decision")

class IntentClassifier:
    def __init__(self):
        self.intent_examples = {
            "dining": DINING_EXAMPLES,
            "service": SERVICE_EXAMPLES
        }
        self.validator = IntentValidator()
    async def parse_intent_classification(self, response: Union[str, dict]) -> IntentClassification:
        try:
            if isinstance(response, str):
                return IntentClassification.parse_raw(response)
            elif isinstance(response, dict):
                return IntentClassification.parse_obj(response)
            else:
                raise TypeError("Unsupported input type for intent parsing")
        except ValidationError as e:
            # Optional: log or handle gracefully
            raise ValueError(f"Invalid intent classification format: {e}")
    async def classify(self, query: str, context: str = None) -> IntentClassification:
        """Classify the user's query and return the intent with confidence score"""
        try:
            # First classification attempt
            logger.info(f"Starting classification for query: {query}")
            logger.info(f"Using context: {context}")
            
            # Prepare the full context for classification
            full_context = ""
            if context:
                full_context = f"""Previous conversation context:
{context}

Current query: {query}

Please classify the intent considering both the current query and the conversation context."""
            else:
                full_context = query
            
            response = await llm_manager.classify_intent(
                full_context,
                CLASSIFIER_FORMAT_INSTRUCTIONS
            )
            
            if not response:
                logger.error("Empty response from LLM manager")
                return IntentClassification(
                    intent="service",
                    confidence=0.0,
                    requires_rag=False,
                    target_agent="service",
                    explanation="Failed to get response from LLM"
                )
                
            try:
                classification = await self.parse_intent_classification(response)
            except Exception as e:
                logger.error(f"Failed to parse classification: {str(e)}")
                return IntentClassification(
                    intent="service",
                    confidence=0.0,
                    requires_rag=False,
                    target_agent="service",
                    explanation=f"Failed to parse response: {str(e)}"
                )
            
            # First validation attempt with context
            is_valid, validated_intent, validated_confidence = await self.validator.validate_classification(
                full_context,  # Pass full context to validator
                classification.intent,
                classification.confidence
            )
            
            if is_valid:
                # If first validation succeeds, return the classification
                classification.intent = validated_intent
                classification.confidence = validated_confidence
                classification.target_agent = validated_intent
                return classification
                
            # If first validation fails, try classification again with context
            response = await llm_manager.classify_intent(
                full_context,
                CLASSIFIER_FORMAT_INSTRUCTIONS
            )
            
            if not response:
                logger.error("Empty response from second LLM attempt")
                return IntentClassification(
                    intent="service",
                    confidence=0.0,
                    requires_rag=False,
                    target_agent="service",
                    explanation="Failed to get response from second LLM attempt"
                )
                
            try:
                classification = await self.parse_intent_classification(response)
            except Exception as e:
                logger.error(f"Failed to parse second classification: {str(e)}")
                return IntentClassification(
                    intent="service",
                    confidence=0.0,
                    requires_rag=False,
                    target_agent="service",
                    explanation=f"Failed to parse second response: {str(e)}"
                )
            
            # Second validation attempt with context
            is_valid, validated_intent, validated_confidence = await self.validator.validate_classification(
                full_context,  # Pass full context to validator
                classification.intent,
                classification.confidence
            )
            
            if is_valid:
                # If second validation succeeds, return the classification
                classification.intent = validated_intent
                classification.confidence = validated_confidence
                classification.target_agent = validated_intent
                return classification
            
            # If both validations fail, fall back to similarity matching with context
            logger.info("Both validations failed, falling back to similarity matching")
            return await self._fallback_classify(full_context)
            
        except Exception as e:
            logger.error(f"Error in classification: {str(e)}")
            return IntentClassification(
                intent="service",
                confidence=0.0,
                requires_rag=False,
                target_agent="service",
                explanation=f"Error in classification: {str(e)}"
            )
    
    def _validate_classification(self, classification: IntentClassification) -> bool:
        """Validate the classification result"""
        try:
            # Check confidence threshold
            if classification.confidence < settings.CONFIDENCE_THRESHOLD:
                return False
            
            # Validate target agent
            if classification.target_agent not in ["dining", "service"]:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating classification: {str(e)}")
            return False
    
    async def _fallback_classify(self, query: str) -> IntentClassification:
        """Fallback classification using similarity matching"""
        try:
            # Compare query with example intents
            max_similarity = 0.0
            best_intent = "service"  # Default to service
            
            for intent, examples in self.intent_examples.items():
                similarity = await similarity_calculator.find_most_similar(query, examples)
                if similarity["similarity"] > max_similarity:
                    max_similarity = similarity["similarity"]
                    best_intent = intent
            
            return IntentClassification(
                intent=best_intent,
                confidence=max_similarity,
                requires_rag=best_intent == "dining",
                target_agent=best_intent
            )
            
        except Exception as e:
            logger.error(f"Error in fallback classification: {str(e)}")
            # Return default classification
            return IntentClassification(
                intent="service",
                confidence=0.0,
                requires_rag=False,
                target_agent="service"
            ) 