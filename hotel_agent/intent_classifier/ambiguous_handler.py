from typing import List, Dict, Any
from .intent_types import IntentType, IntentClassification
from llm.llm_manager import llm_manager
from utils.logger import logger

class AmbiguousIntentHandler:
    """Handles ambiguous intents by providing clarification and suggestions"""
    
    async def handle_ambiguous(self, classification: IntentClassification, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an ambiguous intent by providing clarification options"""
        try:
            # Generate clarification response
            response = await self._generate_clarification(classification, context)
            
            # Update state with clarification
            state = {
                "messages": context.get("messages", []),
                "tool_results": {},
                "requires_clarification": True,
                "clarification_options": classification.suggested_actions,
                "sub_intents": [intent.value for intent in classification.sub_intents],
                "classification": classification  # Pass the classification object
            }
            
            # Add clarification message
            state["messages"].append({
                "role": "assistant",
                "content": response
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Error handling ambiguous intent: {str(e)}")
            return {
                "messages": context.get("messages", []),
                "error": f"Error handling ambiguous intent: {str(e)}",
                "tool_results": {},
                "classification": classification  # Pass the classification object even in error case
            }
    
    async def _generate_clarification(self, classification: IntentClassification, context: Dict[str, Any]) -> str:
        """Generate a clarification response for ambiguous intent"""
        try:
            prompt = f"""The user's query is ambiguous and could be interpreted in multiple ways.
            Query: {context.get('query', '')}
            Possible intents: {[intent.value for intent in classification.sub_intents]}
            
            Generate a helpful response that:
            1. Acknowledges the ambiguity
            2. Lists the possible interpretations
            3. Asks for clarification
            4. Provides specific options for the user to choose from
            
            Keep the response friendly and concise."""
            
            response = await llm_manager.generate_response(
                system_prompt="You are a helpful hotel assistant handling ambiguous requests.",
                user_prompt=prompt
            )
            
            return response or "I'm not sure if you're asking about dining or service. Could you please clarify if you need help with food/restaurant options or hotel services?"
            
        except Exception as e:
            logger.error(f"Error generating clarification: {str(e)}")
            return "I'm not sure if you're asking about dining or service. Could you please clarify if you need help with food/restaurant options or hotel services?"
    
    def _get_suggested_actions(self, sub_intents: List[IntentType]) -> List[str]:
        """Get suggested actions based on sub-intents"""
        actions = []
        for intent in sub_intents:
            if intent == IntentType.DINING:
                actions.extend([
                    "View restaurant menu",
                    "Make a reservation",
                    "Order room service"
                ])
            elif intent == IntentType.SERVICE:
                actions.extend([
                    "Request housekeeping",
                    "Report maintenance issue",
                    "Request room service"
                ])
        return list(set(actions))  # Remove duplicates 