from enum import Enum, auto
from typing import Optional, List
from pydantic import BaseModel

class IntentType(Enum):
    """Enum for different types of intents"""
    DINING = "dining"
    SERVICE = "service"
    AMBIGUOUS = "ambiguous"

class IntentClassification(BaseModel):
    """Model for intent classification results"""
    intent: IntentType
    confidence: float
    requires_rag: bool = False  # Default to False
    target_agent: Optional[str] = None  # Make optional
    explanation: str = ""
    sub_intents: List[IntentType] = []  # For ambiguous cases
    suggested_actions: List[str] = []  # For ambiguous cases

    def __init__(self, **data):
        super().__init__(**data)
        # Set target_agent based on intent if not provided
        if not self.target_agent and self.intent != IntentType.AMBIGUOUS:
            self.target_agent = self.intent.value
        # Set requires_rag based on intent if not provided
        if not hasattr(self, 'requires_rag'):
            self.requires_rag = self.intent == IntentType.DINING

class IntentContext(BaseModel):
    """Model for intent context"""
    query: str
    memory_context: Optional[str] = None
    previous_intents: List[IntentType] = []
    conversation_history: List[dict] = [] 