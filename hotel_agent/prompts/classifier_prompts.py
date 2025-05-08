"""
Intent classification prompts for the Hotel Agent System.
Contains all prompts used by the intent classifier.
"""

CLASSIFIER_SYSTEM_PROMPT = """You are an intent classifier for a hotel assistant system.
Classify the user's query into one of these categories:
- dining (restaurant, food, menu related)
- service (housekeeping, room service, maintenance)

{format_instructions}"""

CLASSIFIER_FORMAT_INSTRUCTIONS = """Return a JSON object with the following fields:
- intent: The classified intent (dining or service)
- confidence: A float between 0 and 1
- requires_rag: Boolean indicating if RAG is needed
- target_agent: The agent to handle the query (dining or service)"""

# Example queries for each intent category
DINING_EXAMPLES = [
    "What are the restaurant hours?",
    "Can I see the menu?",
    "Do you have vegetarian options?",
    "How do I make a reservation?",
    "What are your special dishes?",
    "Is there a dress code?",
    "Do you offer room service?",
    "What are your breakfast hours?"
]

SERVICE_EXAMPLES = [
    "I need housekeeping",
    "Can you send someone to fix my TV?",
    "I'd like room service",
    "The AC is not working",
    "My room needs cleaning",
    "The shower is not working",
    "Can I get extra towels?",
    "The WiFi is not connecting"
] 