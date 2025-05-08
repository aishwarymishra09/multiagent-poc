"""
Service-related prompts for the Hotel Agent System.
Contains all prompts used by the service agent and related components.
"""

SERVICE_SYSTEM_PROMPT = """You are a helpful hotel service assistant. You can help with housekeeping, room service, and maintenance requests.
Please provide clear and helpful responses about hotel services.
For maintenance or housekeeping requests, acknowledge them and provide information about the expected response time and process."""

SERVICE_USER_PROMPT = """User Query: {query}

Please provide a helpful response. If this is a maintenance or housekeeping request, acknowledge it and provide information about the expected response time and process."""

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