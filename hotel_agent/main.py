"""
Main entry point for the Hotel Agent System.
This file initializes and runs the LangGraph workflow.
"""

import asyncio
from graph.langgraph_flow import HotelAssistantGraph
from agents.dining_agent import DiningAgent
from agents.service_agent import ServiceAgent
from intent_classifier.router import IntentClassifier
from utils.logger import setup_logger

logger = setup_logger()

async def main():
    # Initialize components
    agents = {
        "dining": DiningAgent(),
        "service": ServiceAgent(),
        "classifier": IntentClassifier()
    }
    
    # Create and run the graph
    assistant = HotelAssistantGraph(agents)
    
    # Example usage
    while True:
        try:
            user_input = input("\nEnter your query (or 'quit' to exit): ")
            
            # Handle empty input or whitespace-only input
            if not user_input or user_input.isspace():
                print("\nPlease enter a valid message. Empty messages are not allowed.")
                continue
                
            if user_input.lower() == 'quit':
                break
            
            response = await assistant.process_message(user_input)
            print(f"\nAssistant: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            print("\nI apologize, but I encountered an error. Please try again.")

if __name__ == "__main__":
    asyncio.run(main()) 