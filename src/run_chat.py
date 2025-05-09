from agent.hotel_assistants import app

def main():
    # Initialize the chat with a user ID
    config = {"configurable": {"user_id": "user_1"}}
    
    # Start the conversation
    print("Welcome to the Hotel Assistant System!")
    print("You can ask about:")
    print("- Hotel services (cleaning, laundry, concierge)")
    print("- Dining options and reservations")
    print("- Hotel facilities and general information")
    print("Type 'exit' to quit.")
    
    # Initialize the chat state
    state = {"messages": []}
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
            
        # Add user message to state
        state["messages"].append({"role": "user", "content": user_input})
        
        # Get response from the agent
        response = app.invoke(state, config)
        
        # Get the last message content
        last_message = response["messages"][-1]
        message_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        # Add assistant response to state
        state["messages"].append({"role": "assistant", "content": message_content})
        
        # Print the response
        print("\nAssistant:", message_content)

if __name__ == "__main__":
    main() 