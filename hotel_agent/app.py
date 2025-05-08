import streamlit as st
from graph.langgraph_flow import HotelAssistantGraph
from agents.dining_agent import DiningAgent
from agents.service_agent import ServiceAgent
from intent_classifier.router import IntentClassifier
from intent_classifier.ambiguous_handler import AmbiguousIntentHandler
import asyncio
import os
os.environ["LANGCHAIN_TRACING_V2"] = str(st.secrets["LANGSMITH_TRACING"]).lower()
print(os.environ["LANGCHAIN_TRACING_V2"])
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGSMITH_ENDPOINT"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGSMITH_PROJECT"]
# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the hotel assistant
@st.cache_resource
def get_hotel_assistant():
    # Initialize components
    agents = {
        "dining": DiningAgent(),
        "service": ServiceAgent(),
        "classifier": IntentClassifier(),
        "ambiguous": AmbiguousIntentHandler()
    }
    return HotelAssistantGraph(agents)

def main():
    st.title("🏨 Hotel Assistant")
    st.markdown("""
    Welcome to the Hotel Assistant! I can help you with:
    - Dining reservations and menu information
    - Hotel services and amenities
    - General inquiries about your stay
    
    How can I assist you today?
    """)

    # Initialize the hotel assistant
    hotel_assistant = get_hotel_assistant()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Run the async process_message function
                    response = asyncio.run(hotel_assistant.process_message(prompt))
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_message = "I apologize, but I encountered an error. Please try again."
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main() 