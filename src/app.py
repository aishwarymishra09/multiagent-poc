import streamlit as st
from agent.hotel_assistants import app
from utils.logger import HOTEL_INFO
import datetime
from typing import Dict, List
from langchain_core.messages import AIMessage, HumanMessage
import json

# Page config
st.set_page_config(
    page_title="Hotel Assistant",
    page_icon="🏨",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_id" not in st.session_state:
    st.session_state.user_id = "user_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Initialize user info in HOTEL_INFO
    HOTEL_INFO[st.session_state.user_id] = {
        "service_requests": [],
        "dining_reservations": [],
        "chat_history": []
    }

def format_message(message: Dict) -> str:
    """Format a message for display in the chat."""
    if isinstance(message, AIMessage):
        return f"🏨 Assistant: {message.content}"
    return f"👤 You: {message.content}"

def save_chat_history():
    """Save chat history to user's info."""
    # Convert messages to serializable format
    serializable_messages = []
    for msg in st.session_state.messages:
        if isinstance(msg, (AIMessage, HumanMessage)):
            serializable_messages.append({
                "role": "assistant" if isinstance(msg, AIMessage) else "user",
                "content": msg.content
            })
        else:
            serializable_messages.append(msg)
    HOTEL_INFO[st.session_state.user_id]["chat_history"] = serializable_messages

def load_chat_history():
    """Load chat history from user's info."""
    if "chat_history" in HOTEL_INFO[st.session_state.user_id]:
        # Convert stored messages back to LangChain message objects
        loaded_messages = []
        for msg in HOTEL_INFO[st.session_state.user_id]["chat_history"]:
            if msg["role"] == "assistant":
                loaded_messages.append(AIMessage(content=msg["content"]))
            else:
                loaded_messages.append(HumanMessage(content=msg["content"]))
        st.session_state.messages = loaded_messages

# Sidebar
with st.sidebar:
    st.title("🏨 Hotel Assistant")
    st.markdown("""
    Welcome to the Hotel Assistant! I can help you with:
    - 🍽️ Dining and restaurant information
    - 🧹 Room service and housekeeping
    - ℹ️ Hotel facilities and general information
    """)
    
    # Display user's active requests
    st.subheader("Your Active Requests")
    user_info = HOTEL_INFO[st.session_state.user_id]
    
    if user_info["service_requests"]:
        st.write("Service Requests:")
        for req in user_info["service_requests"]:
            st.write(f"- {req['service_name']} (Status: {req['status']})")
    
    if user_info["dining_reservations"]:
        st.write("Dining Reservations:")
        for res in user_info["dining_reservations"]:
            if isinstance(res, dict) and "type" in res and res["type"] == "food_order":
                st.write(f"- Food Order: {', '.join(item['name'] for item in res['items'])}")
            else:
                st.write(f"- Reservation at {res.get('restaurant', {}).get('name', 'Unknown')}")

# Main chat interface
st.title("Chat with Hotel Assistant")

# Load chat history
load_chat_history()

# Display chat messages with alignment and color
for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        st.markdown(f'''
        <div class="assistant-bubble">{message.content}</div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="user-bubble">{message.content}</div>
        ''', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("How can I help you today?"):
    # Create a HumanMessage for the user's input
    user_message = HumanMessage(content=prompt)
    
    # Add user message to chat history
    st.session_state.messages.append(user_message)
    
    # Display user message
    st.markdown(f'''<div class="user-bubble">{prompt}</div>''', unsafe_allow_html=True)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Prepare the input for the agent
            agent_input = {
                "messages": st.session_state.messages,
                "configurable": {
                    "user_id": st.session_state.user_id
                }
            }
            
            # Get response from the agent
            response = app.invoke(agent_input)
            
            # The response contains AIMessage objects
            if isinstance(response, dict) and "messages" in response:
                # Get the last message from the response
                assistant_message = response["messages"][-1]
                if isinstance(assistant_message, AIMessage):
                    # Display assistant response
                    st.markdown(f'''<div class="assistant-bubble">{assistant_message.content}</div>''', unsafe_allow_html=True)
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append(assistant_message)
                else:
                    st.error("Unexpected response format from assistant")
            else:
                st.error("Failed to get response from assistant")
    
    # Save updated chat history
    save_chat_history()

# Improved CSS for alignment and color
st.markdown("""
<style>
.user-bubble {
    background-color: #e3f0ff;
    color: #222;
    padding: 1rem 1.5rem;
    border-radius: 1.5rem 1.5rem 0.2rem 1.5rem;
    margin-bottom: 1.2rem;
    margin-left: 30%;
    margin-right: 0;
    text-align: right;
    max-width: 60%;
    float: right;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.assistant-bubble {
    background-color: #fffbe6;
    color: #222;
    padding: 1rem 1.5rem;
    border-radius: 1.5rem 1.5rem 1.5rem 0.2rem;
    margin-bottom: 1.2rem;
    margin-right: 30%;
    margin-left: 0;
    text-align: left;
    max-width: 60%;
    float: left;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.stChatInput {
    position: fixed;
    left: 0;
    right: 0;
    bottom: 3rem;
    margin-left: auto;
    margin-right: auto;
    width: 60% !important;
    max-width: 700px;
    min-width: 320px;
    background-color: white;
    padding: 1rem;
    border-top: 1px solid #e6e6e6;
    border-radius: 2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    z-index: 100;
}
</style>
""", unsafe_allow_html=True) 