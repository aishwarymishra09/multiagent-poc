import datetime
from typing import Callable, List, Dict, Optional
import os
from utils.logger import HOTEL_INFO
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from .dining_tools import fetch_menu_details, order_food_items
from .general_info_tools import search_facilities
from config import settings

# Set LangSmith environment variables
os.environ["LANGCHAIN_TRACING_V2"] = str(settings.LANGCHAIN_TRACING_V2)
os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT

model = ChatOpenAI(model="gpt-4o-mini")

# Mock data for tools
HOTEL_INFO = HOTEL_INFO

# Mock data for services
SERVICES = [
    {
        "id": "1",
        "name": "Room Cleaning",
        "description": "Standard room cleaning service",
        "duration": "30 minutes",
        "available_hours": "8:00 AM - 8:00 PM"
    },
    {
        "id": "2",
        "name": "Laundry Service",
        "description": "Professional laundry and dry cleaning",
        "duration": "24 hours",
        "available_hours": "24/7"
    },
    {
        "id": "3",
        "name": "Concierge",
        "description": "Personal concierge service",
        "duration": "On-demand",
        "available_hours": "24/7"
    }
]

# Mock data for dining
RESTAURANTS = [
    {
        "id": "1",
        "name": "Skyline Restaurant",
        "cuisine": "International",
        "hours": "7:00 AM - 10:00 PM",
        "capacity": 100
    },
    {
        "id": "2",
        "name": "Poolside Grill",
        "cuisine": "American",
        "hours": "11:00 AM - 6:00 PM",
        "capacity": 50
    },
    {
        "id": "3",
        "name": "Lobby Lounge",
        "cuisine": "Bar & Snacks",
        "hours": "4:00 PM - 12:00 AM",
        "capacity": 75
    }
]

# Mock data for general info
HOTEL_FACILITIES = [
    {
        "id": "1",
        "name": "Swimming Pool",
        "location": "Level 5",
        "hours": "7:00 AM - 10:00 PM",
        "features": ["Indoor", "Heated", "Jacuzzi"]
    },
    {
        "id": "2",
        "name": "Fitness Center",
        "location": "Level 4",
        "hours": "24/7",
        "features": ["Cardio", "Weights", "Yoga Studio"]
    },
    {
        "id": "3",
        "name": "Business Center",
        "location": "Level 2",
        "hours": "8:00 AM - 8:00 PM",
        "features": ["Computers", "Printing", "Meeting Rooms"]
    }
]

# Service tools
def search_services(service_type: str = None) -> list[dict]:
    """Search available hotel services.
    
    Args:
        service_type: Optional filter for specific type of service
    """
    if service_type:
        return [s for s in SERVICES if service_type.lower() in s["name"].lower()]
    return SERVICES

def request_service(
    service_name: str,
    config: RunnableConfig,
    notes: str = "",
    quantity: int = 1,
) -> str:
    """Request any hotel service or item from room service.
    
    This function handles any type of service request that a hotel guest might need,
    from room service items to housekeeping services. It can process requests for:
    - Room service items (food, drinks, amenities)
    - Housekeeping services (cleaning, turndown)
    - Laundry services
    - Additional amenities (towels, slippers, toiletries)
    - Special requests (extra pillows, room temperature adjustment)
    
    The service name should be extracted directly from the user's message/chat.
    For example:
    - "I need some extra towels"
    - "Can you bring me a bottle of water?"
    - "Please clean my room"
    - "I'd like to order room service"
    
    Args:
        service_name: The service or item being requested, extracted from user's message
                     (e.g., "extra towels", "bottle of water", "room cleaning")
        notes: Additional details or special instructions for the request
              (e.g., "bring 2 bottles", "clean the bathroom first")
        quantity: Number of items or times the service is requested
        config: RunnableConfig containing user configuration
    
    Returns:
        str: A confirmation message containing:
            - Service request status
            - Service/item details
            - Quantity requested
            - Any special notes
            - Estimated time of service (if applicable)
    
    Example:
        >>> request_service("extra towels", config, notes="2 bath towels and 1 hand towel", quantity=3)
        'Successfully requested 3x extra towels service. Notes: 2 bath towels and 1 hand towel'
        
        >>> request_service("bottle of water", config, quantity=2)
        'Successfully requested 2x bottle of water service. Notes: '
    """
    try:
        user_id = config["configurable"].get("user_id")
        
        # Create a service request record
        service_request = {
            "service_name": service_name,
            "notes": notes,
            "quantity": quantity,
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "pending"
        }
        
        # Store the request
        HOTEL_INFO[user_id]["service_requests"].append(service_request)
        return f"Successfully requested {quantity}x {service_name} service. Notes: {notes}"
    except Exception as e:
        print(f"Error in service request: {str(e)}")
        return f"Failed to request service: {str(e)}"

# Dining tools
def search_restaurants(cuisine: str = None) -> list[dict]:
    """Search hotel restaurants.
    
    Args:
        cuisine: Optional filter for specific cuisine type
    """
    if cuisine:
        return [r for r in RESTAURANTS if cuisine.lower() in r["cuisine"].lower()]
    return RESTAURANTS

def make_reservation(
    restaurant_id: str,
    time: str,
    party_size: int,
    config: RunnableConfig,
) -> str:
    """Make a restaurant reservation."""
    user_id = config["configurable"].get("user_id")
    restaurant = [r for r in RESTAURANTS if r["id"] == restaurant_id][0]
    reservation = {
        "restaurant": restaurant,
        "time": time,
        "party_size": party_size
    }
    HOTEL_INFO[user_id]["dining_reservations"].append(reservation)
    return f"Successfully made reservation at {restaurant['name']} for {party_size} people at {time}"

# Define handoff tools
transfer_to_service_assistant = create_handoff_tool(
    agent_name="service_assistant",
    description="""Transfer to the service assistant for any room service, housekeeping, or amenity requests.
    This includes:
    - Room cleaning and housekeeping services
    - Laundry and dry cleaning
    - Additional amenities (towels, toiletries, pillows)
    - Room service items (water, snacks, etc.)
    - Maintenance requests
    - Special room arrangements
    - Any other service-related requests""",
)

transfer_to_dining_assistant = create_handoff_tool(
    agent_name="dining_assistant",
    description="""Transfer to the dining assistant for any food, beverage, or restaurant-related requests.
    This includes:
    - Restaurant reservations and dining options
    - Menu inquiries and food ordering
    - Special dietary requirements
    - Room service food orders
    - Bar and beverage services
    - Restaurant hours and availability
    - Any other food and dining related queries""",
)

transfer_to_general_info_assistant = create_handoff_tool(
    agent_name="general_info_assistant",
    description="""Transfer to the general information assistant for any hotel facility or general information queries.
    This includes:
    - Hotel facilities and amenities
    - Swimming pool and fitness center
    - Business center and meeting rooms
    - Hotel policies and check-in/out times
    - Local attractions and transportation
    - General hotel information
    - Any other non-service, non-dining related queries""",
)

# Define agent prompt
def make_prompt(base_system_prompt: str) -> Callable[[dict, RunnableConfig], list]:
    def prompt(state: dict, config: RunnableConfig) -> list:
        user_id = config["configurable"].get("user_id")
        current_info = HOTEL_INFO[user_id]
        system_prompt = (
            base_system_prompt
            + "\n\nIMPORTANT RULES:"
            + "\n1. NEVER invent or make up information about menus, services, or facilities"
            + "\n2. ALWAYS use the provided tools to fetch accurate information"
            + "\n3. If a tool doesn't return information, respond with 'I don't have that information available'"
            + "\n4. Don't assume or guess about prices, availability, or features"
            + "\n5. If unsure, transfer to the appropriate specialist agent"
            + "\n6. Always verify information through tools before providing it to the user"
            + f"\n\nUser's current information: {current_info}"
            + f"\nToday is: {datetime.datetime.now()}"
        )
        return [{"role": "system", "content": system_prompt}] + state["messages"]

    return prompt

# Define agents
service_assistant = create_react_agent(
    model,
    [search_services, request_service, transfer_to_dining_assistant, transfer_to_general_info_assistant],
    prompt=make_prompt("You are a hotel service assistant. You can help with room service, cleaning, laundry, and other hotel services."),
    name="service_assistant",
)

dining_assistant = create_react_agent(
    model,
    [fetch_menu_details, order_food_items, transfer_to_service_assistant, transfer_to_general_info_assistant],
    prompt=make_prompt("""You are a hotel dining assistant. You can help with menu queries, food ordering, and provide information about dining options.
    IMPORTANT: Only provide menu information that you can verify through the fetch_menu_details tool.
    If a menu item is not found in the search results, inform the user that you don't have that information.
    Never make assumptions about menu items, prices, or availability."""),
    name="dining_assistant",
)

general_info_assistant = create_react_agent(
    model,
    [search_facilities, transfer_to_service_assistant, transfer_to_dining_assistant],
    prompt=make_prompt("""You are a hotel general information assistant. You can provide information about hotel facilities, amenities, and general hotel information.
    IMPORTANT: Only provide facility information that you can verify through the search_facilities.
    If information is not found in the search results, inform the user that you don't have that information.
    Never make assumptions about facilities, features, or availability."""),
    name="general_info_assistant",
)

# Compile and run!
builder = create_swarm(
    [service_assistant, dining_assistant, general_info_assistant],
    default_active_agent="general_info_assistant"
)
app = builder.compile() 