from typing import List, Dict
from langchain_core.runnables import RunnableConfig
import datetime
from .dining_rag2 import DiningRAG
from utils.logger import HOTEL_INFO
# Mock menu data structure
MENU_ITEMS = [
    {
        "id": "M001",
        "name": "Classic Burger",
        "description": "Juicy beef patty with lettuce, tomato, and special sauce",
        "price": 15.99,
        "category": "Main Course",
        "dietary_info": ["Contains Gluten", "Contains Dairy"]
    },
    {
        "id": "M002",
        "name": "Caesar Salad",
        "description": "Fresh romaine lettuce with parmesan and croutons",
        "price": 12.99,
        "category": "Appetizer",
        "dietary_info": ["Vegetarian", "Contains Dairy"]
    },
    # Add more menu items as needed
]

def fetch_menu_details(
    query: str,
    config: RunnableConfig,
) -> List[Dict]:
    """
    Fetch menu details using RAG from OpenSearch.
    
    Args:
        query: The search query to find relevant menu items
        config: RunnableConfig for additional configuration
    
    Returns:
        List of relevant menu items with their details
    """
    try:
        # Initialize DiningRAG
        dining_rag = DiningRAG()
        
        # Search for relevant menu items
        results = dining_rag.query(query)
        print("___________results:{}".format(results))
        # Process and return results
        menu_items = []
        for doc in results:
            # Extract relevant information from the document
            menu_items.append({
                "id": doc.get("id"),
                "name": doc.get("name"),
                "description": doc.get("description"),
                "price": doc.get("price"),
                "category": doc.get("category")
            })
        
        return menu_items
    
    except Exception as e:
        print(f"Error fetching menu details: {str(e)}")
        # Fallback to mock data if RAG is not available
        return [item for item in MENU_ITEMS if query.lower() in item["name"].lower() or query.lower() in item["description"].lower()]

def order_food_items(
    items: List[Dict],
    config: RunnableConfig
) -> str:
    """Place an order for food items from the hotel's menu.
    
    This function processes food orders, calculates totals, and stores order history.
    It can be called when a user wants to order food items from the menu or when a user wants to order food items mentioned before in message from menu.
    
    Args:
        items: List of food items to order. Each item should be a dictionary containing:
            - id: Unique identifier for the menu item
            - name: Name of the food item
            - price: Price of the item (numeric value)
            - description: Description of the food item
            - quantity: (optional) Number of items to order, defaults to 1
            - special_instructions: (optional) Any special preparation instructions
        config: RunnableConfig containing user configuration and context
    
    Returns:
        str: A formatted confirmation message containing:
            - Order status
            - List of ordered items with their prices
            - Total order amount
            - Order timestamp
    
    Example:
        >>> items = [
        ...     {"id": "M001", "name": "Classic Burger", "price": 15.99, "description": "Juicy beef patty"},
        ...     {"id": "M002", "name": "Caesar Salad", "price": 12.99, "description": "Fresh romaine lettuce"}
        ... ]
        >>> order_food_items(items, config)
        'Order placed successfully!
        
        Order Summary:
        - Classic Burger ($15.99)
        - Caesar Salad ($12.99)
        
        Total: $28.98'
    """
    print("inside food order too")
    user_id = config["configurable"].get("user_id")
    
    # Validate items
    if not items:
        return "No items provided for ordering"
    
    # Calculate total
    total = sum(int(item["price"]) for item in items)
    
    # Create order summary
    order_summary = "\n".join([
        f"- {item['name']} (${item['price']:.2f})"
        for item in items
    ])
    
    # Store order in user's dining reservations

    HOTEL_INFO[user_id]["dining_reservations"].append({
        "type": "food_order",
        "items": items,
        "total": total,
        "timestamp": datetime.datetime.now().isoformat()
    })
    
    return f"Order placed successfully!\n\nOrder Summary:\n{order_summary}\n\nTotal: ${total:.2f}" 