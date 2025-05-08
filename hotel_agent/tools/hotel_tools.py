"""
Hotel tools for handling dining orders and service requests.
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional

# Constants for file paths
DINING_ORDERS_FILE = "data/dining_orders.xlsx"
SERVICE_REQUESTS_FILE = "data/service_requests.xlsx"

def ensure_data_directory():
    """Ensure the data directory exists and is writable."""
    try:
        data_dir = Path("data")
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
        
        # Test if directory is writable
        test_file = data_dir / ".test_write"
        try:
            test_file.touch()
            test_file.unlink()
        except (PermissionError, OSError) as e:
            raise PermissionError(f"Data directory is not writable: {str(e)}")
            
    except Exception as e:
        raise PermissionError(f"Failed to create/verify data directory: {str(e)}")

def order_food(food_name: str, price: float, food_id: str) -> dict:
    """
    Record a food order in the Excel file.
    
    Args:
        food_name (str): Name of the food item
        price (float): Price of the food item
        food_id (str): Unique identifier for the food item
        
    Returns:
        dict: Status of the operation
    """
    try:
        # Ensure data directory exists and is writable
        ensure_data_directory()
        
        # Create or load the Excel file
        try:
            if Path(DINING_ORDERS_FILE).exists():
                # Test if file is writable
                try:
                    with open(DINING_ORDERS_FILE, 'a'):
                        pass
                except PermissionError:
                    raise PermissionError(f"Excel file exists but is not writable: {DINING_ORDERS_FILE}")
                df = pd.read_excel(DINING_ORDERS_FILE)
            else:
                df = pd.DataFrame(columns=['timestamp', 'food_name', 'price', 'food_id'])
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to access Excel file: {str(e)}"
            }
        
        # Add new order
        new_order = {
            'timestamp': datetime.now(),
            'food_name': food_name,
            'price': price,
            'food_id': food_id
        }
        
        try:
            df = pd.concat([df, pd.DataFrame([new_order])], ignore_index=True)
            df.to_excel(DINING_ORDERS_FILE, index=False)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to write order to Excel file: {str(e)}"
            }
        
        return {
            "status": "success",
            "message": f"Order recorded for {food_name}",
            "order_details": new_order
        }
        
    except PermissionError as e:
        return {
            "status": "error",
            "message": f"Permission error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to record order: {str(e)}"
        }

def request_service(service_name: str, notes: Optional[str] = None, quantity: int = 1) -> dict:
    """
    Record a service request in the Excel file.
    
    Args:
        service_name (str): Name of the service requested
        notes (str, optional): Additional notes about the service request
        quantity (int): Quantity of the service requested
        
    Returns:
        dict: Status of the operation
    """
    try:
        ensure_data_directory()
        
        # Create or load the Excel file
        try:
            df = pd.read_excel(SERVICE_REQUESTS_FILE)
        except FileNotFoundError:
            df = pd.DataFrame(columns=['timestamp', 'service_name', 'notes', 'quantity'])
        
        # Add new service request
        new_request = {
            'timestamp': datetime.now(),
            'service_name': service_name,
            'notes': notes,
            'quantity': quantity
        }
        
        df = pd.concat([df, pd.DataFrame([new_request])], ignore_index=True)
        df.to_excel(SERVICE_REQUESTS_FILE, index=False)
        
        return {
            "status": "success",
            "message": f"Service request recorded for {service_name}",
            "request_details": new_request
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to record service request: {str(e)}"
        } 