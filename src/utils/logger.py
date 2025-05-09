"""
Logging utility for the Hotel Agent System.
Provides centralized logging configuration.
"""

import logging
import sys
from collections import defaultdict
from pathlib import Path
from config import settings

def setup_logger():
    """Configure and return a logger instance"""
    # Create logs directory if it doesn't exist
    log_path = Path(settings.LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(settings.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("hotel_agent")

def log_error_with_location(logger, e):
    """Log an error with the exact file location and line number where it occurred.
    
    Args:
        logger: The logger instance to use
        e: The exception that was raised
    """
    exc_type, exc_obj, tb = sys.exc_info()
    fname = tb.tb_frame.f_code.co_filename
    lineno = tb.tb_lineno
    logger.error(f"Error: {e} (File: {fname}, Line: {lineno})")

logger = setup_logger()
HOTEL_INFO = defaultdict(lambda: {
        "service_requests": [],
        "dining_reservations": [],
        "general_info_queries": []
    })