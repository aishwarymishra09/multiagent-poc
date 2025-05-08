"""
Service agent for the Hotel Agent System.
Handles housekeeping and maintenance requests.
"""

from typing import Dict, Any
from config import settings
from utils.logger import logger
from llm.llm_manager import llm_manager
from prompts.service_prompts import SERVICE_SYSTEM_PROMPT, SERVICE_USER_PROMPT

class ServiceAgent:
    def __init__(self):
        pass
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process a service-related query"""
        try:
            query = state["messages"][-1]["content"]
            
            # Generate response using LLM manager
            response = await llm_manager.generate_response(
                system_prompt=SERVICE_SYSTEM_PROMPT,
                user_prompt=SERVICE_USER_PROMPT.format(query=query)
            )
            
            # Update state
            state["messages"].append({
                "role": "assistant",
                "content": response
            })
            
            return state
            
        except Exception as e:
            exc_type, exc_obj, tb = sys.exc_info()
            fname = tb.tb_frame.f_code.co_filename
            lineno = tb.tb_lineno
            logger.error(f"Error: {e} (File: {fname}, Line: {lineno})")
            state["error"] = str(e)
            return state 