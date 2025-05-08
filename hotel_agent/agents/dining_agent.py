"""
Dining agent for the Hotel Agent System.
Handles restaurant-related queries using RAG.
"""
import sys
from typing import Dict, Any, List
from config import settings
from utils.logger import logger
from rag.dining_rag import DiningRAG
from llm.llm_manager import llm_manager
from prompts.dining_prompts import DINING_SYSTEM_PROMPT, DINING_USER_PROMPT

class DiningAgent:
    def __init__(self):
        self.rag = DiningRAG()
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process a dining-related query"""
        try:
            print("______ state:{}".format(state))
            query = state["messages"][-1]["content"]
            
            # Get relevant context from RAG
            rag_results = await self.rag.query(query)
            if not rag_results:
                state["error"] = "No relevant information found for your query"
                return state
            
            # Check if this is an order request
            is_order = any(word in query.lower() for word in ["order", "want", "get", "have", "bring"])
            
            if is_order and rag_results:
                # Set up tool results for ordering
                food_item = rag_results[0]
                state["tool_results"] = {
                    "order_food": {
                        "food_name": food_item["name"],
                        "price": food_item.get("price", 0.0),
                        "food_id": food_item.get("id", ""),
                        "currency": food_item.get("currency", "EURO")
                    }
                }
                print("_______Dining Agent____tool_results:{}".format(state["tool_results"]))
            
            # Generate response using LLM manager
            response = await llm_manager.generate_response(
                system_prompt=DINING_SYSTEM_PROMPT,
                user_prompt=DINING_USER_PROMPT.format(query=query),
                context=rag_results
            )
            
            if not response:
                state["error"] = "Failed to generate response"
                return state
            
            # Update state
            state["messages"].append({
                "role": "assistant",
                "content": response
            })
            state["rag_results"] = rag_results
            print("_______Dining Agent____final state:{}".format(state))
            return state
            
        except Exception as e:
            exc_type, exc_obj, tb = sys.exc_info()
            fname = tb.tb_frame.f_code.co_filename
            lineno = tb.tb_lineno
            logger.error(f"Error: {e} (File: {fname}, Line: {lineno})")
            state["error"] = str(e)
            return state 