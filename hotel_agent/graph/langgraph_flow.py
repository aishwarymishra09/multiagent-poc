"""
LangGraph workflow for the Hotel Agent System.
Defines the agent interaction flow and state management.
"""
import sys
import traceback
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from langgraph.graph import Graph, StateGraph
from pydantic import BaseModel, Field
from config import settings
from utils.logger import logger
from tools.hotel_tools import order_food, request_service
import matplotlib.pyplot as plt
from llm.llm_manager import llm_manager
import asyncio

class MemoryEntry(BaseModel):
    """Represents a single memory entry with metadata"""
    content: str
    role: str
    timestamp: datetime = Field(default_factory=datetime.now)
    topic: str = ""
    intent: str = ""

class MemorySummary(BaseModel):
    """Represents a summary of conversation history"""
    topic: str
    query: str
    intent: str
    summary: str
    timestamp: datetime = Field(default_factory=datetime.now)

class AgentState(BaseModel):
    messages: List[Dict[str, Any]]
    current_agent: str
    next_agent: str = "service"
    error: str = None
    confidence: float = 0.0
    requires_rag: bool = False
    rag_results: List[Dict[str, Any]] = []
    tool_results: Dict[str, Any] = {}
    requires_clarification: bool = False
    clarification_options: List[str] = []
    sub_intents: List[str] = []
    
    # Simplified memory management
    memory_entries: List[MemoryEntry] = []
    memory_summaries: List[MemorySummary] = []
    max_messages: int = 4  # Keep only last 4 messages

class HotelAssistantGraph:
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.workflow = self._create_workflow()
        self._load_persistent_memory()
    
    def _load_persistent_memory(self):
        """Load persistent memory if available"""
        try:
            # Ensure the data directory exists
            settings.MEMORY_PERSISTENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            if settings.MEMORY_PERSISTENCE_PATH.exists():
                with open(settings.MEMORY_PERSISTENCE_PATH, 'r') as f:
                    memory_data = json.load(f)
                    # Convert timestamps back to datetime objects
                    for entry in memory_data.get('entries', []):
                        entry['timestamp'] = datetime.fromisoformat(entry['timestamp'])
                    for summary in memory_data.get('summaries', []):
                        summary['timestamp'] = datetime.fromisoformat(summary['timestamp'])
                    return memory_data
        except Exception as e:
            logger.error(f"Error loading persistent memory: {str(e)}")
        return {'entries': [], 'summaries': []}

    def _save_persistent_memory(self, state: AgentState):
        """Save memory to persistent storage"""
        try:
            # Ensure the data directory exists
            settings.MEMORY_PERSISTENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            memory_data = {
                'entries': [entry.model_dump() for entry in state.memory_entries],
                'summaries': [summary.model_dump() for summary in state.memory_summaries]
            }
            with open(settings.MEMORY_PERSISTENCE_PATH, 'w') as f:
                json.dump(memory_data, f, default=str)
        except Exception as e:
            logger.error(f"Error saving persistent memory: {str(e)}")

    async def _calculate_importance(self, message: Dict[str, Any], context: List[str]) -> float:
        """Calculate importance score for a message"""
        try:
            prompt = f"""Rate the importance of this message in the context of a hotel assistant conversation.
            Consider factors like:
            - Is it a specific request or action?
            - Does it contain important user preferences?
            - Is it a critical piece of information?
            
            Message: {message['content']}
            Current context tags: {context}
            
            Rate from 0.0 to 1.0, where 1.0 is most important."""
            
            response = await llm_manager.generate_response(
                system_prompt="You are an importance scoring system. Respond only with a number between 0.0 and 1.0.",
                user_prompt=prompt
            )
            
            try:
                return float(response.strip())
            except ValueError:
                return 0.5  # Default score if parsing fails
                
        except Exception as e:
            logger.error(f"Error calculating importance: {str(e)}")
            return 0.5

    async def _extract_context_tags(self, message: Dict[str, Any]) -> List[str]:
        """Extract relevant context tags from a message"""
        try:
            prompt = f"""Extract key context tags from this message that would be relevant for future reference.
            Focus on:
            - User preferences
            - Specific requirements
            - Important details
            
            Message: {message['content']}
            
            Return only a comma-separated list of tags."""
            
            response = await llm_manager.generate_response(
                system_prompt="You are a context tag extraction system. Respond only with comma-separated tags.",
                user_prompt=prompt
            )
            
            return [tag.strip() for tag in response.split(',')]
            
        except Exception as e:
            logger.error(f"Error extracting context tags: {str(e)}")
            return []

    async def _calculate_relevance(self, entry: MemoryEntry, current_context: List[str]) -> float:
        """Calculate relevance score for a memory entry"""
        try:
            prompt = f"""Calculate the relevance of this memory entry to the current context.
            
            Memory: {entry.content}
            Memory tags: {entry.context_tags}
            Current context: {current_context}
            
            Rate from 0.0 to 1.0, where 1.0 is most relevant."""
            
            response = await llm_manager.generate_response(
                system_prompt="You are a relevance scoring system. Respond only with a number between 0.0 and 1.0.",
                user_prompt=prompt
            )
            
            try:
                return float(response.strip())
            except ValueError:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating relevance: {str(e)}")
            return 0.5

    async def _compress_memory(self, entries: List[MemoryEntry]) -> MemorySummary:
        """Compress multiple memory entries into a summary"""
        try:
            # Group entries by context tags
            tagged_entries = {}
            for entry in entries:
                for tag in entry.context_tags:
                    if tag not in tagged_entries:
                        tagged_entries[tag] = []
                    tagged_entries[tag].append(entry)
            
            # Generate summaries for each tag group
            summaries = []
            for tag, group in tagged_entries.items():
                prompt = f"""Summarize these related memories while preserving key information:
                
                Context tag: {tag}
                Memories:
                {chr(10).join([f"- {entry.content}" for entry in group])}
                
                Provide:
                1. A concise summary
                2. Key points to remember
                3. Updated context tags"""
                
                response = await llm_manager.generate_response(
                    system_prompt="You are a memory compression system. Structure your response with Summary:, Key Points:, and Tags: sections.",
                    user_prompt=prompt
                )
                
                # Parse the structured response
                sections = response.split('\n\n')
                summary = sections[0].replace('Summary:', '').strip()
                key_points = [point.strip() for point in sections[1].replace('Key Points:', '').split('\n') if point.strip()]
                tags = [tag.strip() for tag in sections[2].replace('Tags:', '').split(',') if tag.strip()]
                
                summaries.append(MemorySummary(
                    topic=tag,
                    query="",
                    intent="",
                    summary=summary,
                    relevance_score=sum(entry.relevance_score for entry in group) / len(group)
                ))
            
            # Combine summaries
            combined_summary = MemorySummary(
                topic="",
                query="",
                intent="",
                summary="\n".join([s.summary for s in summaries]),
                relevance_score=sum(s.relevance_score for s in summaries) / len(summaries)
            )
            
            return combined_summary
            
        except Exception as e:
            logger.error(f"Error compressing memory: {str(e)}")
            return MemorySummary(
                topic="",
                query="",
                intent="",
                summary="Error compressing memory",
                relevance_score=0.0
            )

    async def _manage_memory(self, state: AgentState) -> AgentState:
        """Memory management with persistent storage and simplified tracking"""
        try:
            # Process new messages
            for message in state.messages:
                if not any(entry.content == message['content'] for entry in state.memory_entries):
                    # Create new memory entry
                    entry = MemoryEntry(
                        content=message['content'],
                        role=message['role'],
                        topic=state.current_agent,  # Use current agent as topic
                        intent=state.next_agent     # Use next agent as intent
                    )
                    state.memory_entries.append(entry)
            
            # Keep only last 4 messages
            if len(state.memory_entries) > state.max_messages:
                state.memory_entries = state.memory_entries[-state.max_messages:]
            
            # Create summary of last 4 messages
            if state.memory_entries:
                # Get the most recent topic and intent
                current_topic = state.memory_entries[-1].topic
                current_intent = state.memory_entries[-1].intent
                current_query = state.messages[-1]["content"]
                
                # Create summary of last 4 messages
                summary_text = "\n".join([
                    f"{entry.role}: {entry.content}" 
                    for entry in state.memory_entries
                ])
                
                # Create new summary
                summary = MemorySummary(
                    topic=current_topic,
                    query=current_query,
                    intent=current_intent,
                    summary=summary_text
                )
                
                # Keep only the latest summary
                state.memory_summaries = [summary]
                
                # Add memory context to messages
                memory_context = (
                    f"Context:\n"
                    f"Topic: {current_topic}\n"
                    f"Query: {current_query}\n"
                    f"Intent: {current_intent}\n"
                    f"Recent messages:\n{summary_text}"
                )
                
                state.messages.insert(0, {
                    "role": "system",
                    "content": memory_context
                })
            
            # Save to persistent storage
            self._save_persistent_memory(state)
            
            return state
            
        except Exception as e:
            logger.error(f"Error in memory management: {str(e)}")
            state.error = str(e)
            return state
    
    def _create_workflow(self) -> Graph:
        """Create the LangGraph workflow"""
        try:
            logger.info("Starting workflow creation")
            workflow = StateGraph(AgentState)
            
            # Add nodes for each agent
            logger.info("Adding classifier node")
            workflow.add_node("classify_intent", self._classify_intent)
            
            # Add memory management node
            logger.info("Adding memory management node")
            workflow.add_node("memory_manager", self._manage_memory)
            
            # Add agent nodes
            for agent_name in self.agents:
                if agent_name != "classifier":
                    logger.info(f"Adding node for {agent_name} agent")
                    workflow.add_node(f"{agent_name}_agent", self._create_agent_node(agent_name))
            
            # Add tool nodes
            logger.info("Adding tool nodes")
            workflow.add_node("dining_tool", self._handle_dining_tool)
            workflow.add_node("service_tool", self._handle_service_tool)
            
            # Add response handler node
            workflow.add_node("response_handler", self._handle_response)
            
            # Add ambiguous intent handler node
            workflow.add_node("ambiguous_handler", self._handle_ambiguous_intent)
            
            # Define conditional edges from classifier to agents
            conditional_edges = {
                "service": "service_agent",
                "dining": "dining_agent",
                "service_agent": "service_agent",
                "dining_agent": "dining_agent",
                "ambiguous": "ambiguous_handler"
            }
            logger.info(f"Adding conditional edges from classifier: {conditional_edges}")
            
            # Add edge routing function that handles None and invalid cases
            def route_to_agent(state: AgentState) -> str:
                if not state.next_agent:
                    logger.warning("No next_agent specified, routing to service")
                    return "service_agent"
                
                # Handle both with and without _agent suffix
                agent_name = state.next_agent
                if not agent_name.endswith("_agent"):
                    agent_name = f"{agent_name}_agent"
                
                if agent_name not in ["service_agent", "dining_agent"]:
                    logger.warning(f"Invalid next_agent '{state.next_agent}', routing to service")
                    return "service_agent"
                
                print(f"Routing to {agent_name}")
                return agent_name
            
            workflow.add_conditional_edges(
                "classify_intent",
                route_to_agent,
                conditional_edges
            )
            
            # Replace the above edges with conditional routing
            def route_dining(state: AgentState) -> str:
                print(f"Route dining - State: {state.model_dump()}")
                print(f"Route dining - Tool results: {state.tool_results}")
                # Check if the agent has indicated a tool is needed
                if state.tool_results and "order_food" in state.tool_results:
                    print("Routing to dining_tool")
                    return "dining_tool"
                print("Routing to memory_manager")
                return "memory_manager"

            def route_service(state: AgentState) -> str:
                # Check if the agent has indicated a tool is needed
                if state.tool_results and "request_service" in state.tool_results:
                    return "service_tool"
                return "memory_manager"

            # Add conditional edges for dining agent
            workflow.add_conditional_edges(
                "dining_agent",
                route_dining,
                {
                    "dining_tool": "dining_tool",
                    "memory_manager": "memory_manager"
                }
            )

            # Add conditional edges for service agent
            workflow.add_conditional_edges(
                "service_agent",
                route_service,
                {
                    "service_tool": "service_tool",
                    "memory_manager": "memory_manager"
                }
            )

            # Add memory management after tools
            workflow.add_edge("dining_tool", "memory_manager")
            workflow.add_edge("service_tool", "memory_manager")
            workflow.add_edge("ambiguous_handler", "memory_manager")
            
            # Add edge from memory manager to response handler
            workflow.add_edge("memory_manager", "response_handler")
            
            # Set entry point
            logger.info("Setting classify_intent as entry point")
            workflow.set_entry_point("classify_intent")
            
            # Set finish point
            logger.info("Setting response_handler as finish point")
            workflow.set_finish_point("response_handler")
            
            logger.info("Compiling workflow")
            compiled_workflow = workflow.compile()
            logger.info("Workflow creation completed successfully")
            mermaid_code = compiled_workflow.get_graph().draw_mermaid()
            print(mermaid_code)
            return compiled_workflow
            
        except Exception as e:
            logger.error(f"Error creating workflow: {str(e)}")
            raise
    
    async def _handle_response(self, state: AgentState) -> AgentState:
        """Handle the final response before workflow completion"""
        try:
            logger.info("Handling final response")
            
            # If there's an error, add error message
            if state.error:
                state.messages.append({
                    "role": "assistant",
                    "content": "I apologize, but I encountered an error. Please try again or contact support."
                })
                return state
            
            # If no messages, add default response
            if not state.messages:
                state.messages.append({
                    "role": "assistant",
                    "content": "I apologize, but I couldn't process your request. Please try again."
                })
            
            # Log the final response
            logger.info(f"Final response: {state.messages[-1]['content']}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in response handler: {str(e)}")
            state.error = str(e)
            return state
    
    async def _classify_intent(self, state: AgentState) -> AgentState:
        """Classify the user's intent and route to appropriate agent"""
        try:
            # Add memory context to the classification
            memory_context = self._get_memory_context(state)
            last_message = state.messages[-1]["content"]
            
            # Include memory context in classification
            classification = await self.agents["classifier"].classify(
                last_message,
                context=memory_context
            )
            print(f"___________Classification in classify_intent: {classification}")
            
            # Validate classification result
            if not classification or not hasattr(classification, 'target_agent'):
                logger.warning("Invalid classification result received")
                state.next_agent = "service"
                state.confidence = 0.0
                state.requires_rag = False
                return state
                
            # Validate target agent exists
            if classification.target_agent not in self.agents:
                logger.warning(f"Invalid target agent '{classification.target_agent}' received from classifier")
                state.next_agent = "service"
                state.confidence = 0.0
                state.requires_rag = False
                return state
                
            # Set state with validated classification
            state.next_agent = classification.target_agent
            state.confidence = classification.confidence
            state.requires_rag = classification.requires_rag
            
            # If intent is ambiguous, route to ambiguous handler regardless of confidence
            if classification.intent == "ambiguous":
                logger.info(f"Ambiguous intent detected, routing to ambiguous handler")
                state.next_agent = "ambiguous"
                return state
            
            # For non-ambiguous intents, if confidence is too low, route to service agent
            if state.confidence < 0.5:
                logger.info(f"Low confidence classification ({state.confidence}), routing to service agent")
                state.next_agent = "service"
            
            return state
            
        except Exception as e:
            logger.error(f"Error in intent classification: {str(e)}")
            state.error = str(e)
            state.next_agent = "service"
            state.confidence = 0.0
            state.requires_rag = False
            return state

    def _get_memory_context(self, state: AgentState) -> str:
        """Get formatted memory context for use in processing"""
        if not state.memory_entries:
            return ""
        
        # Get the most recent topic and intent
        current_topic = state.memory_entries[-1].topic
        current_intent = state.memory_entries[-1].intent
        current_query = state.messages[-1]["content"] if state.messages else ""
        
        # Create summary of recent messages
        summary_text = "\n".join([
            f"{entry.role}: {entry.content}" 
            for entry in state.memory_entries
        ])
        
        return (
            f"Context:\n"
            f"Topic: {current_topic}\n"
            f"Query: {current_query}\n"
            f"Intent: {current_intent}\n"
            f"Recent messages:\n{summary_text}"
        )

    def _create_agent_node(self, agent_name: str):
        """Create a node function for an agent that properly updates state"""
        def agent_node(state: AgentState) -> AgentState:
            try:
                print(f"A1. Starting {agent_name} agent with state:", state.model_dump())
                
                # Get memory context
                memory_context = self._get_memory_context(state)
                
                # Convert state to dict for agent processing, preserving all fields
                state_dict = {
                    "messages": state.messages,
                    "current_agent": state.current_agent,
                    "next_agent": state.next_agent,
                    "error": state.error,
                    "confidence": state.confidence,
                    "requires_rag": state.requires_rag,
                    "rag_results": state.rag_results,
                    "tool_results": state.tool_results,
                    "memory_entries": [entry.model_dump() for entry in state.memory_entries],
                    "memory_summaries": [summary.model_dump() for summary in state.memory_summaries],
                    "memory_context": memory_context
                }
                print(f"A2. Converted state dict with memory context:", state_dict)
                
                # Process with the agent - use asyncio.run for async processing
                updated_state = asyncio.run(self.agents[agent_name].process(state_dict))
                print(f"A3. Agent returned state:", updated_state)
                
                # Update our state with the agent's response
                state.messages = updated_state["messages"]
                print(f"A4. Updated messages:", state.messages)
                
                if "error" in updated_state:
                    state.error = updated_state["error"]
                    print(f"A5. Updated error:", state.error)
                    
                if "rag_results" in updated_state:
                    state.rag_results = updated_state["rag_results"]
                    print(f"A6. Updated RAG results:", state.rag_results)
                
                # Update tool_results if the agent indicates a tool is needed
                if "tool_results" in updated_state:
                    print(f"A7. Updating tool results from:", state.tool_results)
                    state.tool_results = updated_state["tool_results"]
                    print(f"A7. Updated tool results to:", state.tool_results)
                
                # Preserve memory context
                if "memory_entries" in updated_state:
                    state.memory_entries = [MemoryEntry(**entry) for entry in updated_state["memory_entries"]]
                if "memory_summaries" in updated_state:
                    state.memory_summaries = [MemorySummary(**summary) for summary in updated_state["memory_summaries"]]
                
                print(f"A8. Final state after {agent_name}:", state.model_dump())
                return state
                
            except Exception as e:
                exc_type, exc_obj, tb = sys.exc_info()
                fname = tb.tb_frame.f_code.co_filename
                lineno = tb.tb_lineno
                print(f"A9. Error in {agent_name} agent:", str(e), "at", fname, "line", lineno)
                logger.error(f"Error in {agent_name} agent: {str(e)}")
                state.error = str(e)
                return state
        
        return agent_node
    
    async def _handle_dining_tool(self, state: AgentState) -> AgentState:
        """Handle dining tool operations"""
        try:
            print(f"Dining tool - Initial state: {state.model_dump()}")
            print(f"Dining tool - Tool results: {state.tool_results}")
            
            # Check if we have an order in tool_results
            if state.tool_results and "order_food" in state.tool_results:
                print("Processing order in dining tool")
                order_details = state.tool_results["order_food"]
                print(f"Order details: {order_details}")
                
                # Record the order
                order_result = order_food(
                    food_name=order_details["food_name"],
                    price=order_details["price"],
                    food_id=order_details["food_id"]
                )
                print(f"Order result: {order_result}")
                
                if order_result["status"] == "success":
                    response = f"I've placed your order for {order_details['food_name']}. "
                    if order_details.get("price"):
                        response += f"The price is {order_details['price']} {order_details.get('currency', 'EURO')}. "
                    response += "Your order will be prepared shortly."
                    
                    state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                else:
                    state.messages.append({
                        "role": "assistant",
                        "content": "I apologize, but I couldn't process your order. Please try again."
                    })
            else:
                print("No order found in tool_results")
                # If no order in tool_results, check if this is a new order request
                query = state.messages[-1]["content"]
                if "order" in query.lower() or "want" in query.lower() or "get" in query.lower():
                    if not state.rag_results:
                        state.messages.append({
                            "role": "assistant",
                            "content": "I couldn't find the item you're looking for. Could you please specify what you'd like to order?"
                        })
                        return state
            
            print(f"Dining tool - Final state: {state.model_dump()}")
            return state
            
        except Exception as e:
            logger.error(f"Error in dining tool: {str(e)}")
            state.error = str(e)
            return state
    
    async def _handle_service_tool(self, state: AgentState) -> AgentState:
        """Handle service tool operations"""
        try:
            # Get memory context
            memory_context = self._get_memory_context(state)
            query = state.messages[-1]["content"]
            
            # Determine service type and details
            service_type = None
            notes = None
            quantity = 1
            
            if "housekeeping" in query.lower() or "cleaning" in query.lower():
                service_type = "housekeeping"
                notes = "Room cleaning requested"
            elif "maintenance" in query.lower() or "repair" in query.lower():
                service_type = "maintenance"
                notes = "Maintenance/repair requested"
            elif "room service" in query.lower():
                service_type = "room_service"
                notes = "Room service requested"
            
            # If we identified a service type, process the request
            if service_type:
                service_result = request_service(
                    service_name=service_type,
                    notes=notes,
                    quantity=quantity
                )
                
                if service_result["status"] == "success":
                    response = f"I've submitted your {service_type} request. "
                    if service_type == "housekeeping":
                        response += "Housekeeping will attend to your room shortly."
                    elif service_type == "maintenance":
                        response += "A maintenance technician will be dispatched to your room. Please provide your room number."
                    elif service_type == "room_service":
                        response += "What would you like to order from room service?"
                    
                    state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    state.tool_results = {"request_service": service_result}
                else:
                    state.messages.append({
                        "role": "assistant",
                        "content": "I apologize, but I couldn't process your service request. Please try again."
                    })
            
            return state
            
        except Exception as e:
            logger.error(f"Error in service tool: {str(e)}")
            state.error = str(e)
            return state
    
    async def _handle_ambiguous_intent(self, state: AgentState) -> AgentState:
        """Handle ambiguous intents by providing clarification"""
        try:
            # Get memory context
            memory_context = self._get_memory_context(state)
            query = state.messages[-1]["content"]
            
            # Create context for ambiguous handler
            context = {
                "query": query,
                "memory_context": memory_context,
                "messages": state.messages,
                "previous_intents": [entry.content for entry in state.memory_entries if entry.role == "system"]
            }
            
            # Handle ambiguous intent
            from intent_classifier.ambiguous_handler import AmbiguousIntentHandler
            handler = AmbiguousIntentHandler()
            result = await handler.handle_ambiguous(state.classification, context)
            
            # Update state with clarification
            state.messages = result["messages"]
            state.requires_clarification = result["requires_clarification"]
            state.clarification_options = result["clarification_options"]
            state.sub_intents = result["sub_intents"]
            
            return state
            
        except Exception as e:
            logger.error(f"Error handling ambiguous intent: {str(e)}")
            state.error = str(e)
            return state
    
    async def process_message(self, message: str) -> str:
        """Process a user message through the workflow"""
        try:
            print("1. Starting process_message with:", message)
            
            # Load any existing memory
            memory_data = self._load_persistent_memory()
            
            initial_state = AgentState(
                messages=[{"role": "user", "content": message}],
                current_agent="classify_intent",
                memory_entries=[MemoryEntry(**entry) for entry in memory_data.get('entries', [])],
                memory_summaries=[MemorySummary(**summary) for summary in memory_data.get('summaries', [])]
            )
            print("2. Initial state created:", initial_state.model_dump())
            
            print("3. About to invoke workflow")
            final_state = await self.workflow.ainvoke(initial_state)

            # Check for error first
            if final_state.get("error"):
                print("5. Error found in final state:", final_state.get("error"))
                logger.error(f"Error in workflow: {final_state.get('error')}")
                # Add error message to the conversation
                final_state.get("messages").append({
                    "role": "assistant",
                    "content": "I apologize, but I encountered an error. Please try again or contact support."
                })
                return final_state.get('messages')[-1]["content"]
            
            # Get the last message from the state
            if final_state.get('messages') and len(final_state.get('messages')) > 0:
                print("6. Returning last message:", final_state.get('messages')[-1]["content"])
                return final_state.get('messages')[-1]["content"]
            else:
                print("7. No messages in final state")
                return "I apologize, but I couldn't process your request. Please try again."
            
        except Exception as e:
            exc_type, exc_obj, tb = sys.exc_info()
            fname = tb.tb_frame.f_code.co_filename
            lineno = tb.tb_lineno
            full_traceback = traceback.format_exc()

            print(f"❌ Exception caught: {e} at {fname} line {lineno}")
            print("📄 Full traceback:")
            print(full_traceback)

            logger.error(f"Error: {e} (File: {fname}, Line: {lineno})")
            logger.error("Full traceback:\n%s", full_traceback)
            return "I apologize, but I encountered an error. Please try again or contact support." 