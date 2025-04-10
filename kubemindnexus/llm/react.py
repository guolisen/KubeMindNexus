"""ReAct loop implementation for KubeMindNexus."""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from ..config import Configuration
from ..constants import REACT_MAX_ITERATIONS, REACT_SAFETY_TIMEOUT
from ..database import DatabaseManager
from ..mcp.hub import MCPHub
from .base import BaseLLM, LLMMessage, MessageRole

logger = logging.getLogger(__name__)


class ReactLoop:
    """ReAct loop for LLM interaction with MCP tools."""
    
    def __init__(
        self,
        config: Configuration,
        db_manager: DatabaseManager,
        mcp_hub: MCPHub,
        llm: BaseLLM,
    ):
        """Initialize ReAct loop.
        
        Args:
            config: Configuration instance.
            db_manager: Database manager instance.
            mcp_hub: MCP hub instance.
            llm: LLM instance.
        """
        self.config = config
        self.db_manager = db_manager
        self.mcp_hub = mcp_hub
        self.llm = llm
        
        # Get ReAct settings from config
        self.max_iterations = config.config.react_max_iterations
        self.safety_timeout = config.config.react_safety_timeout
    
    async def run(
        self,
        user_message: str,
        conversation_history: Optional[List[Tuple[str, str]]] = None,
        current_cluster: Optional[str] = None,
    ) -> Tuple[str, int]:
        """Run the ReAct loop.
        
        Args:
            user_message: User input message.
            conversation_history: Optional conversation history as [(user_msg, assistant_msg), ...].
            current_cluster: Optional current cluster context.
            
        Returns:
            Tuple of (final_response, chat_id).
        """
        # Start time for timeout tracking
        start_time = time.time()
        
        # Initialize chat history entry
        chat_id = None
        if conversation_history is None:
            conversation_history = []
            
        # Convert conversation history to LLM messages
        messages: List[LLMMessage] = []
        for user_msg, assistant_msg in conversation_history:
            messages.append(LLMMessage(role=MessageRole.USER, content=user_msg))
            messages.append(LLMMessage(role=MessageRole.ASSISTANT, content=assistant_msg))
            
        # Add the current user message
        messages.append(LLMMessage(role=MessageRole.USER, content=user_message))
        
        # Get available tools
        all_tools = self.mcp_hub.get_all_available_tools()
        
        # Flatten tools into a single list
        tools: List[Dict[str, Any]] = []
        for server_name, server_tools in all_tools.items():
            tools.extend(server_tools)
            
        # Convert tools to prepare for LLM
        # For some models we need to map between inputSchema and parameters
        for tool in tools:
            if "inputSchema" not in tool and "parameters" in tool:
                tool["inputSchema"] = tool["parameters"]
        
        # Create system prompt with tool information and cluster context
        system_prompt = self.config.config.system_prompt_template
        if not system_prompt:
            system_prompt = "You are KubeMindNexus, a Kubernetes management assistant."
            
        # Add tools information to system prompt
        tools_description = self.mcp_hub.format_tools_for_prompt()
        system_prompt = system_prompt.format(
            available_tools=tools_description,
            cluster_context=current_cluster or "None"
        )
        
        # Run the ReAct loop
        iteration = 0
        tool_messages: List[LLMMessage] = []
        final_response = "I encountered an error processing your request."
        
        while iteration < self.max_iterations:
            # Check timeout
            if time.time() - start_time > self.safety_timeout:
                logger.warning("ReAct loop timed out")
                final_response = "I'm taking too long to process your request. Please try again or rephrase your query."
                break
                
            iteration += 1
            logger.info(f"ReAct iteration {iteration}")
            
            # Create the full message list for this iteration
            current_messages = messages + tool_messages
            
            # Generate response with tools
            response_text, tool_calls = await self.llm.generate_with_tools(
                current_messages, tools, system_prompt
            )
            
            # If no tool calls, we have our final response
            if not tool_calls:
                final_response = response_text
                break
                
            # Process tool calls
            for tool_call in tool_calls:
                # Extract tool information
                tool_id = tool_call.get("id", "")
                function_data = tool_call.get("function", {})
                tool_name = function_data.get("name", "")
                tool_args = function_data.get("arguments", {})
                
                logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                
                # Find the server for this tool
                server_name = self.mcp_hub.find_server_for_tool(tool_name)
                
                if not server_name:
                    tool_result = f"Error: Tool {tool_name} not found in any available server."
                    logger.error(tool_result)
                else:
                    # Execute the tool
                    success, tool_result = await self.mcp_hub.execute_tool(
                        server_name, tool_name, tool_args, chat_id
                    )
                    
                    if not success:
                        tool_result = f"Error executing tool {tool_name}: {tool_result}"
                        logger.error(tool_result)
                
                # Add tool result to messages
                tool_messages.append(LLMMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=[tool_call]
                ))
                
                tool_messages.append(LLMMessage(
                    role=MessageRole.TOOL,
                    content=str(tool_result),
                    name=tool_name,
                    tool_call_id=tool_id
                ))
            
            # If we've reached the max iterations, generate a final response
            if iteration == self.max_iterations:
                # Create messages with all tool results
                final_messages = messages + tool_messages
                
                # Add a prompt to summarize
                final_messages.append(LLMMessage(
                    role=MessageRole.USER,
                    content="Please provide a final response based on the tool results above."
                ))
                
                # Generate final response (no tools this time)
                final_response, _ = await self.llm.generate(final_messages, system_prompt)
        
        # Save the conversation in the database
        try:
            chat_id = self.db_manager.add_chat_message(
                user_message=user_message,
                assistant_message=final_response,
                cluster_id=None  # We could look up the cluster ID by name here
            )
        except Exception as e:
            logger.error(f"Failed to save chat message: {str(e)}")
        
        return final_response, chat_id
