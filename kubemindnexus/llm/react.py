"""ReAct loop implementation for KubeMindNexus.

This module provides an enhanced ReAct (Reasoning + Acting) loop implementation
that supports modular system prompts, better context management, and improved
error handling. It also supports streaming responses for real-time UI updates.
"""
import re
import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast, AsyncGenerator
from enum import Enum

class EventType(Enum):
    """Types of events that can be emitted during ReactLoop processing."""
    THINKING = "thinking"     # LLM is generating a response
    RESPONSE = "response"     # Partial or complete LLM text response
    TOOL_CALL = "tool_call"   # LLM is calling a tool
    TOOL_RESULT = "tool_result"  # Result from tool execution
    COMPLETION = "completion"  # Final completion of the task

from ..config import Configuration
from ..constants import REACT_MAX_ITERATIONS, REACT_SAFETY_TIMEOUT, ATTEMPT_COMPLETION_TOOL_NAME
from ..database import DatabaseManager
from ..mcp.hub import MCPHub
from ..prompts.system import generate_system_prompt, generate_tool_format
from .base import BaseLLM, MessageRole

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
        
        # Task completion tracking
        self.task_completed = False
        self.completion_result = None
    
    async def run(
        self,
        user_message: str,
        conversation_history: Optional[List[Tuple[str, str]]] = None,
        current_cluster: Optional[str] = None,
    ) -> Tuple[str, int, Dict[str, Any]]:
        """Run the ReAct loop with enhanced system prompt and context management.
        
        Args:
            user_message: User input message.
            conversation_history: Optional conversation history as [(user_msg, assistant_msg), ...].
            current_cluster: Optional current cluster context.
            
        Returns:
            Tuple of (final_response, chat_id, metadata_dict) where metadata_dict contains 
            information about the task completion and execution.
        """
        # Start time for timeout tracking
        start_time = time.time()
        
        # Initialize chat history entry
        chat_id = None
        if conversation_history is None:
            conversation_history = []
            
        # Convert conversation history to messages
        messages: List[Dict[str, Any]] = []
        for user_msg, assistant_msg in conversation_history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
            
        # Add the current user message
        messages.append({"role": "user", "content": user_message})
        
        cluster_tools = {}
        # If a cluster context is provided, make sure to include both cluster-specific
        # and local tools to provide a complete set of capabilities
        if current_cluster:
            cluster_tools = self.mcp_hub.get_tools_by_cluster(current_cluster)

        # Get local tools
        local_tools = self.mcp_hub.get_local_tools()
        
        # Convert tools to prepare for LLM
        # For some models we need to map between inputSchema and parameters
        server_tools_map = {}
        for server_name, server_tools in cluster_tools.items():
            for tool in server_tools:
                server_tools_map[tool["name"]] = server_name
                if "inputSchema" not in tool and "parameters" in tool:
                    tool["inputSchema"] = tool["parameters"]
        for server_name, server_tools in local_tools.items():
            for tool in server_tools:
                server_tools_map[tool["name"]] = server_name
                if "inputSchema" not in tool and "parameters" in tool:
                    tool["inputSchema"] = tool["parameters"]
       
        # Format tools for the system prompt with cluster context
        # Either use the enhanced formatter or fall back to the basic one
        try:
            tools_description = generate_tool_format(cluster_tools, local_tools, current_cluster)
            logger.info(f"Using enhanced tool formatting for system prompt with cluster: {current_cluster}")
        except Exception as e:
            logger.warning(f"Error using enhanced tool formatting, falling back to basic: {str(e)}")
            tools_description = self.mcp_hub.format_tools_for_prompt(current_cluster)
        
        # Generate the system prompt
        system_prompt_template = self.config.config.system_prompt_template
        
        if system_prompt_template == "use_enhanced":
            try:
                # Get system prompt options
                system_prompt_options = getattr(self.config.config, "system_prompt_options", {})
                if not system_prompt_options:
                    system_prompt_options = {}
                    
                # Extract options with defaults
                include_react_guidance = system_prompt_options.get("include_react_guidance", True)
                include_mcp_guidance = system_prompt_options.get("include_mcp_guidance", True)
                include_kubernetes_guidance = system_prompt_options.get("include_kubernetes_guidance", True)
                
                # Generate the enhanced modular system prompt
                system_prompt = generate_system_prompt(
                    available_tools=tools_description,
                    cluster_context=current_cluster,
                    include_mcp_guidance=include_mcp_guidance,
                    include_react_guidance=include_react_guidance
                )
                logger.info("Using enhanced modular system prompt")
            except Exception as e:
                # Fall back to the legacy template if there's an error
                logger.warning(f"Error generating enhanced system prompt, falling back to legacy: {str(e)}")
                legacy_prompt = getattr(self.config.config, "legacy_system_prompt", None)
                if legacy_prompt:
                    system_prompt = legacy_prompt
                else:
                    system_prompt = "You are KubeMindNexus, a Kubernetes management assistant."
                    
                # Use the basic format with the template
                system_prompt = system_prompt.format(
                    available_tools=tools_description,
                    cluster_context=current_cluster or "None"
                )
        else:
            # Use the provided template
            if not system_prompt_template:
                system_prompt_template = "You are KubeMindNexus, a Kubernetes management assistant."
                
            # Format the template
            system_prompt = system_prompt_template.format(
                available_tools=tools_description,
                cluster_context=current_cluster or "None"
            )
        
        # Run the ReAct loop
        iteration = 0
        tool_messages: List[Dict[str, Any]] = []
        final_response = "I encountered an error processing your request."
        
        while iteration < self.max_iterations:
            # Check timeout with more detailed logging
            elapsed_time = time.time() - start_time
            '''
            if elapsed_time > self.safety_timeout:
                logger.warning(f"ReAct loop timed out after {elapsed_time:.2f}s (limit: {self.safety_timeout}s)")
                final_response = (
                    "I'm taking too long to process your request. This might be due to the complexity "
                    "of the task or current system load. Please try again with a more specific query "
                    "or break your request into smaller steps."
                )
                break
            '''
            iteration += 1
            logger.info(f"ReAct iteration {iteration}")
            
            # Create the full message list for this iteration
            current_messages = messages + tool_messages
            
            # Generate response with tools
            response_text, _ = await self.llm.generate(
                current_messages, system_prompt
            )

            logger.info(f"LLM Response: {response_text}")
            
            logger.info("Receive LLM response: {}".format(response_text))

            # Translate tool format if needed

            def translate_tool_format(text):
                """
                Translate from simple format to nested JSON if the first line is a tool name.
                
                FROM:
                filesystem_list_directory
                {"path": "/home"}
                
                TO:
                {
                  "tool": "filesystem_list_directory",
                  "parameters": {
                    "path": "/home"
                  }
                }
                """
                lines = text.strip().split('\n', 1)
                if len(lines) < 2:
                    return text
                
                potential_tool_name = lines[0].strip()
                potential_params = lines[1].strip()
                
                # Check if first line looks like a tool name (no spaces, no JSON characters)
                if ' ' in potential_tool_name or '{' in potential_tool_name or '}' in potential_tool_name:
                    return text
                
                # Try to parse the second part as JSON
                try:
                    params = json.loads(potential_params)
                    # Create the new format
                    transformed = {
                        "tool": potential_tool_name,
                        "parameters": params
                    }
                    return json.dumps(transformed)
                except json.JSONDecodeError:
                    return text
            
            # Apply translation if needed
            response_text = translate_tool_format(response_text)
            
            try:
                tool_call = json.loads(response_text)

                if isinstance(tool_call, dict) and "tool" in tool_call and "parameters" in tool_call:
                    # This is an MCP tool call
                    tool_name = tool_call["tool"]
                    tool_args = tool_call["parameters"]
                    
                    logger.info(f"Executing MCP tool: {tool_name} with args: {tool_args}")
                    

                    # Check for attempt_completion tool
                    if tool_name == ATTEMPT_COMPLETION_TOOL_NAME:
                        # This is a task completion signal
                        logger.info("Detected attempt_completion tool call, completing task")
                        
                        # Get the completion result
                        self.task_completed = True
                        self.completion_result = tool_args.get("result", "Task completed successfully.")
                        
                        # Set the final response from the completion result
                        final_response = self.completion_result
                        
                        # Add command execution logic if provided
                        command = tool_args.get("command")
                        if command:
                            logger.info(f"Completion includes command to demonstrate result: {command}")
                            # Note: In a real implementation, you might want to execute this command
                            # or notify the UI to execute it
                        
                        # Log the completion
                        tool_messages.append({
                            "role": "assistant",
                            "content": f"I've completed the task: {self.completion_result}"
                        })
                        
                        # Break the loop as the task is complete
                        break
                
                    # Find the server for this tool
                    # server_name = tool_call["server"] #self.mcp_hub.find_server_for_tool(tool_name)
                    # server_name = self.mcp_hub.find_server_for_tool(tool_name)
                    server_name = server_tools_map.get(tool_name)

                    # If server_name is None, try to search tool name in mcp_hub
                    if not server_name:
                        server_name = self.mcp_hub.find_server_for_tool(tool_name)
                        if server_name:
                            logger.info(f"Found server {server_name} for tool {tool_name} via mcp_hub search")
                        else:
                            tool_result = f"Error: Tool {tool_name} not found in any available server."
                            logger.error(tool_result)
                            
                            # Add to conversation history
                            tool_messages.append({
                                "role": "assistant",
                                "content": response_text
                            })
                            
                            tool_messages.append({
                                "role": "user",
                                "content": tool_result
                            })
                            continue
                    # Execute the tool if we have a valid server_name
                    else:
                        success, tool_result = await self.mcp_hub.execute_tool(
                            server_name, tool_name, tool_args, chat_id
                        )
                        
                        if not success:
                            tool_result = f"Error executing tool {tool_name}: {tool_result}"
                            logger.error(tool_result)
                        logger.info("Call tool result: " + str(tool_result))

                    # Add tool result to messages for context
                    tool_messages.append({
                        "role": "assistant",
                        "content": response_text
                    })
                    
                    tool_messages.append({
                        "role": "user",
                        #"content": str("thinking and try to answer the previous user query according to following information: " + tool_result + "\n")
                        "content": str(tool_result)
                    })
                    
                    # Continue to next iteration
                    continue
                else:
                    # Not a tool call, just a regular response
                    final_response = response_text
                    break
            except json.JSONDecodeError as e:
                # Not JSON, just a regular response
                logger.info(f"Response is not a JSON tool call: {str(e)}")

                # Add tool result to messages for context
                tool_messages.append({
                    "role": "assistant",
                    "content": response_text
                })
                
                tool_messages.append({
                    "role": "user",
                    "content": str("check whether complete the react loop, if so, call attempt_completion tool with final response")
                })
                continue

            # If we've reached the max iterations, generate a final response
            if iteration == self.max_iterations:
                # Create messages with all tool results
                final_messages = messages + tool_messages
                
                # Add a prompt to summarize
                final_messages.append({
                    "role": "user",
                    "content": "Please provide a final response based on the tool results above."
                })
                
                # Generate final response (no tools this time)
                final_response, _ = await self.llm.generate(final_messages, system_prompt)
        
        # Save the conversation in the database with more metadata
        try:
            # Determine if we should include cluster information
            cluster_id = None
            if current_cluster:
                # Try to find the cluster ID by name (if implemented in the db_manager)
                try:
                    cluster = self.db_manager.get_cluster_by_name(current_cluster)
                    if cluster:
                        cluster_id = cluster["id"]
                except Exception as cluster_err:
                    logger.warning(f"Could not get cluster ID: {str(cluster_err)}")
            
            # Add the chat message with additional metadata
            chat_id = self.db_manager.add_chat_message(
                user_message=user_message,
                assistant_message=final_response,
                cluster_id=cluster_id,
                metadata={
                    "iterations": iteration,
                    "execution_time": time.time() - start_time,
                    "tool_count": len(tool_messages) // 2  # Each tool use is 2 messages (call + result)
                }
            )
            logger.info(f"Saved chat message with ID {chat_id}")
        except Exception as e:
            logger.error(f"Failed to save chat message: {str(e)}")
        
        # Create metadata dictionary with execution info
        metadata = {
            "task_completed": self.task_completed,
            "completion_result": self.completion_result,
            "iterations": iteration,
            "execution_time": time.time() - start_time,
            "tool_count": len(tool_messages) // 2
        }
        
        # Return the final response, chat ID, and metadata
        return final_response, chat_id, metadata
        
    async def run_stream(
        self,
        user_message: str,
        conversation_history: Optional[List[Tuple[str, str]]] = None,
        current_cluster: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run the ReAct loop with streaming support.
        
        Args:
            user_message: User input message.
            conversation_history: Optional conversation history as [(user_msg, assistant_msg), ...].
            current_cluster: Optional current cluster context.
            
        Yields:
            Events representing different stages of processing.
        """
        # Start time for timeout tracking
        start_time = time.time()
        
        # Initialize chat history entry
        chat_id = None
        if conversation_history is None:
            conversation_history = []
            
        # Convert conversation history to messages
        messages: List[Dict[str, Any]] = []
        for user_msg, assistant_msg in conversation_history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
            
        # Add the current user message
        messages.append({"role": "user", "content": user_message})
        
        # Set up tools and system prompt similar to the run method
        cluster_tools = {}
        if current_cluster:
            cluster_tools = self.mcp_hub.get_tools_by_cluster(current_cluster)

        # Get local tools
        local_tools = self.mcp_hub.get_local_tools()
        
        # Convert tools to prepare for LLM
        server_tools_map = {}
        for server_name, server_tools in cluster_tools.items():
            for tool in server_tools:
                server_tools_map[tool["name"]] = server_name
                if "inputSchema" not in tool and "parameters" in tool:
                    tool["inputSchema"] = tool["parameters"]
        for server_name, server_tools in local_tools.items():
            for tool in server_tools:
                server_tools_map[tool["name"]] = server_name
                if "inputSchema" not in tool and "parameters" in tool:
                    tool["inputSchema"] = tool["parameters"]
       
        # Format tools for the system prompt
        try:
            tools_description = generate_tool_format(cluster_tools, local_tools, current_cluster)
        except Exception as e:
            logger.warning(f"Error using enhanced tool formatting, falling back to basic: {str(e)}")
            tools_description = self.mcp_hub.format_tools_for_prompt(current_cluster)
        
        # Generate the system prompt
        system_prompt_template = self.config.config.system_prompt_template
        
        if system_prompt_template == "use_enhanced":
            try:
                # Get system prompt options
                system_prompt_options = getattr(self.config.config, "system_prompt_options", {})
                if not system_prompt_options:
                    system_prompt_options = {}
                    
                # Extract options with defaults
                include_react_guidance = system_prompt_options.get("include_react_guidance", True)
                include_mcp_guidance = system_prompt_options.get("include_mcp_guidance", True)
                include_kubernetes_guidance = system_prompt_options.get("include_kubernetes_guidance", True)
                
                # Generate the enhanced modular system prompt
                system_prompt = generate_system_prompt(
                    available_tools=tools_description,
                    cluster_context=current_cluster,
                    include_mcp_guidance=include_mcp_guidance,
                    include_react_guidance=include_react_guidance
                )
            except Exception as e:
                # Fall back to the legacy template
                logger.warning(f"Error generating enhanced system prompt, falling back to legacy: {str(e)}")
                legacy_prompt = getattr(self.config.config, "legacy_system_prompt", None)
                if legacy_prompt:
                    system_prompt = legacy_prompt
                else:
                    system_prompt = "You are KubeMindNexus, a Kubernetes management assistant."
                
                system_prompt = system_prompt.format(
                    available_tools=tools_description,
                    cluster_context=current_cluster or "None"
                )
        else:
            # Use the provided template
            if not system_prompt_template:
                system_prompt_template = "You are KubeMindNexus, a Kubernetes management assistant."
                
            # Format the template
            system_prompt = system_prompt_template.format(
                available_tools=tools_description,
                cluster_context=current_cluster or "None"
            )
        
        # Run the ReAct loop
        iteration = 0
        tool_messages: List[Dict[str, Any]] = []
        final_response = "I encountered an error processing your request."
        
        while iteration < self.max_iterations:
            # Check timeout with more detailed logging
            elapsed_time = time.time() - start_time
            '''
            if elapsed_time > self.safety_timeout:
                logger.warning(f"ReAct loop timed out after {elapsed_time:.2f}s (limit: {self.safety_timeout}s)")
                yield {
                    "type": EventType.COMPLETION.value,
                    "data": {
                        "result": "I'm taking too long to process your request. This might be due to the complexity "
                                "of the task or current system load. Please try again with a more specific query "
                                "or break your request into smaller steps."
                    },
                    "timestamp": time.time()
                }
                return
            '''
            
            iteration += 1
            logger.info(f"ReAct iteration {iteration}")
            
            # Yield thinking event
            yield {
                "type": EventType.THINKING.value,
                "data": {
                    "iteration": iteration,
                    "message": f"Thinking about how to respond (iteration {iteration})..."
                },
                "timestamp": time.time()
            }
            
            # Create the full message list for this iteration
            current_messages = messages + tool_messages
            
            # Generate response with tools, using streaming if available
            try:
                response_chunks = []
                
                # Try to use streaming generate method if available
                async for chunk in self.llm.generate_stream(current_messages, system_prompt):
                    response_chunks.append(chunk)
                    # Yield each chunk as a response event
                    yield {
                        "type": EventType.RESPONSE.value,
                        "data": {
                            "content": chunk,
                            "is_partial": True
                        },
                        "timestamp": time.time()
                    }
                
                # Combine chunks to get the full response
                response_text = "".join(response_chunks)
                
            except Exception as e:
                # Fall back to normal generate method
                logger.warning(f"Streaming generation failed, falling back to normal generation: {str(e)}")
                response_text, _ = await self.llm.generate(current_messages, system_prompt)
                
                # Yield the full response as a single event
                yield {
                    "type": EventType.RESPONSE.value,
                    "data": {
                        "content": response_text,
                        "is_partial": False
                    },
                    "timestamp": time.time()
                }
            
            logger.info(f"SSLLM Response: {response_text}")
            
            # Translate tool format if needed
            def translate_tool_format(text):
                """
                Translate from simple format to nested JSON if the first line is a tool name.
                """
                lines = text.strip().split('\n', 1)
                if len(lines) < 2:
                    return text
                
                potential_tool_name = lines[0].strip()
                potential_params = lines[1].strip()
                
                # Check if first line looks like a tool name (no spaces, no JSON characters)
                if ' ' in potential_tool_name or '{' in potential_tool_name or '}' in potential_tool_name:
                    return text
                
                # Try to parse the second part as JSON
                try:
                    params = json.loads(potential_params)
                    # Create the new format
                    transformed = {
                        "tool": potential_tool_name,
                        "parameters": params
                    }
                    return json.dumps(transformed)
                except json.JSONDecodeError:
                    return text
            
            # Apply translation if needed
            response_text = translate_tool_format(response_text)
            
            try:
                tool_call = json.loads(response_text)

                if isinstance(tool_call, dict) and "tool" in tool_call and "parameters" in tool_call:
                    # This is an MCP tool call
                    tool_name = tool_call["tool"]
                    tool_args = tool_call["parameters"]
                    
                    logger.info(f"Executing MCP tool: {tool_name} with args: {tool_args}")
                    
                    # Yield tool call event
                    yield {
                        "type": EventType.TOOL_CALL.value,
                        "data": {
                            "tool_name": tool_name,
                            "parameters": tool_args
                        },
                        "timestamp": time.time()
                    }

                    # Check for attempt_completion tool
                    if tool_name == ATTEMPT_COMPLETION_TOOL_NAME:
                        # This is a task completion signal
                        logger.info("Detected attempt_completion tool call, completing task")
                        
                        # Get the completion result
                        self.task_completed = True
                        self.completion_result = tool_args.get("result", "Task completed successfully.")
                        
                        # Set the final response from the completion result
                        final_response = self.completion_result
                        
                        # Add command execution logic if provided
                        command = tool_args.get("command")
                        
                        # Yield completion event
                        yield {
                            "type": EventType.COMPLETION.value,
                            "data": {
                                "result": self.completion_result,
                                "command": command
                            },
                            "timestamp": time.time()
                        }
                        
                        # Add to conversation history
                        tool_messages.append({
                            "role": "assistant",
                            "content": f"I've completed the task: {self.completion_result}"
                        })
                        
                        # Break the loop as the task is complete
                        break
                
                    # Find the server for this tool
                    server_name = server_tools_map.get(tool_name)

                    # If server_name is None, try to search tool name in mcp_hub
                    if not server_name:
                        server_name = self.mcp_hub.find_server_for_tool(tool_name)
                        if server_name:
                            logger.info(f"Found server {server_name} for tool {tool_name} via mcp_hub search")
                        else:
                            tool_result = f"Error: Tool {tool_name} not found in any available server."
                            logger.error(tool_result)
                            
                            # Yield tool result event (error)
                            yield {
                                "type": EventType.TOOL_RESULT.value,
                                "data": {
                                    "tool_name": tool_name,
                                    "result": tool_result,
                                    "success": False
                                },
                                "timestamp": time.time()
                            }
                            
                            # Add to conversation history
                            tool_messages.append({
                                "role": "assistant",
                                "content": response_text
                            })
                            
                            tool_messages.append({
                                "role": "user",
                                "content": tool_result
                            })
                            continue
                            
                    # Execute the tool if we have a valid server_name
                    else:
                        success, tool_result = await self.mcp_hub.execute_tool(
                            server_name, tool_name, tool_args, chat_id
                        )
                        
                        if not success:
                            tool_result = f"Error executing tool {tool_name}: {tool_result}"
                            logger.error(tool_result)
                            
                        logger.info("Call tool result: " + str(tool_result))
                        
                        # Yield tool result event
                        yield {
                            "type": EventType.TOOL_RESULT.value,
                            "data": {
                                "tool_name": tool_name,
                                "result": tool_result,
                                "success": success
                            },
                            "timestamp": time.time()
                        }

                    # Add tool result to messages for context
                    tool_messages.append({
                        "role": "assistant",
                        "content": response_text
                    })
                    
                    tool_messages.append({
                        "role": "user",
                        "content": str(tool_result)
                    })
                    
                    # Continue to next iteration
                    continue
                else:
                    # Not a tool call, just a regular response
                    final_response = response_text
                    
                    # Yield a final response event
                    yield {
                        "type": EventType.RESPONSE.value,
                        "data": {
                            "content": final_response,
                            "is_partial": False,
                            "is_final": True
                        },
                        "timestamp": time.time()
                    }
                    break
            except json.JSONDecodeError as e:
                # Not JSON, just a regular response
                logger.info(f"Response is not a JSON tool call: {str(e)}")

                # Add to conversation history
                tool_messages.append({
                    "role": "assistant",
                    "content": response_text
                })
                
                tool_messages.append({
                    "role": "user",
                    "content": str("check whether complete the react loop, if so, call attempt_completion tool with final response")
                })
                continue

            # If we've reached the max iterations, generate a final response
            if iteration == self.max_iterations:
                # Create messages with all tool results
                final_messages = messages + tool_messages
                
                # Add a prompt to summarize
                final_messages.append({
                    "role": "user",
                    "content": "Please provide a final response based on the tool results above."
                })
                
                # Generate final response (no tools this time)
                final_response, _ = await self.llm.generate(final_messages, system_prompt)
                
                # Yield final response
                yield {
                    "type": EventType.RESPONSE.value,
                    "data": {
                        "content": final_response,
                        "is_partial": False,
                        "is_final": True
                    },
                    "timestamp": time.time()
                }
        
        # Save the conversation in the database
        try:
            # Determine if we should include cluster information
            cluster_id = None
            if current_cluster:
                try:
                    cluster = self.db_manager.get_cluster_by_name(current_cluster)
                    if cluster:
                        cluster_id = cluster["id"]
                except Exception as cluster_err:
                    logger.warning(f"Could not get cluster ID: {str(cluster_err)}")
            
            # Add the chat message with additional metadata
            chat_id = self.db_manager.add_chat_message(
                user_message=user_message,
                assistant_message=final_response,
                cluster_id=cluster_id,
                metadata={
                    "iterations": iteration,
                    "execution_time": time.time() - start_time,
                    "tool_count": len(tool_messages) // 2  # Each tool use is 2 messages (call + result)
                }
            )
            logger.info(f"Saved chat message with ID {chat_id}")
        except Exception as e:
            logger.error(f"Failed to save chat message: {str(e)}")
        
        # Yield final completion event if not already sent
        if not self.task_completed:
            yield {
                "type": EventType.COMPLETION.value,
                "data": {
                    "result": final_response
                },
                "timestamp": time.time()
            }
    
    async def _save_chat_message(
        self,
        user_message: str,
        assistant_message: str,
        current_cluster: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Save chat message to database.
        
        Args:
            user_message: User input message.
            assistant_message: Assistant response message.
            current_cluster: Optional current cluster context.
            metadata: Optional metadata to save with the message.
            
        Returns:
            Chat message ID.
        """
        try:
            # Determine if we should include cluster information
            cluster_id = None
            if current_cluster:
                try:
                    cluster = self.db_manager.get_cluster_by_name(current_cluster)
                    if cluster:
                        cluster_id = cluster["id"]
                except Exception as cluster_err:
                    logger.warning(f"Could not get cluster ID: {str(cluster_err)}")
            
            # Add the chat message with additional metadata
            chat_id = self.db_manager.add_chat_message(
                user_message=user_message,
                assistant_message=assistant_message,
                cluster_id=cluster_id,
                metadata=metadata or {}
            )
            logger.info(f"Saved chat message with ID {chat_id}")
            return chat_id
        except Exception as e:
            logger.error(f"Failed to save chat message: {str(e)}")
            return 0
