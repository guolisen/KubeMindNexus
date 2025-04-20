"""OpenAI LLM implementation for KubeMindNexus."""

import json
import logging
import re
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncGenerator

import openai
from openai import OpenAI

from ..constants import LLMProvider
from .base import BaseLLM, MessageRole

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation."""
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """Initialize OpenAI LLM.
        
        Args:
            model: Model name.
            api_key: OpenAI API key.
            base_url: Base URL for OpenAI API.
            parameters: Additional parameters for the model.
        """
        super().__init__(LLMProvider.OPENAI, model, api_key, base_url, parameters)
        
        # Create OpenAI client
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
            
        try:
            self.client = OpenAI(**client_kwargs)
            logger.info(f"Initialized OpenAI client with model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
    
    async def generate(
        self, messages: List[Dict[str, Any]], system_prompt: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate a response from the OpenAI LLM.
        
        Args:
            messages: List of messages in the conversation.
            system_prompt: Optional system prompt to prepend to the conversation.
            
        Returns:
            Tuple of (response_text, tool_calls) where tool_calls is a list of tool calls
            that were extracted from the response.
        """
        # Prepare messages for OpenAI format
        openai_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            openai_messages.append({
                "role": "system",
                "content": system_prompt,
            })
        
        # Add conversation messages (already in dictionary format)
        openai_messages.extend(messages)
        
        # Generate response
        try:
            # Build request parameters
            params = {
                "model": self.model,
                "messages": openai_messages,
                "stream": False,
                **self.parameters,
            }
            
            # Send request - remove await as the client doesn't return a coroutine
            response = self.client.chat.completions.create(**params)
            
            if not response.choices:
                return "", []
                
            choice = response.choices[0]
            response_message = choice.message
            
            # Extract response text
            response_text = response_message.content or ""
            
            # Extract tool calls
            tool_calls = []
            if hasattr(response_message, "tool_calls") and response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    # Extract tool call data
                    tool_call_data = {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    
                    # Parse arguments from JSON string
                    try:
                        tool_call_data["function"]["arguments"] = json.loads(
                            tool_call.function.arguments
                        )
                    except json.JSONDecodeError:
                        # If parsing fails, keep the string version
                        pass
                        
                    tool_calls.append(tool_call_data)
            
            return response_text, tool_calls
            
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {str(e)}")
            return f"Error generating response: {str(e)}", []
    
    async def generate_stream(
        self, messages: List[Dict[str, Any]], system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Generate a response from the OpenAI LLM with streaming support.
        
        Args:
            messages: List of messages in the conversation.
            system_prompt: Optional system prompt to prepend to the conversation.
            
        Yields:
            Chunks of the generated response as they become available.
        """
        # Prepare messages for OpenAI format
        openai_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            openai_messages.append({
                "role": "system",
                "content": system_prompt,
            })
        
        # Add conversation messages
        openai_messages.extend(messages)
        
        # Generate streaming response
        stream = None
        try:
            # Build request parameters
            params = {
                "model": self.model,
                "messages": openai_messages,
                "stream": True,  # Enable streaming
                **self.parameters,
            }
            
            # Send request with streaming
            stream = self.client.chat.completions.create(**params)
            
            current_content = ""
            # Use a try-except inside the loop to handle socket errors for individual chunks
            for chunk in stream:
                try:
                    if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                        content_delta = chunk.choices[0].delta.content
                        if content_delta:
                            # Yield the new chunk
                            yield content_delta
                            current_content += content_delta
                except (BrokenPipeError, ConnectionError, OSError) as socket_err:
                    # Log the socket error but don't crash the entire stream
                    logger.warning(f"Socket error while streaming chunk: {str(socket_err)}")
                    # Try to continue with the next chunk
                    continue
                    
            # If no content was yielded, yield an empty string to ensure the generator yields at least once
            if not current_content:
                yield ""
                
        except Exception as e:
            logger.error(f"Error streaming response from OpenAI: {str(e)}")
            yield f"Error streaming response: {str(e)}"
        finally:
            # Ensure resources are properly cleaned up
            if stream and hasattr(stream, 'close'):
                try:
                    stream.close()
                except Exception as close_err:
                    # Just log cleanup errors
                    logger.warning(f"Error closing stream: {str(close_err)}")
    
    async def generate_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate a response from the OpenAI LLM with tool definitions.
        
        Args:
            messages: List of messages in the conversation.
            tools: List of tool definitions.
            system_prompt: Optional system prompt to prepend to the conversation.
            
        Returns:
            Tuple of (response_text, tool_calls) where tool_calls is a list of tool calls
            that were extracted from the response.
        """
        # Prepare messages for OpenAI format
        openai_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            openai_messages.append({
                "role": "system",
                "content": system_prompt,
            })
        
        # Add conversation messages (already in dictionary format)
        openai_messages.extend(messages)
        
        # Generate response
        try:
            # Convert tools to OpenAI format
            openai_tools = []
            for tool in tools:
                # Create OpenAI tool format
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                    }
                }
                
                # Add parameter schema if provided
                if "inputSchema" in tool:
                    openai_tool["function"]["parameters"] = tool["inputSchema"]
                    
                openai_tools.append(openai_tool)
            
            # Build request parameters
            params = {
                "model": self.model,
                "messages": openai_messages,
                "tools": openai_tools,
                "tool_choice": "auto",
                **self.parameters,
            }
            
            # Send request - remove await as the client doesn't return a coroutine
            response = self.client.chat.completions.create(**params)
            
            if not response.choices:
                return "", []
                
            choice = response.choices[0]
            response_message = choice.message
            
            # Extract response text
            response_text = response_message.content or ""
            
            # Extract tool calls
            tool_calls = []
            if hasattr(response_message, "tool_calls") and response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    # Extract tool call data
                    tool_call_data = {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    
                    # Parse arguments from JSON string
                    try:
                        tool_call_data["function"]["arguments"] = json.loads(
                            tool_call.function.arguments
                        )
                    except json.JSONDecodeError:
                        # If parsing fails, keep the string version
                        pass
                        
                    tool_calls.append(tool_call_data)
            
            return response_text, tool_calls
            
        except Exception as e:
            logger.error(f"Error generating response from OpenAI with tools: {str(e)}")
            return f"Error generating response: {str(e)}", []
