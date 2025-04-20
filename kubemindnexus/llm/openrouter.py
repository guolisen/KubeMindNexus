"""OpenRouter LLM implementation for KubeMindNexus."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncGenerator

import httpx
import asyncio

from ..constants import LLMProvider
from .base import BaseLLM, MessageRole

logger = logging.getLogger(__name__)


class OpenRouterLLM(BaseLLM):
    """OpenRouter LLM implementation."""
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """Initialize OpenRouter LLM.
        
        Args:
            model: Model name.
            api_key: OpenRouter API key.
            base_url: Base URL for OpenRouter API.
            parameters: Additional parameters for the model.
        """
        super().__init__(LLMProvider.OPENROUTER, model, api_key, base_url, parameters)
        
        # Set default base URL if not provided
        self.base_url = base_url or "https://openrouter.ai/api/v1"
        
        # Initialize HTTP client with auth headers
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        # OpenRouter requires a HTTP_REFERER and X-Title header
        headers["HTTP_REFERER"] = "https://kubemindnexus.local"
        headers["X-Title"] = "KubeMindNexus"
            
        self.client = httpx.AsyncClient(base_url=self.base_url, headers=headers)
        
        logger.info(f"Initialized OpenRouter client with model: {model}")
    
    async def generate(
        self, messages: List[Dict[str, Any]], system_prompt: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate a response from the OpenRouter LLM.
        
        Args:
            messages: List of messages in the conversation.
            system_prompt: Optional system prompt to prepend to the conversation.
            
        Returns:
            Tuple of (response_text, tool_calls) where tool_calls is a list of tool calls
            that were extracted from the response.
        """
        # Prepare messages for OpenRouter format (compatible with OpenAI)
        api_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            api_messages.append({
                "role": "system",
                "content": system_prompt,
            })
        
        # Add conversation messages (already in dictionary format)
        api_messages.extend(messages)
        
        # Generate response
        try:
            # Build request parameters
            api_params = {
                "model": self.model,
                "messages": api_messages,
                **self.parameters,
            }
            
            # Send request
            response = await self.client.post("/chat/completions", json=api_params)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            choice = result.get("choices", [{}])[0]
            message = choice.get("message", {})
            
            # Extract response text
            response_text = message.get("content", "")
            
            # Extract tool calls if available
            tool_calls = []
            if "tool_calls" in message:
                for tool_call in message["tool_calls"]:
                    # Extract tool call data
                    tool_call_data = {
                        "id": tool_call.get("id", f"openrouter-tool-{len(tool_calls)}"),
                        "type": tool_call.get("type", "function"),
                        "function": {
                            "name": tool_call.get("function", {}).get("name", ""),
                            "arguments": tool_call.get("function", {}).get("arguments", "{}"),
                        },
                    }
                    
                    # Parse arguments from JSON string if needed
                    if isinstance(tool_call_data["function"]["arguments"], str):
                        try:
                            tool_call_data["function"]["arguments"] = json.loads(
                                tool_call_data["function"]["arguments"]
                            )
                        except json.JSONDecodeError:
                            # If parsing fails, keep the string version
                            pass
                        
                    tool_calls.append(tool_call_data)
            
            return response_text, tool_calls
            
        except Exception as e:
            logger.error(f"Error generating response from OpenRouter: {str(e)}")
            return f"Error generating response: {str(e)}", []
            
    async def generate_stream(
        self, messages: List[Dict[str, Any]], system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Generate a response from the OpenRouter LLM with streaming support.
        
        Args:
            messages: List of messages in the conversation.
            system_prompt: Optional system prompt to prepend to the conversation.
            
        Yields:
            Chunks of the generated response as they become available.
        """
        # Prepare messages for OpenRouter format (compatible with OpenAI)
        api_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            api_messages.append({
                "role": "system",
                "content": system_prompt,
            })
        
        # Add conversation messages
        api_messages.extend(messages)
        
        # Generate streaming response
        response = None
        try:
            # Build request parameters with streaming enabled
            api_params = {
                "model": self.model,
                "messages": api_messages,
                "stream": True,  # Enable streaming
                **self.parameters,
            }
            
            # Send request with streaming
            response = await self.client.post(
                "/chat/completions", 
                json=api_params,
                timeout=120.0  # Extend timeout for streaming
            )
            response.raise_for_status()
            
            # Process the streaming response
            current_content = ""
            async for line in response.aiter_lines():
                if not line.strip() or line.startswith(":"):
                    # Skip empty lines and SSE comments
                    continue
                    
                # Each line should be a "data: {...}" JSON object
                if line.startswith("data: "):
                    try:
                        # Extract and parse the JSON data
                        json_str = line[6:]  # Remove "data: " prefix
                        
                        # Check for the [DONE] message
                        if json_str.strip() == "[DONE]":
                            break
                            
                        data = json.loads(json_str)
                        
                        # Extract the delta content if available
                        choices = data.get("choices", [])
                        if choices and "delta" in choices[0]:
                            delta = choices[0]["delta"]
                            content = delta.get("content", "")
                            
                            if content:
                                try:
                                    # Yield the content chunk
                                    yield content
                                    current_content += content
                                except (BrokenPipeError, ConnectionError, OSError) as socket_err:
                                    # Log the socket error but continue processing
                                    logger.warning(f"Socket error while streaming chunk: {str(socket_err)}")
                                    # Try to continue with the next chunk
                                    continue
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing SSE JSON: {str(e)} - Line: {line}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing stream chunk: {str(e)}")
                        continue
            
            # If no content was yielded, yield an empty string
            if not current_content:
                yield ""
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error streaming from OpenRouter: {e.response.status_code} - {str(e)}")
            yield f"Error streaming response: HTTP {e.response.status_code}"
        except httpx.RequestError as e:
            logger.error(f"Request error streaming from OpenRouter: {str(e)}")
            yield f"Error streaming response: Connection error"
        except Exception as e:
            logger.error(f"Error streaming response from OpenRouter: {str(e)}")
            yield f"Error streaming response: {str(e)}"
        finally:
            # Ensure proper cleanup of resources
            if response and not response.is_closed:
                try:
                    response.close()
                except Exception as close_err:
                    logger.warning(f"Error closing response stream: {str(close_err)}")
    
    async def generate_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate a response from the OpenRouter LLM with tool definitions.
        
        Args:
            messages: List of messages in the conversation.
            tools: List of tool definitions.
            system_prompt: Optional system prompt to prepend to the conversation.
            
        Returns:
            Tuple of (response_text, tool_calls) where tool_calls is a list of tool calls
            that were extracted from the response.
        """
        # Prepare messages for OpenRouter format (compatible with OpenAI)
        api_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            api_messages.append({
                "role": "system",
                "content": system_prompt,
            })
        
        # Add conversation messages (already in dictionary format)
        api_messages.extend(messages)
        
        # Convert tools to OpenRouter format (compatible with OpenAI)
        api_tools = []
        for tool in tools:
            # Create OpenAI-compatible tool format
            api_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                },
            }
            
            # Add parameter schema if provided
            if "inputSchema" in tool:
                api_tool["function"]["parameters"] = tool["inputSchema"]
                
            api_tools.append(api_tool)
        
        # Generate response
        try:
            # Build request parameters
            api_params = {
                "model": self.model,
                "messages": api_messages,
                "tools": api_tools,
                "tool_choice": "auto",
                **self.parameters,
            }
            
            # Send request
            response = await self.client.post("/chat/completions", json=api_params)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            choice = result.get("choices", [{}])[0]
            message = choice.get("message", {})
            
            # Extract response text
            response_text = message.get("content", "")
            
            # Extract tool calls if available
            tool_calls = []
            if "tool_calls" in message:
                for tool_call in message["tool_calls"]:
                    # Extract tool call data
                    tool_call_data = {
                        "id": tool_call.get("id", f"openrouter-tool-{len(tool_calls)}"),
                        "type": tool_call.get("type", "function"),
                        "function": {
                            "name": tool_call.get("function", {}).get("name", ""),
                            "arguments": tool_call.get("function", {}).get("arguments", "{}"),
                        },
                    }
                    
                    # Parse arguments from JSON string if needed
                    if isinstance(tool_call_data["function"]["arguments"], str):
                        try:
                            tool_call_data["function"]["arguments"] = json.loads(
                                tool_call_data["function"]["arguments"]
                            )
                        except json.JSONDecodeError:
                            # If parsing fails, keep the string version
                            pass
                        
                    tool_calls.append(tool_call_data)
            
            return response_text, tool_calls
            
        except Exception as e:
            logger.error(f"Error generating response from OpenRouter with tools: {str(e)}")
            return f"Error generating response: {str(e)}", []
