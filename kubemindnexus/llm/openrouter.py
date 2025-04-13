"""OpenRouter LLM implementation for KubeMindNexus."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

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
