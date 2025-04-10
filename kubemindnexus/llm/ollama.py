"""Ollama LLM implementation for KubeMindNexus."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from ..constants import LLMProvider
from .base import BaseLLM, LLMMessage, MessageRole

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """Ollama LLM implementation."""
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Ollama LLM.
        
        Args:
            model: Model name.
            api_key: API key (not used by Ollama, but kept for consistency).
            base_url: Base URL for Ollama API.
            parameters: Additional parameters for the model.
        """
        super().__init__(LLMProvider.OLLAMA, model, api_key, base_url, parameters)
        
        # Set default base URL if not provided
        self.base_url = base_url or "http://localhost:11434"
        
        # Initialize HTTP client
        self.client = httpx.AsyncClient(base_url=self.base_url)
        
        logger.info(f"Initialized Ollama client with model: {model}, base_url: {self.base_url}")
    
    async def _convert_messages_to_prompt(
        self, messages: List[LLMMessage], system_prompt: Optional[str] = None
    ) -> str:
        """Convert messages to a prompt format that Ollama can understand.
        
        Args:
            messages: List of messages in the conversation.
            system_prompt: Optional system prompt to prepend to the conversation.
            
        Returns:
            Formatted prompt string.
        """
        parts = []
        
        # Add system prompt
        if system_prompt:
            parts.append(f"<system>\n{system_prompt}\n</system>\n\n")
        
        # Add conversation messages
        for message in messages:
            if message.role == MessageRole.SYSTEM:
                parts.append(f"<system>\n{message.content}\n</system>\n\n")
            elif message.role == MessageRole.USER:
                parts.append(f"<user>\n{message.content}\n</user>\n\n")
            elif message.role == MessageRole.ASSISTANT:
                parts.append(f"<assistant>\n{message.content}\n</assistant>\n\n")
            elif message.role == MessageRole.TOOL:
                # For tool messages, include the name if available
                name = message.name or "tool"
                parts.append(f"<{name}>\n{message.content}\n</{name}>\n\n")
        
        # Add assistant prompt
        parts.append("<assistant>\n")
        
        return "".join(parts)
    
    async def generate(
        self, messages: List[LLMMessage], system_prompt: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate a response from the Ollama LLM.
        
        Args:
            messages: List of messages in the conversation.
            system_prompt: Optional system prompt to prepend to the conversation.
            
        Returns:
            Tuple of (response_text, tool_calls) where tool_calls is a list of tool calls
            that were extracted from the response.
        """
        # Convert messages to prompt
        prompt = await self._convert_messages_to_prompt(messages, system_prompt)
        
        # Generate response
        try:
            # Build request parameters
            api_params = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                **self.parameters,
            }
            
            # Send request
            response = await self.client.post("/api/generate", json=api_params)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            response_text = result.get("response", "")
            
            # Extract tool calls using regex-based parsing
            # Ollama doesn't have native function calling, so we parse from text
            tool_calls = await self._extract_tool_calls(response_text)
            
            return response_text, tool_calls
            
        except Exception as e:
            logger.error(f"Error generating response from Ollama: {str(e)}")
            return f"Error generating response: {str(e)}", []
    
    async def generate_with_tools(
        self,
        messages: List[LLMMessage],
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate a response from the Ollama LLM with tool definitions.
        
        Note: Ollama doesn't natively support function calling or tools,
        so we add them to the system prompt and parse the response.
        
        Args:
            messages: List of messages in the conversation.
            tools: List of tool definitions.
            system_prompt: Optional system prompt to prepend to the conversation.
            
        Returns:
            Tuple of (response_text, tool_calls) where tool_calls is a list of tool calls
            that were extracted from the response.
        """
        # Create tool definitions in prompt format
        tools_prompt = "You have access to the following tools:\n\n"
        
        for tool in tools:
            tools_prompt += f"- {tool.get('name')}: {tool.get('description', '')}\n"
            
            # Add parameter information if available
            if "inputSchema" in tool and "properties" in tool["inputSchema"]:
                tools_prompt += "  Parameters:\n"
                
                for param_name, param_info in tool["inputSchema"]["properties"].items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    is_required = "required" in tool["inputSchema"] and param_name in tool["inputSchema"]["required"]
                    req_marker = " (required)" if is_required else " (optional)"
                    
                    tools_prompt += f"  - {param_name}: {param_type}{req_marker} - {param_desc}\n"
        
        # Add instructions on how to call tools
        tools_prompt += "\n"
        tools_prompt += "To use a tool, respond with JSON in the following format:\n"
        tools_prompt += "```json\n"
        tools_prompt += '{"tool": "tool_name", "params": {"param1": "value1", "param2": "value2", ...}}\n'
        tools_prompt += "```\n"
        tools_prompt += "Only respond with the tool call JSON when you want to use a tool.\n"
        
        # Combine with system prompt if provided
        combined_system_prompt = tools_prompt
        if system_prompt:
            combined_system_prompt = f"{system_prompt}\n\n{tools_prompt}"
        
        # Generate response with combined prompt
        response_text, tool_calls = await self.generate(messages, combined_system_prompt)
        
        # If no tool calls were found, try to extract them
        if not tool_calls:
            tool_calls = await self._extract_tool_calls(response_text)
            
        return response_text, tool_calls
    
    async def _extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls from text response.
        
        Args:
            text: Response text.
            
        Returns:
            List of extracted tool calls.
        """
        tool_calls = []
        
        # Try to find JSON tool calls in the text
        # Look for ```json blocks
        json_block_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
        json_blocks = re.finditer(json_block_pattern, text)
        
        for match in json_blocks:
            json_str = match.group(1)
            try:
                data = json.loads(json_str)
                
                # Check if this looks like a tool call
                if "tool" in data and "params" in data:
                    tool_call = {
                        "id": f"ollama-tool-{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": data["tool"],
                            "arguments": data["params"],
                        },
                    }
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                # Not valid JSON, skip
                continue
        
        # If no JSON blocks were found, try to find inline JSON
        if not tool_calls:
            inline_json_pattern = r"\{[\s\S]*?\"tool\"[\s\S]*?\"params\"[\s\S]*?\}"
            matches = re.finditer(inline_json_pattern, text)
            
            for match in matches:
                json_str = match.group(0)
                try:
                    data = json.loads(json_str)
                    
                    # Check if this looks like a tool call
                    if "tool" in data and "params" in data:
                        tool_call = {
                            "id": f"ollama-tool-{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": data["tool"],
                                "arguments": data["params"],
                            },
                        }
                        tool_calls.append(tool_call)
                except json.JSONDecodeError:
                    # Not valid JSON, skip
                    continue
        
        return tool_calls
