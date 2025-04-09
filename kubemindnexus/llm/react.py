"""ReAct module for KubeMindNexus implementing Reasoning and Acting pattern."""
import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from ..config.settings import REACT_MAX_ITERATIONS
from ..mcp.cluster import ClusterManager
from ..utils.logger import LoggerMixin
from .client import LLMClient


class ReActEngine(LoggerMixin):
    """Implements the Reasoning + Acting pattern for LLM tool use."""
    
    def __init__(self, llm_client: LLMClient, cluster_manager: ClusterManager,
                max_iterations: int = REACT_MAX_ITERATIONS) -> None:
        """Initialize a ReActEngine instance.
        
        Args:
            llm_client: The LLM client to use.
            cluster_manager: The cluster manager to use for executing tools.
            max_iterations: Maximum number of ReAct iterations to prevent infinite loops.
        """
        self.llm_client = llm_client
        self.cluster_manager = cluster_manager
        self.max_iterations = max_iterations
    
    async def process(self, user_message: str, conversation_history: List[Dict[str, str]]) -> str:
        """Process a user message using the ReAct pattern.
        
        Args:
            user_message: The user's message.
            conversation_history: The conversation history.
            
        Returns:
            The final response from the LLM.
        """
        # Add user message to history
        history = conversation_history.copy()
        history.append({"role": "user", "content": user_message})
        
        # Add system message at the beginning if not present
        if not history or history[0]["role"] != "system":
            capabilities = await self._get_capabilities_prompt()
            history.insert(0, {
                "role": "system",
                "content": self._get_system_prompt() + capabilities
            })
        
        # Extract potential cluster name from the message
        cluster_name = await self.cluster_manager.extract_cluster_name_from_message(user_message)
        if cluster_name:
            self.logger.info(f"Detected cluster name in message: {cluster_name}")
            
        iterations = 0
        while iterations < self.max_iterations:
            iterations += 1
            self.logger.info(f"ReAct iteration {iterations}/{self.max_iterations}")
            
            # Get LLM response
            llm_response = await self.llm_client.get_response(history)
            
            # Check if the response contains a tool call
            tool_call = self._extract_tool_call(llm_response)
            
            if not tool_call:
                # No tool needed, return the final response
                self.logger.info("No tool call detected, returning response")
                return llm_response
            
            # Extract tool information
            tool_name = tool_call["tool"]
            arguments = tool_call["arguments"]
            
            self.logger.info(f"Tool call detected: {tool_name}")
            self.logger.info(f"Arguments: {json.dumps(arguments)}")
            
            # Add the LLM's decision to use a tool to the history
            history.append({"role": "assistant", "content": llm_response})
            
            # Execute tool
            try:
                self.logger.info(f"Executing tool {tool_name} with cluster {cluster_name or 'auto'}")
                tool_result = await self.cluster_manager.execute_tool(
                    tool_name, arguments, cluster_name=cluster_name
                )
                
                # Format tool result for the LLM
                result_str = self._format_tool_result(tool_result)
                self.logger.info(f"Tool result: {result_str}")
                
                # Add tool result to history
                history.append({"role": "system", "content": f"Tool result: {result_str}"})
                
            except Exception as e:
                error_message = f"Error executing tool {tool_name}: {str(e)}"
                self.logger.error(error_message)
                
                # Add error message to history
                history.append({"role": "system", "content": f"Tool execution error: {error_message}"})
        
        # If we reach max iterations, generate a final response
        self.logger.warning(f"Reached max iterations ({self.max_iterations}), generating final response")
        final_response = await self.llm_client.get_response(history)
        return final_response
    
    def _extract_tool_call(self, llm_response: str) -> Optional[Dict[str, Any]]:
        """Extract a tool call from an LLM response.
        
        Args:
            llm_response: The LLM response to analyze.
            
        Returns:
            A dict with tool name and arguments if a tool call is detected, None otherwise.
        """
        # Try to parse the entire response as JSON
        try:
            tool_call = json.loads(llm_response)
            if isinstance(tool_call, dict) and "tool" in tool_call and "arguments" in tool_call:
                return tool_call
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON object using regex
        json_pattern = r'```json\s*\n(.*?)\n\s*```|`?{.*?}`?'
        matches = re.findall(json_pattern, llm_response, re.DOTALL)
        
        for match in matches:
            try:
                match = match.strip()
                # Remove backticks if present
                if match.startswith('`') and match.endswith('`'):
                    match = match[1:-1]
                
                tool_call = json.loads(match)
                if isinstance(tool_call, dict) and "tool" in tool_call and "arguments" in tool_call:
                    return tool_call
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _format_tool_result(self, result: Any) -> str:
        """Format a tool result for the LLM.
        
        Args:
            result: The tool result to format.
            
        Returns:
            A formatted string representation of the tool result.
        """
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2)
        return str(result)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM.
        
        Returns:
            The system prompt.
        """
        return """You are KubeMindNexus, an AI assistant specialized in managing Kubernetes clusters.

When you need to perform actions on Kubernetes clusters or gather information, you can use the tools available to you. To use a tool, respond ONLY with a JSON object in the following format:
```json
{
    "tool": "tool_name",
    "arguments": {
        "arg1": "value1",
        "arg2": "value2"
    }
}
```

If you don't need to use a tool, respond conversationally to help the user with their question.

When a tool returns a result, analyze it carefully before providing a helpful, conversational response to the user. Transform raw data into insights that are easy to understand.

Here are the tools available to you:

"""
    
    async def _get_capabilities_prompt(self) -> str:
        """Get a prompt describing the capabilities of all connected clusters and servers.
        
        Returns:
            A prompt describing the capabilities.
        """
        parts = []
        
        # Add clusters
        clusters = self.cluster_manager.get_all_clusters()
        if clusters:
            parts.append("# Kubernetes Clusters")
            for cluster in clusters:
                connected_servers = cluster.get_connected_servers()
                parts.append(f"## {cluster.name}")
                for server in connected_servers:
                    parts.append(server.format_capabilities_for_llm())
        
        # Add local servers
        local_servers = self.cluster_manager.get_all_local_servers()
        if local_servers:
            parts.append("# Local Services")
            for server in local_servers:
                if server.is_connected:
                    parts.append(server.format_capabilities_for_llm())
        
        if not parts:
            return "No connected clusters or servers available."
        
        return "\n".join(parts)
