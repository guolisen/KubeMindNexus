"""MCP hub for KubeMindNexus.

This module provides a central hub for MCP server functionality, enabling
tool execution across multiple MCP servers.
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from ..config import Configuration
from ..database import DatabaseManager
from .manager import MCPManager
from .server import MCPServer

logger = logging.getLogger(__name__)


class MCPHub:
    """Central hub for MCP server functionality."""
    
    def __init__(
        self,
        config: Configuration,
        db_manager: DatabaseManager,
    ):
        """Initialize MCP hub.
        
        Args:
            config: Configuration instance.
            db_manager: Database manager instance.
        """
        self.config = config
        self.db_manager = db_manager
        self.manager = MCPManager(config, db_manager)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize MCP hub."""
        await self.manager.initialize()
        await self.manager.connect_default_servers()
        self._initialized = True
    
    async def shutdown(self) -> None:
        """Shutdown MCP hub."""
        await self.manager.disconnect_all_servers()
    
    async def execute_tool(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any], 
        chat_id: Optional[int] = None
    ) -> Tuple[bool, Any]:
        """Execute a tool on an MCP server.
        
        Args:
            server_name: Server name.
            tool_name: Tool name.
            arguments: Tool arguments.
            chat_id: Optional chat ID for logging.
            
        Returns:
            (success, result) tuple.
        """
        # Record start time
        start_time = asyncio.get_event_loop().time()
        
        # Call the tool
        success, result = await self.manager.call_tool(server_name, tool_name, arguments)
        
        # Calculate execution time
        execution_time = asyncio.get_event_loop().time() - start_time
        
        # Log tool execution if chat_id provided
        if chat_id is not None:
            try:
                self.db_manager.add_tool_execution(
                    chat_id=chat_id,
                    tool_name=tool_name,
                    server_name=server_name,
                    arguments=arguments,
                    result=result,
                    execution_time=execution_time,
                )
            except Exception as e:
                logger.error(f"Error logging tool execution: {str(e)}")
        
        return success, result
    
    async def access_resource(
        self, server_name: str, uri: str
    ) -> Tuple[bool, Any]:
        """Access a resource from an MCP server.
        
        Args:
            server_name: Server name.
            uri: Resource URI.
            
        Returns:
            (success, content) tuple.
        """
        return await self.manager.read_resource(server_name, uri)
    
    def find_server_for_tool(self, tool_name: str) -> Optional[str]:
        """Find a server that provides a specific tool.
        
        Args:
            tool_name: Tool name.
            
        Returns:
            Server name, or None if not found.
        """
        # Get all connected servers
        connected_servers = self.manager.get_connected_servers()
        
        for server in connected_servers:
            for tool in server.available_tools:
                if tool.get("name") == tool_name:
                    return server.name
        
        return None
    
    def get_all_available_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all available tools from connected servers.
        
        Returns:
            Dictionary mapping server names to lists of available tools.
        """
        result = {}
        
        # Get all connected servers
        connected_servers = self.manager.get_connected_servers()
        
        for server in connected_servers:
            if server.available_tools:
                result[server.name] = server.available_tools
        
        return result
    
    def get_tools_by_cluster(self, cluster_name: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get available tools for a specific cluster.
        
        Args:
            cluster_name: Cluster name.
            
        Returns:
            Dictionary mapping server names to lists of available tools.
        """
        result = {}
        
        # Get servers for the cluster
        cluster_servers = self.manager.get_servers_by_cluster(cluster_name)
        
        for server in cluster_servers:
            if server.name in self.manager.connected_servers and server.available_tools:
                result[server.name] = server.available_tools
        
        return result
    
    def get_local_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get available tools from local servers.
        
        Returns:
            Dictionary mapping server names to lists of available tools.
        """
        result = {}
        
        # Get local servers
        local_servers = self.manager.get_local_servers()
        
        for server in local_servers:
            if server.name in self.manager.connected_servers and server.available_tools:
                result[server.name] = server.available_tools
        
        return result
    
    def get_all_available_resources(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all available resources from connected servers.
        
        Returns:
            Dictionary mapping server names to lists of available resources.
        """
        result = {}
        
        # Get all connected servers
        connected_servers = self.manager.get_connected_servers()
        
        for server in connected_servers:
            resources = []
            
            # Add direct resources
            if server.available_resources:
                resources.extend(server.available_resources)
                
            # Add resource templates
            if server.resource_templates:
                resources.extend(server.resource_templates)
                
            if resources:
                result[server.name] = resources
        
        return result
    
    async def analyze_user_message(
        self, message: str
    ) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """Analyze a user message to extract tool calls.
        
        This function looks for patterns like "use <cluster> to <action>" or
        similar patterns that would indicate which cluster/server the user
        wants to use.
        
        Args:
            message: User message.
            
        Returns:
            Tuple of (server_name, tool_name, arguments) if a tool call is detected,
            None otherwise.
        """
        # Check for cluster name mentions
        cluster_pattern = r"(?:use|check|on|in|for|from)\s+(?:the\s+)?([A-Za-z0-9_-]+)\s+(?:cluster|server)"
        cluster_matches = re.finditer(cluster_pattern, message, re.IGNORECASE)
        
        for match in cluster_matches:
            cluster_name = match.group(1)
            
            # Check if this cluster exists in the database
            cluster = self.db_manager.get_cluster_by_name(cluster_name)
            if cluster:
                # Get servers for this cluster
                cluster_servers = self.manager.get_servers_by_cluster(cluster_name)
                
                if cluster_servers:
                    # For simplicity, just return the first server and a default tool
                    # In a real implementation, more sophisticated NLP would be used
                    server = cluster_servers[0]
                    
                    # Find a suitable tool (e.g. get_status, get_info)
                    tool_name = None
                    for tool in server.available_tools:
                        if "status" in tool.get("name", "").lower() or "info" in tool.get("name", "").lower():
                            tool_name = tool.get("name")
                            break
                    
                    if tool_name:
                        return server.name, tool_name, {}
        
        # No specific cluster/tool found
        return None
    
    def format_tools_for_prompt(self) -> str:
        """Format available tools for use in an LLM prompt.
        
        Returns:
            Formatted string describing available tools.
        """
        tools_by_server = self.get_all_available_tools()
        
        if not tools_by_server:
            return "No tools available."
            
        sections = []
        
        for server_name, tools in tools_by_server.items():
            server = self.manager.get_server_by_name(server_name)
            
            if not server:
                continue
                
            # Create server section with metadata
            section_info = []
            if server.cluster_name:
                section_title = f"Cluster '{server.cluster_name}' - Server '{server_name}'"
                section_info.append(f"Cluster Context: {server.cluster_name}")
            else:
                section_title = f"Server '{server_name}'"
                
            if server.is_local:
                section_info.append("Type: Local Server")
            else:
                section_info.append("Type: Remote Server")
                
            # Add server metadata if available
            section_header = [section_title]
            if section_info:
                section_header.append("  " + ", ".join(section_info))
                
            tool_descriptions = []
            
            for tool in tools:
                name = tool.get("name", "unknown")
                description = tool.get("description", "No description available")
                
                # Format input schema information if available
                input_params = []
                input_schema = tool.get("inputSchema", {})
                properties = input_schema.get("properties", {})
                required = input_schema.get("required", [])
                
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    is_required = param_name in required
                    req_marker = " (required)" if is_required else " (optional)"
                    
                    input_params.append(f"    - {param_name}: {param_type}{req_marker} - {param_desc}")
                
                # Format tool description
                tool_desc = f"  * {name}: {description}"
                if input_params:
                    tool_desc += "\n    Parameters:\n" + "\n".join(input_params)
                    
                # Add example usage in JSON format
                if properties:
                    example = {
                        "tool": name,
                        "parameters": {}
                    }
                    
                    # Add example values for required parameters
                    for param_name, param_info in properties.items():
                        if param_name in required:
                            param_type = param_info.get("type", "string")
                            example_value = ""
                            
                            if param_type == "string":
                                example_value = f"<{param_name}>"
                            elif param_type == "number" or param_type == "integer":
                                example_value = 0
                            elif param_type == "boolean":
                                example_value = False
                            elif param_type == "array":
                                example_value = []
                            elif param_type == "object":
                                example_value = {}
                                
                            example["parameters"][param_name] = example_value
                    
                    # Format the example JSON
                    try:
                        example_json = json.dumps(example, indent=2)
                        tool_desc += f"\n    Example Usage:\n    ```json\n{example_json}\n    ```"
                    except Exception as e:
                        logger.warning(f"Error formatting example JSON for tool {name}: {str(e)}")
                    
                tool_descriptions.append(tool_desc)
                
            # Add server section to sections
            sections.append("\n".join(section_header) + "\n\n" + "\n\n".join(tool_descriptions))
        
        return "\n\n".join(sections)
