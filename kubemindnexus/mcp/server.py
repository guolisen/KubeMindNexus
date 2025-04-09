"""MCP server module for KubeMindNexus."""
import asyncio
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from urllib.parse import urlparse

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

from ..config.settings import ServerConfigModel, MAX_MCP_RETRIES, MCP_RETRY_DELAY
from ..utils.logger import LoggerMixin


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self, name: str, description: str, input_schema: Dict[str, Any]
    ) -> None:
        """Initialize a Tool instance.
        
        Args:
            name: The name of the tool.
            description: The description of the tool.
            input_schema: The JSON schema for the tool's input.
        """
        self.name: str = name
        self.description: str = description
        self.input_schema: Dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""

    def __repr__(self) -> str:
        """Get string representation of the tool.
        
        Returns:
            String representation.
        """
        return f"Tool(name={self.name}, description={self.description})"


class Resource:
    """Represents a resource with its properties."""

    def __init__(
        self, uri: str, name: str, mime_type: Optional[str] = None, description: Optional[str] = None
    ) -> None:
        """Initialize a Resource instance.
        
        Args:
            uri: The URI of the resource.
            name: The name of the resource.
            mime_type: The MIME type of the resource.
            description: The description of the resource.
        """
        self.uri: str = str(uri) if uri is not None else ""
        self.name: str = name
        self.mime_type: Optional[str] = mime_type
        self.description: Optional[str] = description

    def format_for_llm(self) -> str:
        """Format resource information for LLM.

        Returns:
            A formatted string describing the resource.
        """
        return f"""
Resource: {self.name}
URI: {self.uri}
MIME Type: {self.mime_type or 'Not specified'}
Description: {self.description or 'No description'}
"""

    def __repr__(self) -> str:
        """Get string representation of the resource.
        
        Returns:
            String representation.
        """
        return f"Resource(name={self.name}, uri={self.uri})"


class ResourceTemplate:
    """Represents a resource template with its properties."""

    def __init__(
        self, uri_template: str, name: str, mime_type: Optional[str] = None, description: Optional[str] = None
    ) -> None:
        """Initialize a ResourceTemplate instance.
        
        Args:
            uri_template: The URI template of the resource.
            name: The name of the resource.
            mime_type: The MIME type of the resource.
            description: The description of the resource.
        """
        self.uri_template: str = str(uri_template) if uri_template is not None else ""
        self.name: str = name
        self.mime_type: Optional[str] = mime_type
        self.description: Optional[str] = description

    def format_for_llm(self) -> str:
        """Format resource template information for LLM.

        Returns:
            A formatted string describing the resource template.
        """
        return f"""
Resource Template: {self.name}
URI Template: {self.uri_template}
MIME Type: {self.mime_type or 'Not specified'}
Description: {self.description or 'No description'}
"""

    def __repr__(self) -> str:
        """Get string representation of the resource template.
        
        Returns:
            String representation.
        """
        return f"ResourceTemplate(name={self.name}, uri_template={self.uri_template})"


class MCPServer(LoggerMixin):
    """Manages connection to an MCP server and tool execution."""

    def __init__(self, name: str, config: ServerConfigModel) -> None:
        """Initialize an MCPServer instance.
        
        Args:
            name: The name of the server.
            config: The server configuration.
        """
        self.name: str = name
        self.config: ServerConfigModel = config
        self.session: Optional[ClientSession] = None
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self._tools: List[Tool] = []
        self._resources: List[Resource] = []
        self._resource_templates: List[ResourceTemplate] = []
        self._connected: bool = False
        self._server_info: Optional[Any] = None  # Implementation object from MCP

    @property
    def is_connected(self) -> bool:
        """Check if the server is connected.
        
        Returns:
            True if the server is connected, False otherwise.
        """
        return self._connected and self.session is not None

    @property
    def server_info(self) -> Optional[Any]:
        """Get the server info.
        
        Returns:
            The server info if available, None otherwise.
        """
        return self._server_info
    
    @property
    def tools(self) -> List[Tool]:
        """Get the list of available tools.
        
        Returns:
            The list of available tools.
        """
        return self._tools
    
    @property
    def resources(self) -> List[Resource]:
        """Get the list of available resources.
        
        Returns:
            The list of available resources.
        """
        return self._resources
    
    @property
    def resource_templates(self) -> List[ResourceTemplate]:
        """Get the list of available resource templates.
        
        Returns:
            The list of available resource templates.
        """
        return self._resource_templates

    async def connect(self) -> bool:
        """Connect to the MCP server.
        
        Returns:
            True if the connection was successful, False otherwise.
        """
        if self.is_connected:
            self.logger.warning(f"Server {self.name} is already connected")
            return True
        
        try:
            if self.config.type.lower() == "sse":
                if not self.config.url:
                    self.logger.error(f"URL is required for SSE server {self.name}")
                    return False
                
                url = self.config.url
                if not urlparse(url).scheme:
                    self.logger.error(f"Invalid URL for SSE server {self.name}: {url}")
                    return False
                
                self.logger.info(f"Connecting to SSE server {self.name} at {url}")
                streams = await self._exit_stack.enter_async_context(sse_client(url))
                read, write = streams
            
            elif self.config.type.lower() == "stdio":
                if not self.config.command:
                    self.logger.error(f"Command is required for stdio server {self.name}")
                    return False
                
                command = (
                    shutil.which("npx")
                    if self.config.command == "npx"
                    else self.config.command
                )
                if command is None:
                    self.logger.error(f"Command {self.config.command} not found")
                    return False
                
                args = self.config.args
                env = {**os.environ, **self.config.env} if self.config.env else None
                
                self.logger.info(f"Connecting to stdio server {self.name} with command {command}")
                stdio_transport = await self._exit_stack.enter_async_context(
                    stdio_client((command, args, env))
                )
                read, write = stdio_transport
            
            else:
                self.logger.error(f"Unsupported server type for {self.name}: {self.config.type}")
                return False
            
            # Initialize the session
            self.session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            init_result = await self.session.initialize()
            self._server_info = init_result.serverInfo
            
            self.logger.info(f"Connected to MCP server {self.name}")
            self._connected = True
            
            # Load the server capabilities
            await self._load_capabilities()
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error connecting to MCP server {self.name}: {e}")
            await self.disconnect()
            return False

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        async with self._cleanup_lock:
            try:
                await self._exit_stack.aclose()
                self.session = None
                self._connected = False
                self._tools = []
                self._resources = []
                self._resource_templates = []
                self._server_info = None
                self.logger.info(f"Disconnected from MCP server {self.name}")
            except Exception as e:
                self.logger.error(f"Error during disconnect of server {self.name}: {e}")

    async def _load_capabilities(self) -> None:
        """Load server capabilities (tools, resources, etc.)."""
        if not self.is_connected or not self.session:
            raise RuntimeError(f"Server {self.name} not connected")
        
        # Load tools
        try:
            tools_response = await self.session.list_tools()
            self._tools = []
            
            if hasattr(tools_response, "tools"):
                for tool in tools_response.tools:
                    self._tools.append(Tool(tool.name, tool.description, tool.inputSchema))
                    self.logger.info(f"Found tool: {tool.name}")
        except Exception as e:
            self.logger.error(f"Error loading tools from {self.name}: {e}")
        
        # Load resources
        try:
            resources_response = await self.session.list_resources()
            self._resources = []
            
            if hasattr(resources_response, "resources"):
                for resource in resources_response.resources:
                    try:
                        self._resources.append(
                            Resource(
                                getattr(resource, "uri", ""), 
                                getattr(resource, "name", "Unknown"),
                                getattr(resource, "mimeType", None),
                                getattr(resource, "description", None)
                            )
                        )
                        self.logger.info(f"Found resource: {resource.name}")
                    except Exception as res_err:
                        self.logger.warning(f"Error adding resource: {res_err}. Skipping.")
        except Exception as e:
            self.logger.error(f"Error loading resources from {self.name}: {e}")
        
        # Load resource templates
        try:
            templates_response = await self.session.list_resource_templates()
            self._resource_templates = []
            
            if hasattr(templates_response, "resourceTemplates"):
                for template in templates_response.resourceTemplates:
                    try:
                        self._resource_templates.append(
                            ResourceTemplate(
                                getattr(template, "uriTemplate", ""), 
                                getattr(template, "name", "Unknown"),
                                getattr(template, "mimeType", None),
                                getattr(template, "description", None)
                            )
                        )
                        self.logger.info(f"Found resource template: {template.name}")
                    except Exception as template_err:
                        self.logger.warning(f"Error adding resource template: {template_err}. Skipping.")
        except Exception as e:
            self.logger.error(f"Error loading resource templates from {self.name}: {e}")

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        retries: int = MAX_MCP_RETRIES,
        delay: float = MCP_RETRY_DELAY,
    ) -> Any:
        """Execute a tool.
        
        Args:
            tool_name: The name of the tool to execute.
            arguments: The arguments to pass to the tool.
            retries: The number of retries.
            delay: The delay between retries.
            
        Returns:
            The result of the tool execution.
            
        Raises:
            RuntimeError: If the server is not connected.
            Exception: If the tool execution fails.
        """
        if not self.is_connected or not self.session:
            raise RuntimeError(f"Server {self.name} not connected")
        
        attempt = 0
        last_exception = None
        
        while attempt < retries:
            try:
                self.logger.info(f"Executing {tool_name} on {self.name}...")
                result = await self.session.call_tool(tool_name, arguments)
                return result
            
            except Exception as e:
                attempt += 1
                last_exception = e
                self.logger.warning(
                    f"Error executing tool {tool_name}: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    self.logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
        
        self.logger.error(f"Max retries reached for tool {tool_name}. Failing.")
        if last_exception:
            raise last_exception
        raise RuntimeError(f"Failed to execute tool {tool_name} after {retries} attempts")

    async def read_resource(self, uri: str) -> Any:
        """Read a resource from the server.
        
        Args:
            uri: The URI of the resource to read.
            
        Returns:
            The content of the resource.
            
        Raises:
            RuntimeError: If the server is not connected.
            Exception: If the resource read fails.
        """
        if not self.is_connected or not self.session:
            raise RuntimeError(f"Server {self.name} not connected")
        
        # Ensure uri is a string
        uri_str = str(uri) if uri is not None else ""
        if not uri_str:
            raise ValueError("Resource URI cannot be empty")
            
        try:
            result = await self.session.read_resource(uri_str)
            return result
        except Exception as e:
            self.logger.error(f"Error reading resource {uri_str} from {self.name}: {e}")
            raise

    async def has_tool(self, tool_name: str) -> bool:
        """Check if the server has a tool with the given name.
        
        Args:
            tool_name: The name of the tool to check.
            
        Returns:
            True if the server has the tool, False otherwise.
        """
        return any(tool.name == tool_name for tool in self._tools)

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name.
        
        Args:
            tool_name: The name of the tool to get.
            
        Returns:
            The tool if found, None otherwise.
        """
        for tool in self._tools:
            if tool.name == tool_name:
                return tool
        return None

    def format_capabilities_for_llm(self) -> str:
        """Format server capabilities for LLM.
        
        Returns:
            A formatted string describing the server's capabilities.
        """
        parts = [f"## MCP Server: {self.name}"]
        
        if self._tools:
            parts.append("### Tools:")
            for tool in self._tools:
                parts.append(tool.format_for_llm())
        
        if self._resources:
            parts.append("### Resources:")
            for resource in self._resources:
                parts.append(resource.format_for_llm())
        
        if self._resource_templates:
            parts.append("### Resource Templates:")
            for template in self._resource_templates:
                parts.append(template.format_for_llm())
        
        return "\n".join(parts)
    
    def __repr__(self) -> str:
        """Get string representation of the server.
        
        Returns:
            String representation.
        """
        return f"MCPServer(name={self.name}, type={self.config.type}, connected={self.is_connected})"
