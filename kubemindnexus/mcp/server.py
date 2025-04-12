"""MCP server implementation for KubeMindNexus."""

import asyncio
import json
import logging
import os
import subprocess
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

# Import the proper MCP client SDK components
try:
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.types import Implementation, ResourceContent
except ImportError:
    logging.warning("MCP SDK not found. Fallback to minimal functionality.")
    # Define minimal types if MCP SDK is not available
    class Implementation:
        """Server implementation information."""
        pass
    
    class ResourceContent:
        """Resource content returned by an MCP server."""
        def __init__(self, text="", uri="", mimeType=""):
            self.text = text
            self.uri = uri
            self.mimeType = mimeType

from ..constants import ServerType

logger = logging.getLogger(__name__)


class Tool:
    """Represents a tool with its properties."""

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


class MCPServer:
    """Base class for MCP server implementations."""
    
    def __init__(
        self,
        name: str,
        server_type: Union[str, ServerType],
        cluster_name: Optional[str] = None,
        is_local: bool = False,
    ):
        """Initialize MCP server.
        
        Args:
            name: Server name.
            server_type: Server type (stdio or sse).
            cluster_name: Optional cluster name.
            is_local: Whether this is a local server.
        """
        self.name = name
        self.server_type = ServerType(server_type) if isinstance(server_type, str) else server_type
        self.cluster_name = cluster_name
        self.is_local = is_local
        self.session: Optional[ClientSession] = None
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self._connected = False
        self._server_info: Optional[Implementation] = None
        self._tools: List[Tool] = []
        self._resources: List[Resource] = []
        self._resource_templates: List[ResourceTemplate] = []
    
    @property
    def is_connected(self) -> bool:
        """Check if the server is connected.
        
        Returns:
            True if the server is connected, False otherwise.
        """
        return self._connected and self.session is not None
    
    @property
    def server_info(self) -> Optional[Implementation]:
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
            True if connection was successful, False otherwise.
        """
        if self.is_connected:
            logger.warning(f"Server {self.name} is already connected")
            return True
            
        try:
            if not await self._connect_impl():
                return False
            self._connected = True

            # Query available tools and resources
            await self._load_capabilities()
            
            logger.info(f"Connected to MCP server: {self.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.name}: {str(e)}")
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
                logger.info(f"Disconnected from MCP server: {self.name}")
            except Exception as e:
                logger.error(f"Error during disconnect of server {self.name}: {e}")
    
    async def call_tool(
        self, tool_name: str, arguments: Dict[str, Any], retries: int = 2, delay: float = 1.0
    ) -> Tuple[bool, Any]:
        """Call a tool on the MCP server.
        
        Args:
            tool_name: Tool name.
            arguments: Tool arguments.
            retries: Number of retries on failure.
            delay: Delay between retries in seconds.
            
        Returns:
            (success, result) tuple.
        """
        if not self.is_connected or not self.session:
            return False, "Not connected to MCP server"
            
        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name} on {self.name}...")
                result = await self.session.call_tool(tool_name, arguments)
                
                # Extract result text from the response
                if not result.content:
                    return True, ""
                    
                result_texts = []
                for content in result.content:
                    if hasattr(content, "text"):
                        result_texts.append(content.text)
                    else:
                        # For non-text content, just convert to string
                        result_texts.append(str(content))
                
                return True, "\n".join(result_texts)
                
            except Exception as e:
                attempt += 1
                logger.warning(
                    f"Error calling tool {tool_name} on server {self.name}: {str(e)}. "
                    f"Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("Max retries reached. Failing.")
                    return False, str(e)
    
    async def read_resource(self, uri: str) -> Tuple[bool, Any]:
        """Read a resource from the MCP server.
        
        Args:
            uri: Resource URI.
            
        Returns:
            (success, content) tuple.
        """
        if not self.is_connected or not self.session:
            return False, "Not connected to MCP server"
            
        try:
            # Ensure uri is a string
            uri_str = str(uri) if uri is not None else ""
            if not uri_str:
                return False, "Resource URI cannot be empty"
                
            response = await self.session.read_resource(uri_str)
            
            # Extract content from the response
            if not response.contents or not response.contents[0].text:
                return True, ""
                
            return True, response.contents[0].text
            
        except Exception as e:
            logger.error(f"Error reading resource {uri} from server {self.name}: {str(e)}")
            return False, str(e)
    
    async def _connect_impl(self) -> bool:
        """Implement the connection to the server.
        
        This method should be overridden by subclasses.
        
        Returns:
            True if connection was successful, False otherwise.
        """
        # This is an abstract method intended to be overridden by subclasses
        # Each subclass (StdioMCPServer and SSEMCPServer) implements its own connection logic
        raise NotImplementedError("Subclasses must implement _connect_impl method")
    
    async def _load_capabilities(self) -> None:
        """Load server capabilities (tools, resources, etc.)."""
        if not self.is_connected or not self.session:
            return
            
        # Load tools
        try:
            tools_response = await self.session.list_tools()
            self._tools = []
            
            if hasattr(tools_response, "tools"):
                for tool in tools_response.tools:
                    self._tools.append(Tool(
                        getattr(tool, "name", "Unknown"), 
                        getattr(tool, "description", ""), 
                        getattr(tool, "inputSchema", {})
                    ))
                    
        except Exception as e:
            logger.error(f"Error loading tools from {self.name}: {str(e)}")
            self._tools = []
            
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
                    except Exception as res_err:
                        logger.warning(f"Error adding resource: {res_err}. Skipping.")
        except Exception as e:
            logger.error(f"Error loading resources from {self.name}: {str(e)}")
            self._resources = []
            
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
                    except Exception as template_err:
                        logger.warning(f"Error adding resource template: {template_err}. Skipping.")
        except Exception as e:
            logger.error(f"Error loading resource templates from {self.name}: {str(e)}")
            self._resource_templates = []
            
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


class StdioMCPServer(MCPServer):
    """MCP server that communicates over stdio."""
    
    def __init__(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cluster_name: Optional[str] = None,
        is_local: bool = False,
    ):
        """Initialize stdio MCP server.
        
        Args:
            name: Server name.
            command: Command to execute.
            args: Command arguments.
            env: Environment variables.
            cluster_name: Optional cluster name.
            is_local: Whether this is a local server.
        """
        super().__init__(name, ServerType.STDIO, cluster_name, is_local)
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.process: Optional[subprocess.Popen] = None
    
    async def _connect_impl(self) -> bool:
        """Connect to the stdio MCP server.
        
        Returns:
            True if connection was successful, False otherwise.
        """
        try:
            # Prepare environment variables
            full_env = os.environ.copy()
            full_env.update(self.env)
            
            # Construct command and args
            command = self.command
            args = self.args
            
            logging.info(f"Connecting to stdio server {self.name} with command {command}")
            stdio_params = (command, args, full_env)
            
            # Use AsyncExitStack to manage resources
            stdio_transport = await self._exit_stack.enter_async_context(
                stdio_client(stdio_params)
            )
            read, write = stdio_transport
            
            # Initialize the session
            self.session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            
            # Initialize the client
            init_result = await self.session.initialize()
            self._server_info = init_result.serverInfo
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to stdio MCP server {self.name}: {str(e)}")
            return False


class SSEMCPServer(MCPServer):
    """MCP server that communicates over SSE."""
    
    def __init__(
        self,
        name: str,
        url: str,
        cluster_name: Optional[str] = None,
        is_local: bool = False,
    ):
        """Initialize SSE MCP server.
        
        Args:
            name: Server name.
            url: Server URL.
            cluster_name: Optional cluster name.
            is_local: Whether this is a local server.
        """
        super().__init__(name, ServerType.SSE, cluster_name, is_local)
        self.url = url
    
    async def _connect_impl(self) -> bool:
        """Connect to the SSE MCP server.
        
        Returns:
            True if connection was successful, False otherwise.
        """
        try:
            url = self.url
            if not urlparse(url).scheme:
                logger.error(f"Invalid URL for SSE server {self.name}: {url}")
                return False
            
            logging.info(f"Connecting to SSE server {self.name} at {url}")
            
            # Use AsyncExitStack to manage resources
            streams = await self._exit_stack.enter_async_context(sse_client(url))
            read, write = streams
            
            # Initialize the session
            self.session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            
            # Initialize the client
            init_result = await self.session.initialize()
            self._server_info = init_result.serverInfo
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to SSE MCP server {self.name}: {str(e)}")
            return False
