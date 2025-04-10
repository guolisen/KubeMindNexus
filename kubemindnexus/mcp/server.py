"""MCP server implementation for KubeMindNexus."""

import asyncio
import enum
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from mcp.client import ClientOptions, McpClient
from mcp.common import TextContent
from mcp.rpc import CallToolResponse, ReadResourceResponse

from ..constants import ServerType

logger = logging.getLogger(__name__)


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
        self.client: Optional[McpClient] = None
        self.connected = False
        self.available_tools: List[Dict[str, Any]] = []
        self.available_resources: List[Dict[str, Any]] = []
        self.resource_templates: List[Dict[str, Any]] = []
    
    async def connect(self) -> bool:
        """Connect to the MCP server.
        
        Returns:
            True if connection was successful, False otherwise.
        """
        if self.connected:
            return True
            
        try:
            if not await self._connect_impl():
                return False
                
            # Query available tools and resources
            await self._query_available_tools()
            await self._query_available_resources()
            
            self.connected = True
            logger.info(f"Connected to MCP server: {self.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.name}: {str(e)}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if not self.connected or not self.client:
            return
            
        try:
            await self.client.close()
            self.connected = False
            logger.info(f"Disconnected from MCP server: {self.name}")
            
        except Exception as e:
            logger.error(f"Error disconnecting from MCP server {self.name}: {str(e)}")
    
    async def call_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Tuple[bool, Any]:
        """Call a tool on the MCP server.
        
        Args:
            tool_name: Tool name.
            arguments: Tool arguments.
            
        Returns:
            (success, result) tuple.
        """
        if not self.connected or not self.client:
            return False, "Not connected to MCP server"
            
        try:
            response = await self.client.call_tool(tool_name, arguments)
            
            # Extract result text from the response
            if not response.content:
                return True, ""
                
            result_texts = []
            for content in response.content:
                if isinstance(content, TextContent):
                    result_texts.append(content.text)
                else:
                    # For non-text content, just convert to string
                    result_texts.append(str(content))
            
            return True, "\n".join(result_texts)
            
        except Exception as e:
            logger.error(f"Error calling tool {tool_name} on server {self.name}: {str(e)}")
            return False, str(e)
    
    async def read_resource(self, uri: str) -> Tuple[bool, Any]:
        """Read a resource from the MCP server.
        
        Args:
            uri: Resource URI.
            
        Returns:
            (success, content) tuple.
        """
        if not self.connected or not self.client:
            return False, "Not connected to MCP server"
            
        try:
            response = await self.client.read_resource(uri)
            
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
        raise NotImplementedError("Subclasses must implement _connect_impl")
    
    async def _query_available_tools(self) -> None:
        """Query available tools from the server."""
        if not self.client:
            return
            
        try:
            response = await self.client.list_tools()
            self.available_tools = response.tools or []
            
        except Exception as e:
            logger.error(f"Error querying tools from server {self.name}: {str(e)}")
            self.available_tools = []
    
    async def _query_available_resources(self) -> None:
        """Query available resources and resource templates from the server."""
        if not self.client:
            return
            
        # Query resources
        try:
            resources_response = await self.client.list_resources()
            self.available_resources = resources_response.resources or []
            
        except Exception as e:
            logger.error(f"Error querying resources from server {self.name}: {str(e)}")
            self.available_resources = []
            
        # Query resource templates
        try:
            templates_response = await self.client.list_resource_templates()
            self.resource_templates = templates_response.resourceTemplates or []
            
        except Exception as e:
            logger.error(f"Error querying resource templates from server {self.name}: {str(e)}")
            self.resource_templates = []


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
            
            # Start the process
            self.process = subprocess.Popen(
                [self.command] + self.args,
                env=full_env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,  # Binary mode for stdin/stdout
            )
            
            # Create client options
            options = ClientOptions()
            
            # Create MCP client
            self.client = McpClient(
                process_in=self.process.stdin,
                process_out=self.process.stdout,
                options=options,
            )
            
            # Connect client
            await self.client.connect()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to stdio MCP server {self.name}: {str(e)}")
            
            # Clean up process if needed
            if self.process:
                try:
                    self.process.kill()
                except:
                    pass
                    
                self.process = None
                
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the stdio MCP server."""
        await super().disconnect()
        
        # Clean up process
        if self.process:
            try:
                self.process.kill()
            except:
                pass
                
            self.process = None


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
            # Create client options
            options = ClientOptions()
            
            # Create MCP client
            self.client = McpClient(
                sse_url=self.url,
                options=options,
            )
            
            # Connect client
            await self.client.connect()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to SSE MCP server {self.name}: {str(e)}")
            return False
