"""MCP server manager for KubeMindNexus."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from ..config import Configuration 
from ..database import DatabaseManager
from .server import MCPServer, SSEMCPServer, StdioMCPServer

logger = logging.getLogger(__name__)


class MCPManager:
    """Manager for MCP servers."""
    
    def __init__(
        self,
        config: Configuration,
        db_manager: DatabaseManager,
    ):
        """Initialize MCP manager.
        
        Args:
            config: Configuration instance.
            db_manager: Database manager instance.
        """
        self.config = config
        self.db_manager = db_manager
        self.servers: Dict[str, MCPServer] = {}
        self.connected_servers: Set[str] = set()
        
    async def initialize(self) -> None:
        """Initialize MCP manager."""
        # Load MCP server configurations from database
        await self._load_servers()
    
    async def _load_servers(self) -> None:
        """Load MCP servers from database."""
        try:
            # Get all MCP servers from database
            server_configs = self.db_manager.get_all_mcp_servers()
            
            for server_config in server_configs:
                server_id = server_config["id"]
                name = server_config["name"]
                server_type = server_config["type"]
                command = server_config["command"]
                args = server_config["args"]
                url = server_config["url"]
                cluster_id = server_config["cluster_id"]
                is_local = bool(server_config["is_local"])
                env = server_config["env"]
                
                # Get cluster name if cluster_id is provided
                cluster_name = None
                if cluster_id is not None:
                    cluster = self.db_manager.get_cluster(cluster_id)
                    if cluster:
                        cluster_name = cluster["name"]
                
                # Create server instance based on type
                server: MCPServer
                if server_type == "stdio":
                    if not command:
                        logger.warning(f"Missing command for stdio MCP server {name}")
                        continue
                        
                    server = StdioMCPServer(
                        name=name,
                        command=command,
                        args=args,
                        env=env,
                        cluster_name=cluster_name,
                        is_local=is_local,
                    )
                elif server_type == "sse":
                    if not url:
                        logger.warning(f"Missing URL for SSE MCP server {name}")
                        continue
                        
                    server = SSEMCPServer(
                        name=name,
                        url=url,
                        cluster_name=cluster_name,
                        is_local=is_local,
                    )
                else:
                    logger.warning(f"Unknown MCP server type: {server_type}")
                    continue
                
                # Add server to the manager
                self.servers[name] = server
                
                logger.info(f"Loaded MCP server: {name} ({server_type})")
            
            logger.info(f"Loaded {len(self.servers)} MCP servers")
            
        except Exception as e:
            logger.error(f"Error loading MCP servers: {str(e)}")
    
    async def connect_server(self, name: str) -> bool:
        """Connect to an MCP server.
        
        Args:
            name: Server name.
            
        Returns:
            True if connection was successful, False otherwise.
        """
        if name not in self.servers:
            logger.warning(f"MCP server not found: {name}")
            return False
            
        if name in self.connected_servers:
            logger.info(f"MCP server already connected: {name}")
            return True
            
        server = self.servers[name]
        success = await server.connect()
        
        if success:
            self.connected_servers.add(name)
            logger.info(f"Connected to MCP server: {name}")
        else:
            logger.error(f"Failed to connect to MCP server: {name}")
            
        return success
    
    async def disconnect_server(self, name: str) -> bool:
        """Disconnect from an MCP server.
        
        Args:
            name: Server name.
            
        Returns:
            True if disconnection was successful, False otherwise.
        """
        if name not in self.servers:
            logger.warning(f"MCP server not found: {name}")
            return False
            
        if name not in self.connected_servers:
            logger.info(f"MCP server not connected: {name}")
            return True
            
        server = self.servers[name]
        
        try:
            await server.disconnect()
            self.connected_servers.remove(name)
            logger.info(f"Disconnected from MCP server: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from MCP server {name}: {str(e)}")
            return False
    
    async def connect_all_servers(self) -> None:
        """Connect to all MCP servers."""
        for name in self.servers:
            await self.connect_server(name)
    
    async def connect_default_servers(self) -> None:
        """Connect to default MCP servers."""
        # Get default server names from configuration
        default_servers = self.config.get_default_mcp_servers()
        
        for name in default_servers:
            if name in self.servers:
                await self.connect_server(name)
            else:
                logger.warning(f"Default MCP server not found: {name}")
    
    async def disconnect_all_servers(self) -> None:
        """Disconnect from all MCP servers."""
        for name in list(self.connected_servers):
            await self.disconnect_server(name)
    
    async def call_tool(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any]
    ) -> Tuple[bool, Any]:
        """Call a tool on an MCP server.
        
        Args:
            server_name: Server name.
            tool_name: Tool name.
            arguments: Tool arguments.
            
        Returns:
            (success, result) tuple.
        """
        if server_name not in self.servers:
            return False, f"MCP server not found: {server_name}"
            
        if server_name not in self.connected_servers:
            # Try to connect to the server
            if not await self.connect_server(server_name):
                return False, f"Failed to connect to MCP server: {server_name}"
                
        server = self.servers[server_name]
        return await server.call_tool(tool_name, arguments)
    
    async def read_resource(self, server_name: str, uri: str) -> Tuple[bool, Any]:
        """Read a resource from an MCP server.
        
        Args:
            server_name: Server name.
            uri: Resource URI.
            
        Returns:
            (success, content) tuple.
        """
        if server_name not in self.servers:
            return False, f"MCP server not found: {server_name}"
            
        if server_name not in self.connected_servers:
            # Try to connect to the server
            if not await self.connect_server(server_name):
                return False, f"Failed to connect to MCP server: {server_name}"
                
        server = self.servers[server_name]
        return await server.read_resource(uri)
    
    def get_server_by_name(self, name: str) -> Optional[MCPServer]:
        """Get an MCP server by name.
        
        Args:
            name: Server name.
            
        Returns:
            MCP server instance, or None if not found.
        """
        return self.servers.get(name)
    
    def get_servers_by_cluster(self, cluster_name: str) -> List[MCPServer]:
        """Get MCP servers for a specific cluster.
        
        Args:
            cluster_name: Cluster name.
            
        Returns:
            List of MCP server instances.
        """
        return [
            server for server in self.servers.values()
            if server.cluster_name == cluster_name
        ]
    
    def get_local_servers(self) -> List[MCPServer]:
        """Get local MCP servers.
        
        Returns:
            List of local MCP server instances.
        """
        return [
            server for server in self.servers.values()
            if server.is_local
        ]
    
    def get_connected_servers(self) -> List[MCPServer]:
        """Get connected MCP servers.
        
        Returns:
            List of connected MCP server instances.
        """
        return [
            self.servers[name] for name in self.connected_servers
            if name in self.servers
        ]
    
    def is_server_connected(self, name: str) -> bool:
        """Check if an MCP server is connected.
        
        Args:
            name: Server name.
            
        Returns:
            True if the server is connected, False otherwise.
        """
        return name in self.connected_servers
    
    async def add_server(
        self,
        name: str,
        server_type: str,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        url: Optional[str] = None,
        cluster_id: Optional[int] = None,
        is_local: bool = False,
        is_default: bool = False,
        env: Optional[Dict[str, str]] = None,
    ) -> int:
        """Add a new MCP server.
        
        Args:
            name: Server name.
            server_type: Server type (stdio or sse).
            command: Command to execute (for stdio servers).
            args: Command arguments (for stdio servers).
            url: Server URL (for sse servers).
            cluster_id: Optional cluster ID to associate with.
            is_local: Whether this is a local server.
            is_default: Whether this is a default server.
            env: Environment variables for the server.
            
        Returns:
            Server ID.
        """
        # Add server to database
        server_id = self.db_manager.add_mcp_server(
            name=name,
            type_str=server_type,
            command=command,
            args=args,
            url=url,
            cluster_id=cluster_id,
            is_local=is_local,
            is_default=is_default,
            env=env,
        )
        
        # Get cluster name if cluster_id is provided
        cluster_name = None
        if cluster_id is not None:
            cluster = self.db_manager.get_cluster(cluster_id)
            if cluster:
                cluster_name = cluster["name"]
        
        # Create server instance based on type
        server: MCPServer
        if server_type == "stdio":
            if not command:
                raise ValueError("Command is required for stdio MCP server")
                
            server = StdioMCPServer(
                name=name,
                command=command,
                args=args,
                env=env,
                cluster_name=cluster_name,
                is_local=is_local,
            )
        elif server_type == "sse":
            if not url:
                raise ValueError("URL is required for SSE MCP server")
                
            server = SSEMCPServer(
                name=name,
                url=url,
                cluster_name=cluster_name,
                is_local=is_local,
            )
        else:
            raise ValueError(f"Unknown MCP server type: {server_type}")
        
        # Add server to the manager
        self.servers[name] = server
        
        logger.info(f"Added MCP server: {name} ({server_type})")
        
        return server_id
    
    async def update_server(
        self,
        server_id: int,
        name: Optional[str] = None,
        server_type: Optional[str] = None,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        url: Optional[str] = None,
        cluster_id: Optional[int] = None,
        is_local: Optional[bool] = None,
        is_default: Optional[bool] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Update an MCP server.
        
        Args:
            server_id: Server ID.
            name: New server name.
            server_type: New server type.
            command: New command.
            args: New command arguments.
            url: New server URL.
            cluster_id: New cluster ID.
            is_local: New local flag.
            is_default: New default flag.
            env: New environment variables.
            
        Returns:
            True if the update was successful, False otherwise.
        """
        # Get current server configuration
        server_config = self.db_manager.get_mcp_server(server_id)
        if not server_config:
            logger.warning(f"MCP server not found: {server_id}")
            return False
            
        current_name = server_config["name"]
        
        # Disconnect server if it's connected
        if current_name in self.connected_servers:
            await self.disconnect_server(current_name)
        
        # Update server in database
        success = self.db_manager.update_mcp_server(
            server_id=server_id,
            name=name,
            type=server_type,
            command=command,
            args=args,
            url=url,
            cluster_id=cluster_id,
            is_local=is_local,
            is_default=is_default,
            env=env,
        )
        
        if not success:
            logger.error(f"Failed to update MCP server in database: {server_id}")
            return False
            
        # Remove old server from manager
        if current_name in self.servers:
            del self.servers[current_name]
            
        # Get updated server configuration
        updated_config = self.db_manager.get_mcp_server(server_id)
        if not updated_config:
            logger.error(f"Failed to get updated MCP server configuration: {server_id}")
            return False
            
        # Get cluster name if cluster_id is provided
        cluster_name = None
        if updated_config["cluster_id"] is not None:
            cluster = self.db_manager.get_cluster(updated_config["cluster_id"])
            if cluster:
                cluster_name = cluster["name"]
        
        # Create new server instance based on updated configuration
        new_name = updated_config["name"]
        new_type = updated_config["type"]
        
        server: MCPServer
        if new_type == "stdio":
            if not updated_config["command"]:
                logger.error("Command is required for stdio MCP server")
                return False
                
            server = StdioMCPServer(
                name=new_name,
                command=updated_config["command"],
                args=updated_config["args"],
                env=updated_config["env"],
                cluster_name=cluster_name,
                is_local=bool(updated_config["is_local"]),
            )
        elif new_type == "sse":
            if not updated_config["url"]:
                logger.error("URL is required for SSE MCP server")
                return False
                
            server = SSEMCPServer(
                name=new_name,
                url=updated_config["url"],
                cluster_name=cluster_name,
                is_local=bool(updated_config["is_local"]),
            )
        else:
            logger.error(f"Unknown MCP server type: {new_type}")
            return False
        
        # Add updated server to the manager
        self.servers[new_name] = server
        
        logger.info(f"Updated MCP server: {new_name} ({new_type})")
        
        return True
    
    async def remove_server(self, server_id: int) -> bool:
        """Remove an MCP server.
        
        Args:
            server_id: Server ID.
            
        Returns:
            True if the removal was successful, False otherwise.
        """
        # Get server configuration
        server_config = self.db_manager.get_mcp_server(server_id)
        if not server_config:
            logger.warning(f"MCP server not found: {server_id}")
            return False
            
        name = server_config["name"]
        
        # Disconnect server if it's connected
        if name in self.connected_servers:
            await self.disconnect_server(name)
        
        # Remove server from manager
        if name in self.servers:
            del self.servers[name]
        
        # Remove server from database
        success = self.db_manager.delete_mcp_server(server_id)
        
        if success:
            logger.info(f"Removed MCP server: {name}")
        else:
            logger.error(f"Failed to remove MCP server from database: {server_id}")
            
        return success
