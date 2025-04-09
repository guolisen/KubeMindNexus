"""MCP server cluster module for KubeMindNexus."""
import asyncio
from typing import Dict, List, Optional, Any, Set

from ..config.settings import ClusterConfigModel
from ..utils.logger import LoggerMixin
from .server import MCPServer, ServerConfigModel


class Cluster(LoggerMixin):
    """Represents a Kubernetes cluster with associated MCP servers."""
    
    def __init__(self, name: str, config: ClusterConfigModel) -> None:
        """Initialize a Cluster instance.
        
        Args:
            name: Name of the cluster.
            config: Configuration for the cluster.
        """
        self.name = name
        self.config = config
        self.ip = config.ip
        self.port = config.port
        self.description = config.description
        self.servers: Dict[str, MCPServer] = {}
        self._initialize_servers()
    
    def _initialize_servers(self) -> None:
        """Initialize MCP servers for this cluster."""
        for server_name, server_config in self.config.servers.items():
            self.servers[server_name] = MCPServer(server_name, server_config)
            self.logger.info(f"Initialized server {server_name} for cluster {self.name}")
    
    async def connect(self) -> Dict[str, bool]:
        """Connect to all servers for this cluster.
        
        Returns:
            Dictionary of server names to connection success.
        """
        results = {}
        for name, server in self.servers.items():
            try:
                self.logger.info(f"Connecting to server {name} for cluster {self.name}")
                results[name] = await server.connect()
                if results[name]:
                    self.logger.info(f"Connected to server {name} for cluster {self.name}")
                else:
                    self.logger.error(f"Failed to connect to server {name} for cluster {self.name}")
            except Exception as e:
                self.logger.error(f"Error connecting to server {name} for cluster {self.name}: {e}")
                results[name] = False
        
        return results
    
    async def disconnect(self) -> None:
        """Disconnect from all servers for this cluster."""
        for name, server in self.servers.items():
            try:
                self.logger.info(f"Disconnecting from server {name} for cluster {self.name}")
                await server.disconnect()
                self.logger.info(f"Disconnected from server {name} for cluster {self.name}")
            except Exception as e:
                self.logger.error(f"Error disconnecting from server {name} for cluster {self.name}: {e}")
    
    def get_connected_servers(self) -> List[MCPServer]:
        """Get all connected servers for this cluster.
        
        Returns:
            List of connected MCP servers.
        """
        return [server for server in self.servers.values() if server.is_connected]
    
    async def has_tool(self, tool_name: str) -> bool:
        """Check if any server in the cluster has a tool with the given name.
        
        Args:
            tool_name: Name of the tool to check.
            
        Returns:
            True if any server has the tool, False otherwise.
        """
        for server in self.get_connected_servers():
            if await server.has_tool(tool_name):
                return True
        return False
    
    async def get_server_for_tool(self, tool_name: str) -> Optional[MCPServer]:
        """Get the appropriate server for a specific tool.
        
        Args:
            tool_name: Name of the tool to find a server for.
            
        Returns:
            The appropriate MCPServer or None if not found.
        """
        for server in self.get_connected_servers():
            if await server.has_tool(tool_name):
                return server
        return None
    
    async def get_all_tools(self) -> List[Any]:
        """Get all tools from all servers in the cluster.
        
        Returns:
            List of all tools.
        """
        tools = []
        for server in self.get_connected_servers():
            tools.extend(server.tools)
        return tools
    
    async def get_all_resources(self) -> List[Any]:
        """Get all resources from all servers in the cluster.
        
        Returns:
            List of all resources.
        """
        resources = []
        for server in self.get_connected_servers():
            resources.extend(server.resources)
        return resources
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool on the appropriate server.
        
        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments to pass to the tool.
            
        Returns:
            Result of the tool execution.
            
        Raises:
            ValueError: If no server is found with the tool.
        """
        server = await self.get_server_for_tool(tool_name)
        if not server:
            raise ValueError(f"No server found with tool {tool_name} in cluster {self.name}")
        
        return await server.execute_tool(tool_name, arguments)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get status information for the cluster.
        
        Returns:
            Dictionary of status information.
        """
        connected_servers = self.get_connected_servers()
        tools_count = 0
        resources_count = 0
        
        for server in connected_servers:
            tools_count += len(server.tools)
            resources_count += len(server.resources)
        
        return {
            "name": self.name,
            "ip": self.ip,
            "port": self.port,
            "description": self.description,
            "servers": {
                "total": len(self.servers),
                "connected": len(connected_servers),
                "names": [server.name for server in connected_servers]
            },
            "capabilities": {
                "tools_count": tools_count,
                "resources_count": resources_count
            }
        }
    
    def __repr__(self) -> str:
        """Get string representation of the cluster.
        
        Returns:
            String representation.
        """
        return f"Cluster(name={self.name}, ip={self.ip}, port={self.port}, servers={len(self.servers)})"


class ClusterManager(LoggerMixin):
    """Manages multiple Kubernetes clusters."""
    
    def __init__(self) -> None:
        """Initialize a ClusterManager instance."""
        self.clusters: Dict[str, Cluster] = {}
        self.local_servers: Dict[str, MCPServer] = {}
        self._cleanup_lock = asyncio.Lock()
    
    def add_cluster(self, name: str, config: ClusterConfigModel) -> Cluster:
        """Add a cluster to the manager.
        
        Args:
            name: Name of the cluster.
            config: Configuration for the cluster.
            
        Returns:
            The added cluster.
        """
        if name in self.clusters:
            self.logger.warning(f"Cluster {name} already exists, updating configuration")
        
        self.clusters[name] = Cluster(name, config)
        self.logger.info(f"Added cluster {name} to manager")
        return self.clusters[name]
    
    def remove_cluster(self, name: str) -> bool:
        """Remove a cluster from the manager.
        
        Args:
            name: Name of the cluster to remove.
            
        Returns:
            True if the cluster was removed, False otherwise.
        """
        if name not in self.clusters:
            self.logger.warning(f"Cluster {name} not found")
            return False
        
        del self.clusters[name]
        self.logger.info(f"Removed cluster {name} from manager")
        return True
    
    def get_cluster(self, name: str) -> Optional[Cluster]:
        """Get a cluster by name.
        
        Args:
            name: Name of the cluster.
            
        Returns:
            The cluster if found, None otherwise.
        """
        return self.clusters.get(name)
    
    def get_all_clusters(self) -> List[Cluster]:
        """Get all clusters.
        
        Returns:
            List of all clusters.
        """
        return list(self.clusters.values())
    
    def add_local_server(self, name: str, config: ServerConfigModel) -> MCPServer:
        """Add a local server to the manager.
        
        Args:
            name: Name of the server.
            config: Configuration for the server.
            
        Returns:
            The added server.
        """
        if name in self.local_servers:
            self.logger.warning(f"Local server {name} already exists, updating configuration")
        
        self.local_servers[name] = MCPServer(name, config)
        self.logger.info(f"Added local server {name} to manager")
        return self.local_servers[name]
    
    def remove_local_server(self, name: str) -> bool:
        """Remove a local server from the manager.
        
        Args:
            name: Name of the server to remove.
            
        Returns:
            True if the server was removed, False otherwise.
        """
        if name not in self.local_servers:
            self.logger.warning(f"Local server {name} not found")
            return False
        
        del self.local_servers[name]
        self.logger.info(f"Removed local server {name} from manager")
        return True
    
    def get_local_server(self, name: str) -> Optional[MCPServer]:
        """Get a local server by name.
        
        Args:
            name: Name of the server.
            
        Returns:
            The server if found, None otherwise.
        """
        return self.local_servers.get(name)
    
    def get_all_local_servers(self) -> List[MCPServer]:
        """Get all local servers.
        
        Returns:
            List of all local servers.
        """
        return list(self.local_servers.values())
    
    async def connect_cluster(self, name: str) -> bool:
        """Connect to a cluster.
        
        Args:
            name: Name of the cluster to connect to.
            
        Returns:
            True if at least one server was connected, False otherwise.
        """
        cluster = self.get_cluster(name)
        if not cluster:
            self.logger.error(f"Cluster {name} not found")
            return False
        
        results = await cluster.connect()
        return any(results.values())
    
    async def connect_all_clusters(self) -> Dict[str, bool]:
        """Connect to all clusters.
        
        Returns:
            Dictionary of cluster names to connection success.
        """
        results = {}
        for name, cluster in self.clusters.items():
            results[name] = await self.connect_cluster(name)
        return results
    
    async def connect_local_server(self, name: str) -> bool:
        """Connect to a local server.
        
        Args:
            name: Name of the server to connect to.
            
        Returns:
            True if the server was connected, False otherwise.
        """
        server = self.get_local_server(name)
        if not server:
            self.logger.error(f"Local server {name} not found")
            return False
        
        return await server.connect()
    
    async def connect_all_local_servers(self) -> Dict[str, bool]:
        """Connect to all local servers.
        
        Returns:
            Dictionary of server names to connection success.
        """
        results = {}
        for name, server in self.local_servers.items():
            results[name] = await server.connect()
        return results
    
    async def disconnect_cluster(self, name: str) -> None:
        """Disconnect from a cluster.
        
        Args:
            name: Name of the cluster to disconnect from.
        """
        cluster = self.get_cluster(name)
        if not cluster:
            self.logger.error(f"Cluster {name} not found")
            return
        
        await cluster.disconnect()
    
    async def disconnect_all_clusters(self) -> None:
        """Disconnect from all clusters."""
        for name in list(self.clusters.keys()):
            await self.disconnect_cluster(name)
    
    async def disconnect_local_server(self, name: str) -> None:
        """Disconnect from a local server.
        
        Args:
            name: Name of the server to disconnect from.
        """
        server = self.get_local_server(name)
        if not server:
            self.logger.error(f"Local server {name} not found")
            return
        
        await server.disconnect()
    
    async def disconnect_all_local_servers(self) -> None:
        """Disconnect from all local servers."""
        for name in list(self.local_servers.keys()):
            await self.disconnect_local_server(name)
    
    async def disconnect_all(self) -> None:
        """Disconnect from all clusters and local servers."""
        async with self._cleanup_lock:
            await self.disconnect_all_clusters()
            await self.disconnect_all_local_servers()
    
    async def get_appropriate_server(self, cluster_name: Optional[str], tool_name: str) -> Optional[MCPServer]:
        """Get the appropriate server for a tool, based on cluster name.
        
        Args:
            cluster_name: Name of the cluster, or None to check all.
            tool_name: Name of the tool to execute.
            
        Returns:
            The appropriate MCPServer instance to handle the tool.
        """
        # First check cluster-specific servers
        if cluster_name:
            cluster = self.get_cluster(cluster_name)
            if cluster:
                server = await cluster.get_server_for_tool(tool_name)
                if server:
                    return server
        
        # Then check local servers
        for server in self.get_all_local_servers():
            if server.is_connected and await server.has_tool(tool_name):
                return server
        
        # If cluster not specified, check all clusters
        if not cluster_name:
            for cluster in self.get_all_clusters():
                server = await cluster.get_server_for_tool(tool_name)
                if server:
                    return server
        
        return None
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any],
                          cluster_name: Optional[str] = None) -> Any:
        """Execute a tool on the appropriate server.
        
        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments to pass to the tool.
            cluster_name: Name of the cluster to execute the tool on, or None to find automatically.
            
        Returns:
            Result of the tool execution.
            
        Raises:
            ValueError: If no server is found with the tool.
        """
        server = await self.get_appropriate_server(cluster_name, tool_name)
        if not server:
            raise ValueError(f"No server found with tool {tool_name}" +
                           (f" in cluster {cluster_name}" if cluster_name else ""))
        
        return await server.execute_tool(tool_name, arguments)
    
    async def get_cluster_status(self, name: str) -> Dict[str, Any]:
        """Get status information for a cluster.
        
        Args:
            name: Name of the cluster.
            
        Returns:
            Dictionary of status information.
            
        Raises:
            ValueError: If the cluster is not found.
        """
        cluster = self.get_cluster(name)
        if not cluster:
            raise ValueError(f"Cluster {name} not found")
        
        return await cluster.get_status()
    
    async def get_all_clusters_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all clusters.
        
        Returns:
            Dictionary of cluster names to status information.
        """
        results = {}
        for name, cluster in self.clusters.items():
            results[name] = await cluster.get_status()
        return results
    
    async def get_all_local_servers_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all local servers.
        
        Returns:
            Dictionary of server names to status information.
        """
        results = {}
        for name, server in self.local_servers.items():
            results[name] = {
                "name": server.name,
                "connected": server.is_connected,
                "tools_count": len(server.tools),
                "resources_count": len(server.resources)
            }
        return results
    
    async def extract_cluster_name_from_message(self, message: str) -> Optional[str]:
        """Extract a cluster name from a message.
        
        This function attempts to identify if a specific cluster is mentioned in a message.
        
        Args:
            message: The message to analyze.
            
        Returns:
            The name of the mentioned cluster, or None if no cluster is mentioned.
        """
        # Convert message to lowercase for case-insensitive matching
        message_lower = message.lower()
        
        # Create a set of cluster names for faster lookup
        cluster_names: Set[str] = set(self.clusters.keys())
        
        # Simple approach: check if any cluster name is mentioned in the message
        for name in cluster_names:
            # Check different variations of how the cluster might be mentioned
            patterns = [
                f" {name.lower()} cluster",  # "A cluster"
                f"cluster {name.lower()}",   # "cluster A"
                f" {name.lower()} ",         # " A " (with spaces)
                f"'{name.lower()}'",         # "'A'"
                f"\"{name.lower()}\"",       # "\"A\""
            ]
            
            if name.lower() in message_lower:
                for pattern in patterns:
                    if pattern in message_lower:
                        return name
                        
        return None
