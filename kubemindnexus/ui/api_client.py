"""API client for KubeMindNexus."""

import json
import logging
from typing import Any, Dict, List, Optional, Union

import httpx

logger = logging.getLogger(__name__)


class ApiClient:
    """Client for KubeMindNexus API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize API client.
        
        Args:
            base_url: Base URL for API.
        """
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    # Cluster endpoints
    
    async def get_clusters(self) -> List[Dict[str, Any]]:
        """Get all clusters.
        
        Returns:
            List of clusters.
        """
        try:
            response = await self.client.get("/api/clusters")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get clusters: {str(e)}")
            return []
    
    async def get_cluster(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        """Get a cluster by ID.
        
        Args:
            cluster_id: Cluster ID.
            
        Returns:
            Cluster data or None if not found.
        """
        try:
            response = await self.client.get(f"/api/clusters/{cluster_id}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logger.error(f"HTTP error getting cluster {cluster_id}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Failed to get cluster {cluster_id}: {str(e)}")
            return None
    
    async def get_cluster_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a cluster by name.
        
        Args:
            name: Cluster name.
            
        Returns:
            Cluster data or None if not found.
        """
        try:
            clusters = await self.get_clusters()
            for cluster in clusters:
                if cluster["name"] == name:
                    return cluster
            return None
        except Exception as e:
            logger.error(f"Failed to get cluster by name {name}: {str(e)}")
            return None
    
    async def add_cluster(
        self,
        name: str,
        ip: str,
        port: int,
        description: Optional[str] = None,
    ) -> Optional[int]:
        """Add a new cluster.
        
        Args:
            name: Cluster name.
            ip: Cluster IP address.
            port: Cluster port.
            description: Optional cluster description.
            
        Returns:
            Cluster ID or None if failed.
        """
        try:
            data = {
                "name": name,
                "ip": ip,
                "port": port,
                "description": description,
            }
            response = await self.client.post("/api/clusters", json=data)
            response.raise_for_status()
            cluster = response.json()
            return cluster["id"]
        except Exception as e:
            logger.error(f"Failed to add cluster: {str(e)}")
            return None
    
    async def update_cluster(
        self,
        cluster_id: int,
        name: Optional[str] = None,
        ip: Optional[str] = None,
        port: Optional[int] = None,
        description: Optional[str] = None,
    ) -> bool:
        """Update a cluster.
        
        Args:
            cluster_id: Cluster ID.
            name: Optional new name.
            ip: Optional new IP address.
            port: Optional new port.
            description: Optional new description.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            data = {}
            if name is not None:
                data["name"] = name
            if ip is not None:
                data["ip"] = ip
            if port is not None:
                data["port"] = port
            if description is not None:
                data["description"] = description
                
            response = await self.client.put(f"/api/clusters/{cluster_id}", json=data)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to update cluster {cluster_id}: {str(e)}")
            return False
    
    async def delete_cluster(self, cluster_id: int) -> bool:
        """Delete a cluster.
        
        Args:
            cluster_id: Cluster ID.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            response = await self.client.delete(f"/api/clusters/{cluster_id}")
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to delete cluster {cluster_id}: {str(e)}")
            return False
    
    # MCP server endpoints
    
    async def get_mcp_servers(self, cluster_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all MCP servers, optionally filtered by cluster ID.
        
        Args:
            cluster_id: Optional cluster ID to filter by.
            
        Returns:
            List of MCP servers.
        """
        try:
            params = {}
            if cluster_id is not None:
                params["cluster_id"] = cluster_id
                
            response = await self.client.get("/api/mcp-servers", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get MCP servers: {str(e)}")
            return []
    
    async def get_mcp_server(self, server_id: int) -> Optional[Dict[str, Any]]:
        """Get an MCP server by ID.
        
        Args:
            server_id: Server ID.
            
        Returns:
            Server data or None if not found.
        """
        try:
            response = await self.client.get(f"/api/mcp-servers/{server_id}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logger.error(f"HTTP error getting MCP server {server_id}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Failed to get MCP server {server_id}: {str(e)}")
            return None
    
    async def get_mcp_servers_by_cluster(self, cluster_id: int) -> List[Dict[str, Any]]:
        """Get MCP servers for a cluster.
        
        Args:
            cluster_id: Cluster ID.
            
        Returns:
            List of MCP servers.
        """
        return await self.get_mcp_servers(cluster_id=cluster_id)
    
    async def add_mcp_server(
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
    ) -> Optional[int]:
        """Add a new MCP server.
        
        Args:
            name: Server name.
            server_type: Server type.
            command: Optional command to execute.
            args: Optional command arguments.
            url: Optional server URL.
            cluster_id: Optional cluster ID.
            is_local: Whether this is a local server.
            is_default: Whether this is a default server.
            env: Optional environment variables.
            
        Returns:
            Server ID or None if failed.
        """
        try:
            data = {
                "name": name,
                "type": server_type,
                "command": command,
                "args": args,
                "url": url,
                "cluster_id": cluster_id,
                "is_local": is_local,
                "is_default": is_default,
                "env": env or {},
            }
            response = await self.client.post("/api/mcp-servers", json=data)
            response.raise_for_status()
            server = response.json()
            return server["id"]
        except Exception as e:
            logger.error(f"Failed to add MCP server: {str(e)}")
            return None
    
    async def update_mcp_server(
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
            name: Optional new name.
            server_type: Optional new server type.
            command: Optional new command.
            args: Optional new arguments.
            url: Optional new URL.
            cluster_id: Optional new cluster ID.
            is_local: Optional new local flag.
            is_default: Optional new default flag.
            env: Optional new environment variables.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            data = {}
            if name is not None:
                data["name"] = name
            if server_type is not None:
                data["type"] = server_type
            if command is not None:
                data["command"] = command
            if args is not None:
                data["args"] = args
            if url is not None:
                data["url"] = url
            if cluster_id is not None:
                data["cluster_id"] = cluster_id
            if is_local is not None:
                data["is_local"] = is_local
            if is_default is not None:
                data["is_default"] = is_default
            if env is not None:
                data["env"] = env
                
            response = await self.client.put(f"/api/mcp-servers/{server_id}", json=data)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to update MCP server {server_id}: {str(e)}")
            return False
    
    async def delete_mcp_server(self, server_id: int) -> bool:
        """Delete an MCP server.
        
        Args:
            server_id: Server ID.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            response = await self.client.delete(f"/api/mcp-servers/{server_id}")
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to delete MCP server {server_id}: {str(e)}")
            return False
    
    async def connect_mcp_server(self, server_id: int) -> bool:
        """Connect to an MCP server.
        
        Args:
            server_id: Server ID.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            response = await self.client.post(f"/api/mcp-servers/{server_id}/connect")
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {server_id}: {str(e)}")
            return False
    
    async def disconnect_mcp_server(self, server_id: int) -> bool:
        """Disconnect from an MCP server.
        
        Args:
            server_id: Server ID.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            response = await self.client.post(f"/api/mcp-servers/{server_id}/disconnect")
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to disconnect from MCP server {server_id}: {str(e)}")
            return False
    
    # Chat endpoints
    
    async def send_chat_message(
        self,
        message: str,
        cluster_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Send a chat message and get the response.
        
        Args:
            message: Chat message.
            cluster_id: Optional cluster ID for context.
            
        Returns:
            Response message or None if failed.
        """
        try:
            data = {
                "message": message,
                "cluster_id": cluster_id,
            }
            response = await self.client.post("/api/chat", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to send chat message: {str(e)}")
            return None
    
    async def get_chat_history(
        self,
        limit: int = 20,
        cluster_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get chat history.
        
        Args:
            limit: Maximum number of messages to retrieve.
            cluster_id: Optional cluster ID to filter by.
            
        Returns:
            List of chat messages.
        """
        try:
            params = {"limit": limit}
            if cluster_id is not None:
                params["cluster_id"] = cluster_id
                
            response = await self.client.get("/api/chat-history", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get chat history: {str(e)}")
            return []
    
    async def clear_chat_history(self, cluster_id: Optional[int] = None) -> bool:
        """Clear chat history.
        
        Args:
            cluster_id: Optional cluster ID to filter by.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            params = {}
            if cluster_id is not None:
                params["cluster_id"] = cluster_id
                
            response = await self.client.delete("/api/chat-history", params=params)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to clear chat history: {str(e)}")
            return False
    
    # MCP server status endpoints
    
    async def get_mcp_servers_status(self) -> List[Dict[str, Any]]:
        """Get status of all MCP servers.
        
        Returns:
            List of server status.
        """
        try:
            response = await self.client.get("/api/mcp-servers/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get MCP server status: {str(e)}")
            return []
    
    async def get_mcp_server_status(self, server_id: int) -> Optional[Dict[str, Any]]:
        """Get status of a specific MCP server.
        
        Args:
            server_id: Server ID.
            
        Returns:
            Server status or None if not found.
        """
        try:
            response = await self.client.get(f"/api/mcp-servers/{server_id}/status")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logger.error(f"HTTP error getting MCP server status {server_id}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Failed to get MCP server status {server_id}: {str(e)}")
            return None
    
    # MCP server tools and resources endpoints
    
    async def get_mcp_server_tools(self, server_id: int) -> List[Dict[str, Any]]:
        """Get available tools for a specific MCP server.
        
        Args:
            server_id: Server ID.
            
        Returns:
            List of tools.
        """
        try:
            response = await self.client.get(f"/api/mcp-servers/{server_id}/tools")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get MCP server tools {server_id}: {str(e)}")
            return []
    
    async def get_mcp_server_resources(self, server_id: int) -> List[Dict[str, Any]]:
        """Get available resources for a specific MCP server.
        
        Args:
            server_id: Server ID.
            
        Returns:
            List of resources.
        """
        try:
            response = await self.client.get(f"/api/mcp-servers/{server_id}/resources")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get MCP server resources {server_id}: {str(e)}")
            return []
    
    # Cluster metrics endpoints
    
    async def get_cluster_performance_metrics(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a specific cluster.
        
        Args:
            cluster_id: Cluster ID.
            
        Returns:
            Metrics data or None if not found.
        """
        try:
            response = await self.client.get(f"/api/clusters/{cluster_id}/metrics/performance")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logger.error(f"HTTP error getting cluster performance metrics {cluster_id}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Failed to get cluster performance metrics {cluster_id}: {str(e)}")
            return None
    
    async def get_cluster_health_metrics(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        """Get health metrics for a specific cluster.
        
        Args:
            cluster_id: Cluster ID.
            
        Returns:
            Metrics data or None if not found.
        """
        try:
            response = await self.client.get(f"/api/clusters/{cluster_id}/metrics/health")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logger.error(f"HTTP error getting cluster health metrics {cluster_id}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Failed to get cluster health metrics {cluster_id}: {str(e)}")
            return None
    
    async def get_cluster_storage_metrics(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        """Get storage metrics for a specific cluster.
        
        Args:
            cluster_id: Cluster ID.
            
        Returns:
            Metrics data or None if not found.
        """
        try:
            response = await self.client.get(f"/api/clusters/{cluster_id}/metrics/storage")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logger.error(f"HTTP error getting cluster storage metrics {cluster_id}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Failed to get cluster storage metrics {cluster_id}: {str(e)}")
            return None
    
    # Kubernetes resources endpoints
    
    async def get_cluster_nodes(self, cluster_id: int) -> List[Dict[str, Any]]:
        """Get nodes for a specific cluster.
        
        Args:
            cluster_id: Cluster ID.
            
        Returns:
            List of nodes.
        """
        try:
            response = await self.client.get(f"/api/clusters/{cluster_id}/nodes")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get cluster nodes {cluster_id}: {str(e)}")
            return []
    
    async def get_cluster_pods(
        self, cluster_id: int, namespace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get pods for a specific cluster, optionally filtered by namespace.
        
        Args:
            cluster_id: Cluster ID.
            namespace: Optional namespace to filter by.
            
        Returns:
            List of pods.
        """
        try:
            params = {}
            if namespace is not None:
                params["namespace"] = namespace
                
            response = await self.client.get(
                f"/api/clusters/{cluster_id}/pods",
                params=params,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get cluster pods {cluster_id}: {str(e)}")
            return []
    
    async def get_cluster_services(
        self, cluster_id: int, namespace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get services for a specific cluster, optionally filtered by namespace.
        
        Args:
            cluster_id: Cluster ID.
            namespace: Optional namespace to filter by.
            
        Returns:
            List of services.
        """
        try:
            params = {}
            if namespace is not None:
                params["namespace"] = namespace
                
            response = await self.client.get(
                f"/api/clusters/{cluster_id}/services",
                params=params,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get cluster services {cluster_id}: {str(e)}")
            return []
    
    async def get_cluster_persistent_volumes(self, cluster_id: int) -> List[Dict[str, Any]]:
        """Get persistent volumes for a specific cluster.
        
        Args:
            cluster_id: Cluster ID.
            
        Returns:
            List of persistent volumes.
        """
        try:
            response = await self.client.get(f"/api/clusters/{cluster_id}/persistent-volumes")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get cluster persistent volumes {cluster_id}: {str(e)}")
            return []
    
    # LLM configuration endpoints
    
    async def get_llm_configs(self) -> List[Dict[str, Any]]:
        """Get all LLM configurations.
        
        Returns:
            List of LLM configurations.
        """
        try:
            response = await self.client.get("/api/llm-config")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get LLM configurations: {str(e)}")
            return []
    
    async def get_llm_config(self, config_id: int) -> Optional[Dict[str, Any]]:
        """Get an LLM configuration by ID.
        
        Args:
            config_id: Configuration ID.
            
        Returns:
            Configuration data or None if not found.
        """
        try:
            response = await self.client.get(f"/api/llm-config/{config_id}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logger.error(f"HTTP error getting LLM configuration {config_id}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Failed to get LLM configuration {config_id}: {str(e)}")
            return None
    
    async def add_llm_config(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        is_default: bool = False,
    ) -> Optional[int]:
        """Add a new LLM configuration.
        
        Args:
            provider: LLM provider.
            model: Model name.
            api_key: Optional API key.
            base_url: Optional base URL.
            parameters: Optional additional parameters.
            is_default: Whether this is the default configuration.
            
        Returns:
            Configuration ID or None if failed.
        """
        try:
            data = {
                "provider": provider,
                "model": model,
                "api_key": api_key,
                "base_url": base_url,
                "parameters": parameters,
                "is_default": is_default,
            }
            response = await self.client.post("/api/llm-config", json=data)
            response.raise_for_status()
            config = response.json()
            return config["id"]
        except Exception as e:
            logger.error(f"Failed to add LLM configuration: {str(e)}")
            return None
    
    async def update_llm_config(
        self,
        config_id: int,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        is_default: Optional[bool] = None,
    ) -> bool:
        """Update an LLM configuration.
        
        Args:
            config_id: Configuration ID.
            provider: Optional new provider.
            model: Optional new model.
            api_key: Optional new API key.
            base_url: Optional new base URL.
            parameters: Optional new parameters.
            is_default: Optional new default flag.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            data = {}
            if provider is not None:
                data["provider"] = provider
            if model is not None:
                data["model"] = model
            if api_key is not None:
                data["api_key"] = api_key
            if base_url is not None:
                data["base_url"] = base_url
            if parameters is not None:
                data["parameters"] = parameters
            if is_default is not None:
                data["is_default"] = is_default
                
            response = await self.client.put(f"/api/llm-config/{config_id}", json=data)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to update LLM configuration {config_id}: {str(e)}")
            return False
    
    async def delete_llm_config(self, config_id: int) -> bool:
        """Delete an LLM configuration.
        
        Args:
            config_id: Configuration ID.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            response = await self.client.delete(f"/api/llm-config/{config_id}")
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to delete LLM configuration {config_id}: {str(e)}")
            return False
    
    # Server status endpoint
    
    async def check_server_status(self) -> bool:
        """Check if the API server is running.
        
        Returns:
            True if the server is running, False otherwise.
        """
        try:
            response = await self.client.get("/api")
            response.raise_for_status()
            return True
        except Exception:
            return False
