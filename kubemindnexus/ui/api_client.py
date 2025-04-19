"""API client for KubeMindNexus."""

import json
import logging
import asyncio
import time
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Callable

import httpx
from httpx import AsyncClient

logger = logging.getLogger(__name__)


class ApiClient:
    """Client for KubeMindNexus API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize API client.
        
        Args:
            base_url: Base URL for API.
        """
        self.base_url = base_url
        self._client = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get the HTTP client, creating it if needed.
        
        Returns:
            The HTTP client.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
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
        stream: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Send a chat message and get the response.
        
        Args:
            message: Chat message.
            cluster_id: Optional cluster ID for context.
            stream: Whether to stream the response. If True, use stream_chat_message instead.
            
        Returns:
            Response message or None if failed.
        """
        if stream:
            logger.warning("stream=True passed to send_chat_message, but this method doesn't support streaming. Use stream_chat_message instead.")
            
        try:
            data = {
                "message": message,
                "cluster_id": cluster_id,
                "stream": False,  # Ensure we're not streaming for this method
            }
            response = await self.client.post("/api/chat", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to send chat message: {str(e)}")
            return None
            
    async def stream_chat_message(
        self,
        message: str, 
        cluster_id: Optional[int] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_thinking: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_tool_result: Optional[Callable[[str, Dict[str, Any], bool], None]] = None,
        on_response: Optional[Callable[[str, bool], None]] = None,
        on_completion: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Send a chat message and stream the response using Server-Sent Events.
        
        This method yields each event as it is received from the server.
        It also calls the appropriate callback for each event type if provided.
        
        Args:
            message: Chat message.
            cluster_id: Optional cluster ID for context.
            on_event: Optional callback for any event.
            on_thinking: Optional callback for thinking events.
            on_tool_call: Optional callback for tool call events.
            on_tool_result: Optional callback for tool result events.
            on_response: Optional callback for response events.
            on_completion: Optional callback for completion events.
            on_error: Optional callback for error events.
            
        Yields:
            Each event from the server as a dictionary.
        """
        data = {
            "message": message,
            "cluster_id": cluster_id,
            "stream": True,
        }
        
        try:
            # Manually handle the streaming request using SSE
            async with self.client.stream("POST", "/api/chat/stream", json=data, timeout=None) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    # Skip empty lines and heartbeat messages
                    if not line or line.startswith(":"):
                        continue
                        
                    # Parse the SSE data
                    if line.startswith("data: "):
                        try:
                            event_data = json.loads(line[6:])  # Remove "data: " prefix
                            
                            # Call the general event callback
                            if on_event:
                                on_event(event_data)
                                
                            # Call specific event type callbacks
                            event_type = event_data.get("type")
                            if event_type == "thinking" and on_thinking:
                                on_thinking(event_data["data"].get("message", "Thinking..."))
                                
                            elif event_type == "tool_call" and on_tool_call:
                                tool_name = event_data["data"].get("tool_name", "unknown_tool")
                                params = event_data["data"].get("parameters", {})
                                on_tool_call(tool_name, params)
                                
                            elif event_type == "tool_result" and on_tool_result:
                                tool_name = event_data["data"].get("tool_name", "unknown_tool")
                                result = event_data["data"].get("result", "")
                                success = event_data["data"].get("success", False)
                                on_tool_result(tool_name, result, success)
                                
                            elif event_type == "response" and on_response:
                                content = event_data["data"].get("content", "")
                                is_partial = event_data["data"].get("is_partial", True)
                                on_response(content, is_partial)
                                
                            elif event_type == "completion" and on_completion:
                                result = event_data["data"].get("result", "")
                                on_completion(result)
                                
                            elif event_type == "error" and on_error:
                                message = event_data["data"].get("message", "Unknown error")
                                on_error(message)
                                
                            # Yield the event to the caller
                            yield event_data
                            
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse SSE event: {e}")
                            if on_error:
                                on_error(f"Failed to parse event: {str(e)}")
                            
        except Exception as e:
            logger.error(f"Error streaming chat message: {str(e)}")
            if on_error:
                on_error(f"Connection error: {str(e)}")
            
            # Yield error event
            error_event = {
                "type": "error",
                "data": {
                    "message": f"Error streaming chat message: {str(e)}"
                },
                "timestamp": time.time()
            }
            yield error_event
    
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
