"""API routes for KubeMindNexus."""
import asyncio
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path, Body
from pydantic import BaseModel, Field

from ..config.config import Configuration
from ..database.client import db_client
from ..database.models import Cluster, MCPServer, ChatHistory
from ..mcp.cluster import ClusterManager
from ..mcp.server import ServerConfigModel
from ..llm.client import LLMClient
from ..llm.factory import llm_factory
from ..llm.react import ReActEngine
from ..utils.logger import LoggerMixin


# Models for request and response payloads
class ClusterCreateRequest(BaseModel):
    """Request model for creating a cluster."""
    
    name: str = Field(..., description="Cluster name")
    ip: str = Field(..., description="Cluster IP address")
    port: int = Field(..., description="Cluster port")
    description: Optional[str] = Field(None, description="Cluster description")


class ClusterResponse(BaseModel):
    """Response model for a cluster."""
    
    id: Optional[int] = Field(None, description="Cluster ID")
    name: str = Field(..., description="Cluster name")
    ip: str = Field(..., description="Cluster IP address")
    port: int = Field(..., description="Cluster port")
    description: Optional[str] = Field(None, description="Cluster description")
    status: Optional[str] = Field(None, description="Cluster status")


class ServerCreateRequest(BaseModel):
    """Request model for creating an MCP server."""
    
    name: str = Field(..., description="Server name")
    type: str = Field(..., description="Server type ('stdio' or 'sse')")
    command: Optional[str] = Field(None, description="Command for stdio servers")
    args: Optional[List[str]] = Field(default_factory=list, description="Command arguments")
    env: Optional[Dict[str, str]] = Field(default_factory=dict, description="Environment variables")
    url: Optional[str] = Field(None, description="URL for SSE servers")
    is_local: bool = Field(False, description="Whether this is a local server")


class ServerResponse(BaseModel):
    """Response model for an MCP server."""
    
    id: Optional[int] = Field(None, description="Server ID")
    name: str = Field(..., description="Server name")
    type: str = Field(..., description="Server type ('stdio' or 'sse')")
    command: Optional[str] = Field(None, description="Command for stdio servers")
    args: Optional[List[str]] = Field(default_factory=list, description="Command arguments")
    url: Optional[str] = Field(None, description="URL for SSE servers")
    is_local: bool = Field(False, description="Whether this is a local server")
    cluster_name: Optional[str] = Field(None, description="Cluster name if not local")
    connected: bool = Field(False, description="Whether the server is connected")


class ChatRequest(BaseModel):
    """Request model for a chat message."""
    
    message: str = Field(..., description="User message")
    cluster_name: Optional[str] = Field(None, description="Specific cluster to target")
    llm_provider: Optional[str] = Field(None, description="LLM provider to use")


class ChatResponse(BaseModel):
    """Response model for a chat message."""
    
    message: str = Field(..., description="Assistant response")
    cluster_name: Optional[str] = Field(None, description="Detected cluster name")
    llm_provider: str = Field(..., description="LLM provider used")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Tools called during processing")


class LLMProviderResponse(BaseModel):
    """Response model for an LLM provider."""
    
    name: str = Field(..., description="Provider name")
    available: bool = Field(..., description="Whether the provider is available")
    is_default: bool = Field(False, description="Whether this is the default provider")


class HealthCheckResponse(BaseModel):
    """Response model for a health check."""
    
    status: str = Field(..., description="Overall status")
    api: Dict[str, Any] = Field(..., description="API status")
    clusters: Dict[str, Any] = Field(..., description="Cluster statuses")


# API routers
clusters_router = APIRouter()
chat_router = APIRouter()
mcp_router = APIRouter()
llm_router = APIRouter()
health_router = APIRouter()


# Dependencies
def get_config() -> Configuration:
    """Get the configuration singleton.
    
    Returns:
        The configuration singleton.
    """
    return Configuration()


def get_cluster_manager() -> ClusterManager:
    """Get the cluster manager singleton.
    
    Returns:
        The cluster manager singleton.
    """
    return ClusterManager()


def get_llm_client(provider: Optional[str] = None) -> LLMClient:
    """Get an LLM client.
    
    Args:
        provider: The provider to use. If None, uses the default provider.
        
    Returns:
        An LLM client instance.
    """
    return llm_factory.create_client(provider)


def get_react_engine(llm_client: LLMClient = Depends(get_llm_client),
                   cluster_manager: ClusterManager = Depends(get_cluster_manager)) -> ReActEngine:
    """Get a ReActEngine instance.
    
    Args:
        llm_client: The LLM client to use.
        cluster_manager: The cluster manager to use.
        
    Returns:
        A ReActEngine instance.
    """
    return ReActEngine(llm_client, cluster_manager)


# Clusters API
@clusters_router.post("/", response_model=ClusterResponse, status_code=201)
async def create_cluster(
    cluster: ClusterCreateRequest,
    config: Configuration = Depends(get_config)
) -> ClusterResponse:
    """Create a new cluster.
    
    Args:
        cluster: The cluster to create.
        config: The configuration singleton.
        
    Returns:
        The created cluster.
        
    Raises:
        HTTPException: If the cluster could not be created.
    """
    db_cluster = db_client.add_cluster(
        name=cluster.name,
        ip=cluster.ip,
        port=cluster.port,
        description=cluster.description
    )
    
    if not db_cluster:
        raise HTTPException(status_code=400, detail=f"Could not create cluster '{cluster.name}'")
    
    # Add to configuration
    config.add_cluster(
        cluster_name=cluster.name,
        ip=cluster.ip,
        port=cluster.port,
        description=cluster.description
    )
    
    return ClusterResponse(
        id=db_cluster.id,
        name=db_cluster.name,
        ip=db_cluster.ip,
        port=db_cluster.port,
        description=db_cluster.description,
        status="created"
    )


@clusters_router.get("/", response_model=List[ClusterResponse])
async def list_clusters() -> List[ClusterResponse]:
    """List all clusters.
    
    Returns:
        A list of all clusters.
    """
    db_clusters = db_client.get_all(Cluster)
    
    return [
        ClusterResponse(
            id=c.id,
            name=c.name,
            ip=c.ip,
            port=c.port,
            description=c.description,
            status="unknown"  # We would need to check actual status
        )
        for c in db_clusters
    ]


@clusters_router.get("/{cluster_name}", response_model=ClusterResponse)
async def get_cluster(
    cluster_name: str = Path(..., description="Cluster name"),
    cluster_manager: ClusterManager = Depends(get_cluster_manager)
) -> ClusterResponse:
    """Get a cluster by name.
    
    Args:
        cluster_name: The name of the cluster.
        cluster_manager: The cluster manager singleton.
        
    Returns:
        The cluster.
        
    Raises:
        HTTPException: If the cluster is not found.
    """
    db_cluster = db_client.get_cluster_by_name(cluster_name)
    
    if not db_cluster:
        raise HTTPException(status_code=404, detail=f"Cluster '{cluster_name}' not found")
    
    # Get status if cluster is in manager
    status = "not_connected"
    cluster = cluster_manager.get_cluster(cluster_name)
    if cluster:
        connected_servers = cluster.get_connected_servers()
        if connected_servers:
            status = "connected"
    
    return ClusterResponse(
        id=db_cluster.id,
        name=db_cluster.name,
        ip=db_cluster.ip,
        port=db_cluster.port,
        description=db_cluster.description,
        status=status
    )


@clusters_router.delete("/{cluster_name}", status_code=204)
async def delete_cluster(
    cluster_name: str = Path(..., description="Cluster name"),
    config: Configuration = Depends(get_config),
    cluster_manager: ClusterManager = Depends(get_cluster_manager)
) -> None:
    """Delete a cluster.
    
    Args:
        cluster_name: The name of the cluster.
        config: The configuration singleton.
        cluster_manager: The cluster manager singleton.
        
    Raises:
        HTTPException: If the cluster is not found.
    """
    db_cluster = db_client.get_cluster_by_name(cluster_name)
    
    if not db_cluster:
        raise HTTPException(status_code=404, detail=f"Cluster '{cluster_name}' not found")
    
    # Disconnect if connected
    cluster = cluster_manager.get_cluster(cluster_name)
    if cluster:
        await cluster_manager.disconnect_cluster(cluster_name)
        cluster_manager.remove_cluster(cluster_name)
    
    # Remove from database
    db_client.delete_cluster(db_cluster.id)
    
    # Remove from configuration
    config.remove_cluster(cluster_name)


@clusters_router.post("/{cluster_name}/connect", response_model=Dict[str, Any])
async def connect_cluster(
    cluster_name: str = Path(..., description="Cluster name"),
    cluster_manager: ClusterManager = Depends(get_cluster_manager),
    config: Configuration = Depends(get_config)
) -> Dict[str, Any]:
    """Connect to a cluster.
    
    Args:
        cluster_name: The name of the cluster.
        cluster_manager: The cluster manager singleton.
        config: The configuration singleton.
        
    Returns:
        The connection result.
        
    Raises:
        HTTPException: If the cluster is not found or cannot be connected.
    """
    db_cluster = db_client.get_cluster_by_name(cluster_name)
    
    if not db_cluster:
        raise HTTPException(status_code=404, detail=f"Cluster '{cluster_name}' not found")
    
    # Check if cluster already in manager
    cluster = cluster_manager.get_cluster(cluster_name)
    if not cluster:
        # Load from configuration
        cluster_config = config.get_cluster(cluster_name)
        if not cluster_config:
            raise HTTPException(
                status_code=400,
                detail=f"Cluster '{cluster_name}' config not found"
            )
        
        # Add to manager
        cluster = cluster_manager.add_cluster(cluster_name, cluster_config)
    
    # Connect
    result = await cluster_manager.connect_cluster(cluster_name)
    
    if not result:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to cluster '{cluster_name}'"
        )
    
    return {"status": "connected", "cluster": cluster_name}


@clusters_router.post("/{cluster_name}/disconnect", response_model=Dict[str, Any])
async def disconnect_cluster(
    cluster_name: str = Path(..., description="Cluster name"),
    cluster_manager: ClusterManager = Depends(get_cluster_manager)
) -> Dict[str, Any]:
    """Disconnect from a cluster.
    
    Args:
        cluster_name: The name of the cluster.
        cluster_manager: The cluster manager singleton.
        
    Returns:
        The disconnection result.
        
    Raises:
        HTTPException: If the cluster is not found.
    """
    cluster = cluster_manager.get_cluster(cluster_name)
    
    if not cluster:
        raise HTTPException(status_code=404, detail=f"Cluster '{cluster_name}' not found or not connected")
    
    await cluster_manager.disconnect_cluster(cluster_name)
    
    return {"status": "disconnected", "cluster": cluster_name}


@clusters_router.get("/{cluster_name}/status", response_model=Dict[str, Any])
async def get_cluster_status(
    cluster_name: str = Path(..., description="Cluster name"),
    cluster_manager: ClusterManager = Depends(get_cluster_manager)
) -> Dict[str, Any]:
    """Get a cluster's status.
    
    Args:
        cluster_name: The name of the cluster.
        cluster_manager: The cluster manager singleton.
        
    Returns:
        The cluster status.
        
    Raises:
        HTTPException: If the cluster is not found.
    """
    cluster = cluster_manager.get_cluster(cluster_name)
    
    if not cluster:
        raise HTTPException(status_code=404, detail=f"Cluster '{cluster_name}' not found or not connected")
    
    try:
        status = await cluster.get_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cluster status: {str(e)}")


# MCP API
@mcp_router.post("/servers", response_model=ServerResponse, status_code=201)
async def create_server(
    server: ServerCreateRequest,
    cluster_name: Optional[str] = Query(None, description="Cluster name for non-local servers"),
    config: Configuration = Depends(get_config)
) -> ServerResponse:
    """Create a new MCP server.
    
    Args:
        server: The server to create.
        cluster_name: The name of the cluster for non-local servers.
        config: The configuration singleton.
        
    Returns:
        The created server.
        
    Raises:
        HTTPException: If the server could not be created.
    """
    # Create server model
    server_config = ServerConfigModel(
        name=server.name,
        type=server.type,
        command=server.command,
        args=server.args or [],
        env=server.env or {},
        url=server.url
    )
    
    # Add to database
    db_server = db_client.add_server(
        name=server.name,
        server_type=server.type,
        cluster_id=None if server.is_local else db_client.get_cluster_by_name(cluster_name).id if cluster_name else None,
        command=server.command,
        args=server.args,
        env=server.env,
        url=server.url,
        is_local=server.is_local
    )
    
    if not db_server:
        raise HTTPException(status_code=400, detail=f"Could not create server '{server.name}'")
    
    # Add to configuration
    if server.is_local:
        config.add_local_server(
            server_name=server.name,
            server_type=server.type,
            command=server.command,
            args=server.args,
            env=server.env,
            url=server.url
        )
    elif cluster_name:
        config.add_server_to_cluster(
            cluster_name=cluster_name,
            server_name=server.name,
            server_type=server.type,
            command=server.command,
            args=server.args,
            env=server.env,
            url=server.url
        )
    else:
        raise HTTPException(status_code=400, detail="Cluster name required for non-local servers")
    
    return ServerResponse(
        id=db_server.id,
        name=db_server.name,
        type=db_server.type,
        command=db_server.command,
        args=db_server.args_list,
        url=db_server.url,
        is_local=db_server.is_local,
        cluster_name=cluster_name if not server.is_local else None,
        connected=False
    )


@mcp_router.get("/servers", response_model=List[ServerResponse])
async def list_servers(
    cluster_name: Optional[str] = Query(None, description="Filter by cluster name")
) -> List[ServerResponse]:
    """List MCP servers.
    
    Args:
        cluster_name: Optional cluster name to filter by.
        
    Returns:
        A list of MCP servers.
    """
    if cluster_name:
        db_cluster = db_client.get_cluster_by_name(cluster_name)
        if not db_cluster:
            raise HTTPException(status_code=404, detail=f"Cluster '{cluster_name}' not found")
        
        db_servers = db_client.get_servers_for_cluster(db_cluster.id)
    else:
        db_servers = db_client.get_all(MCPServer)
    
    return [
        ServerResponse(
            id=s.id,
            name=s.name,
            type=s.type,
            command=s.command,
            args=s.args_list,
            url=s.url,
            is_local=s.is_local,
            cluster_name=s.cluster.name if s.cluster else None,
            connected=False  # We would need to check actual connection status
        )
        for s in db_servers
    ]


@mcp_router.delete("/servers/{server_name}", status_code=204)
async def delete_server(
    server_name: str = Path(..., description="Server name"),
    cluster_name: Optional[str] = Query(None, description="Cluster name for non-local servers"),
    config: Configuration = Depends(get_config),
    cluster_manager: ClusterManager = Depends(get_cluster_manager)
) -> None:
    """Delete an MCP server.
    
    Args:
        server_name: The name of the server.
        cluster_name: The name of the cluster for non-local servers.
        config: The configuration singleton.
        cluster_manager: The cluster manager singleton.
        
    Raises:
        HTTPException: If the server is not found.
    """
    # Find server in database
    servers = db_client.query(MCPServer, name=server_name)
    
    if not servers:
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")
    
    server = None
    if cluster_name:
        for s in servers:
            if s.cluster and s.cluster.name == cluster_name:
                server = s
                break
        
        if not server:
            raise HTTPException(
                status_code=404,
                detail=f"Server '{server_name}' not found in cluster '{cluster_name}'"
            )
    else:
        for s in servers:
            if s.is_local:
                server = s
                break
        
        if not server:
            raise HTTPException(
                status_code=400,
                detail=f"Server '{server_name}' is not a local server, please specify cluster_name"
            )
    
    # Disconnect if connected
    if server.is_local:
        local_server = cluster_manager.get_local_server(server_name)
        if local_server:
            await cluster_manager.disconnect_local_server(server_name)
            cluster_manager.remove_local_server(server_name)
        config.remove_local_server(server_name)
    else:
        cluster = cluster_manager.get_cluster(cluster_name)
        if cluster and server_name in cluster.servers:
            await cluster.disconnect()
        config.remove_server_from_cluster(cluster_name, server_name)
    
    # Remove from database
    db_client.delete(server)


# Chat API
@chat_router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    react_engine: ReActEngine = Depends(get_react_engine),
    llm_provider: Optional[str] = None
) -> ChatResponse:
    """Chat with the LLM.
    
    Args:
        request: The chat request.
        react_engine: The ReActEngine instance.
        llm_provider: The LLM provider to use.
        
    Returns:
        The chat response.
        
    Raises:
        HTTPException: If there was an error processing the chat request.
    """
    try:
        # Override LLM provider if specified in request
        if request.llm_provider and request.llm_provider != llm_provider:
            llm_client = llm_factory.create_client(request.llm_provider)
            react_engine = ReActEngine(llm_client, react_engine.cluster_manager)
        
        # Get recent conversation history
        history = []
        db_history = db_client.get_chat_history(limit=10)
        
        for entry in reversed(db_history):
            history.append({"role": "user", "content": entry.user_message})
            history.append({"role": "assistant", "content": entry.assistant_response})
        
        # Process message with ReAct pattern
        response = await react_engine.process(request.message, history)
        
        # Save to history
        db_client.add_chat_history(
            user_message=request.message,
            assistant_response=response,
            llm_provider=react_engine.llm_client.get_name(),
            cluster_id=None,  # We could try to determine this
            metadata={}
        )
        
        return ChatResponse(
            message=response,
            cluster_name=request.cluster_name,  # We could try to extract this
            llm_provider=react_engine.llm_client.get_name(),
            tool_calls=[]  # We would need to track tool calls during processing
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@chat_router.get("/history", response_model=List[Dict[str, Any]])
async def get_chat_history(
    limit: int = Query(10, description="Maximum number of entries to return")
) -> List[Dict[str, Any]]:
    """Get chat history.
    
    Args:
        limit: Maximum number of entries to return.
        
    Returns:
        A list of chat history entries.
    """
    db_history = db_client.get_chat_history(limit=limit)
    
    return [
        {
            "id": entry.id,
            "timestamp": entry.timestamp.isoformat(),
            "user_message": entry.user_message,
            "assistant_response": entry.assistant_response,
            "llm_provider": entry.llm_provider,
            "cluster_name": entry.cluster.name if entry.cluster else None
        }
        for entry in db_history
    ]


# LLM API
@llm_router.get("/providers", response_model=List[LLMProviderResponse])
async def list_llm_providers(
    config: Configuration = Depends(get_config)
) -> List[LLMProviderResponse]:
    """List available LLM providers.
    
    Args:
        config: The configuration singleton.
        
    Returns:
        A list of available LLM providers.
    """
    providers = llm_factory.get_available_providers()
    default_provider = config.get_default_llm_provider()
    
    return [
        LLMProviderResponse(
            name=provider,
            available=llm_factory.is_provider_available(provider),
            is_default=provider == default_provider
        )
        for provider in providers
    ]


@llm_router.post("/providers/{provider_name}/default", response_model=Dict[str, Any])
async def set_default_llm_provider(
    provider_name: str = Path(..., description="Provider name"),
    config: Configuration = Depends(get_config)
) -> Dict[str, Any]:
    """Set the default LLM provider.
    
    Args:
        provider_name: The name of the provider.
        config: The configuration singleton.
        
    Returns:
        The result of setting the default provider.
        
    Raises:
        HTTPException: If the provider is not supported.
    """
    if not llm_factory.is_provider_available(provider_name):
        raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not supported")
    
    config.set_default_llm_provider(provider_name)
    
    return {"status": "success", "default_provider": provider_name}


# Health API
@health_router.get("", response_model=HealthCheckResponse)
async def health_check(
    cluster_manager: ClusterManager = Depends(get_cluster_manager)
) -> HealthCheckResponse:
    """Get the health status of the system.
    
    Args:
        cluster_manager: The cluster manager singleton.
        
    Returns:
        The health status.
    """
    # Check database
    try:
        db_client.get_all(Cluster)
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"
    
    # Check clusters
    cluster_statuses = {}
    for cluster in cluster_manager.get_all_clusters():
        connected_servers = cluster.get_connected_servers()
        cluster_statuses[cluster.name] = {
            "status": "connected" if connected_servers else "disconnected",
            "servers": {
                "total": len(cluster.servers),
                "connected": len(connected_servers)
            }
        }
    
    # Overall status
    overall_status = "healthy"
    if db_status != "healthy":
        overall_status = "degraded"
    
    return HealthCheckResponse(
        status=overall_status,
        api={
            "status": "healthy",
            "database": db_status
        },
        clusters=cluster_statuses
    )
