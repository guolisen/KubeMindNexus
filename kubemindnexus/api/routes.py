"""API routes for KubeMindNexus."""

import logging
import json
import time
import asyncio
from typing import Any, Dict, List, Optional, AsyncGenerator, Union 

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, status, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..config import Configuration
from ..constants import LLMProvider, ServerType
from ..database import DatabaseManager
from ..llm.base import LLMFactory
from ..llm.react import ReactLoop
from ..mcp.hub import MCPHub
from ..mcp.manager import MCPManager

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="KubeMindNexus API",
    description="API for KubeMindNexus Kubernetes clusters management",
    version="0.1.0",
)

# Create router
router = APIRouter()

# Pydantic models for responses
class ServerStatus(BaseModel):
    """Model for MCP server status."""
    
    id: int = Field(..., description="Server ID")
    name: str = Field(..., description="Server name")
    is_connected: bool = Field(..., description="Whether the server is connected")
    


# Dependency to get database manager
def get_db_manager():
    """Get database manager instance."""
    # This should be properly initialized in the main module
    db_manager = app.state.db_manager
    return db_manager


# Dependency to get configuration
def get_config():
    """Get configuration instance."""
    # This should be properly initialized in the main module
    config = app.state.config
    return config


# Dependency to get MCP hub
def get_mcp_hub():
    """Get MCP hub instance."""
    # This should be properly initialized in the main module
    mcp_hub = app.state.mcp_hub
    return mcp_hub


# Dependency to get ReactLoop
def get_react_loop():
    """Get ReactLoop instance."""
    # This should be properly initialized in the main module
    return app.state.react_loop


# Pydantic models for request/response
class ClusterCreate(BaseModel):
    """Model for creating a cluster."""
    
    name: str = Field(..., description="Cluster name")
    ip: str = Field(..., description="Cluster IP address")
    port: int = Field(..., description="Cluster port")
    description: Optional[str] = Field(None, description="Cluster description")


class ClusterUpdate(BaseModel):
    """Model for updating a cluster."""
    
    name: Optional[str] = Field(None, description="Cluster name")
    ip: Optional[str] = Field(None, description="Cluster IP address")
    port: Optional[int] = Field(None, description="Cluster port")
    description: Optional[str] = Field(None, description="Cluster description")


class Cluster(BaseModel):
    """Model for cluster data."""
    
    id: int = Field(..., description="Cluster ID")
    name: str = Field(..., description="Cluster name")
    ip: str = Field(..., description="Cluster IP address")
    port: int = Field(..., description="Cluster port")
    description: Optional[str] = Field(None, description="Cluster description")
    created_at: str = Field(..., description="Creation timestamp")


class MCPServerEnv(BaseModel):
    """Model for MCP server environment variables."""
    
    key: str = Field(..., description="Environment variable key")
    value: str = Field(..., description="Environment variable value")


class MCPServerCreate(BaseModel):
    """Model for creating an MCP server."""
    
    name: str = Field(..., description="Server name")
    type: str = Field(..., description="Server type (stdio or sse)")
    command: Optional[str] = Field(None, description="Command to execute (for stdio servers)")
    args: Optional[List[str]] = Field(None, description="Command arguments (for stdio servers)")
    url: Optional[str] = Field(None, description="Server URL (for sse servers)")
    cluster_id: Optional[int] = Field(None, description="Cluster ID to associate with")
    is_local: bool = Field(False, description="Whether this is a local server")
    is_default: bool = Field(False, description="Whether this is a default server")
    env: Optional[Dict[str, str]] = Field(None, description="Environment variables")


class MCPServerUpdate(BaseModel):
    """Model for updating an MCP server."""
    
    name: Optional[str] = Field(None, description="Server name")
    type: Optional[str] = Field(None, description="Server type (stdio or sse)")
    command: Optional[str] = Field(None, description="Command to execute (for stdio servers)")
    args: Optional[List[str]] = Field(None, description="Command arguments (for stdio servers)")
    url: Optional[str] = Field(None, description="Server URL (for sse servers)")
    cluster_id: Optional[int] = Field(None, description="Cluster ID to associate with")
    is_local: Optional[bool] = Field(None, description="Whether this is a local server")
    is_default: Optional[bool] = Field(None, description="Whether this is a default server")
    env: Optional[Dict[str, str]] = Field(None, description="Environment variables")


class MCPServer(BaseModel):
    """Model for MCP server data."""
    
    id: int = Field(..., description="Server ID")
    name: str = Field(..., description="Server name")
    type: str = Field(..., description="Server type (stdio or sse)")
    command: Optional[str] = Field(None, description="Command to execute (for stdio servers)")
    args: Optional[List[str]] = Field(None, description="Command arguments (for stdio servers)")
    url: Optional[str] = Field(None, description="Server URL (for sse servers)")
    cluster_id: Optional[int] = Field(None, description="Cluster ID")
    is_local: bool = Field(..., description="Whether this is a local server")
    is_default: bool = Field(..., description="Whether this is a default server")
    env: Dict[str, str] = Field(..., description="Environment variables")
    created_at: str = Field(..., description="Creation timestamp")


class ChatMessage(BaseModel):
    """Model for chat messages."""
    
    message: str = Field(..., description="User message")
    cluster_id: Optional[int] = Field(None, description="Cluster ID")
    stream: bool = Field(False, description="Whether to stream the response")


class ChatResponse(BaseModel):
    """Model for chat responses."""
    
    id: int = Field(..., description="Chat message ID")
    message: str = Field(..., description="Assistant response")


class ChatEvent(BaseModel):
    """Model for streaming chat events."""
    
    type: str = Field(..., description="Event type")
    data: Dict[str, Any] = Field(..., description="Event data")
    timestamp: float = Field(..., description="Event timestamp")


class LLMConfig(BaseModel):
    """Model for LLM configuration."""
    
    provider: str = Field(..., description="LLM provider")
    model: str = Field(..., description="Model name")
    api_key: Optional[str] = Field(None, description="API key")
    base_url: Optional[str] = Field(None, description="Base URL")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")
    is_default: bool = Field(False, description="Whether this is the default configuration")


# Cluster routes
@router.post("/clusters", response_model=Cluster, status_code=status.HTTP_201_CREATED)
async def create_cluster(
    cluster: ClusterCreate,
    db_manager: DatabaseManager = Depends(get_db_manager),
    mcp_hub: MCPHub = Depends(get_mcp_hub),
):
    """Create a new cluster."""
    try:
        cluster_id = db_manager.add_cluster(
            name=cluster.name,
            ip=cluster.ip,
            port=cluster.port,
            description=cluster.description,
        )
        
        created_cluster = db_manager.get_cluster(cluster_id)
        if not created_cluster:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create cluster",
            )
        
        # Automatically add an MCP K8s server for this cluster
        try:
            # Construct the SSE URL using the cluster's IP and port
            sse_url = f"http://{cluster.ip}:{cluster.port}/sse"
            server_name = f"{cluster.name}-mcp-k8s-server"
            
            # Add the MCP server
            await mcp_hub.manager.add_server(
                name=server_name,
                server_type=ServerType.SSE.value,
                url=sse_url,
                cluster_id=cluster_id,
                is_local=False,
                is_default=False,
            )
            
            logger.info(f"Automatically added MCP K8s server {server_name} for cluster {cluster.name}")
        except Exception as e:
            logger.error(f"Failed to automatically add MCP K8s server for cluster {cluster.name}: {str(e)}")
            # Continue even if adding the MCP server fails
            
        return created_cluster
        
    except Exception as e:
        logger.error(f"Failed to create cluster: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create cluster: {str(e)}",
        )


@router.get("/clusters", response_model=List[Cluster])
async def get_clusters(
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Get all clusters."""
    try:
        return db_manager.get_all_clusters()
        
    except Exception as e:
        logger.error(f"Failed to get clusters: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get clusters: {str(e)}",
        )


@router.get("/clusters/{cluster_id}", response_model=Cluster)
async def get_cluster(
    cluster_id: int,
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Get a cluster by ID."""
    try:
        cluster = db_manager.get_cluster(cluster_id)
        if not cluster:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Cluster with ID {cluster_id} not found",
            )
            
        return cluster
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cluster: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cluster: {str(e)}",
        )


@router.put("/clusters/{cluster_id}", response_model=Cluster)
async def update_cluster(
    cluster_id: int,
    cluster: ClusterUpdate,
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Update a cluster."""
    try:
        # Verify cluster exists
        existing_cluster = db_manager.get_cluster(cluster_id)
        if not existing_cluster:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Cluster with ID {cluster_id} not found",
            )
            
        # Update cluster
        success = db_manager.update_cluster(
            cluster_id=cluster_id,
            name=cluster.name,
            ip=cluster.ip,
            port=cluster.port,
            description=cluster.description,
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update cluster",
            )
            
        updated_cluster = db_manager.get_cluster(cluster_id)
        if not updated_cluster:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get updated cluster",
            )
            
        return updated_cluster
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update cluster: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update cluster: {str(e)}",
        )


@router.delete("/clusters/{cluster_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_cluster(
    cluster_id: int,
    db_manager: DatabaseManager = Depends(get_db_manager),
    mcp_hub: MCPHub = Depends(get_mcp_hub),
):
    """Delete a cluster."""
    try:
        # Verify cluster exists
        existing_cluster = db_manager.get_cluster(cluster_id)
        if not existing_cluster:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Cluster with ID {cluster_id} not found",
            )
            
        # Get MCP servers for this cluster
        servers = db_manager.get_mcp_servers_by_cluster(cluster_id)
        
        # Disconnect and delete any associated MCP servers
        for server in servers:
            # First disconnect the server
            await mcp_hub.manager.disconnect_server(server["name"])
            
            # Then delete the server
            server_id = server["id"]
            success = await mcp_hub.manager.remove_server(server_id)
            if success:
                logger.info(f"Deleted MCP server '{server['name']}' associated with cluster '{existing_cluster['name']}'")
            else:
                logger.warning(f"Failed to delete MCP server '{server['name']}' associated with cluster '{existing_cluster['name']}'")
            
        # Delete cluster
        success = db_manager.delete_cluster(cluster_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete cluster",
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete cluster: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete cluster: {str(e)}",
        )


# MCP server routes
@router.post("/mcp-servers", response_model=MCPServer, status_code=status.HTTP_201_CREATED)
async def create_mcp_server(
    server: MCPServerCreate,
    db_manager: DatabaseManager = Depends(get_db_manager),
    mcp_hub: MCPHub = Depends(get_mcp_hub),
):
    """Create a new MCP server."""
    try:
        # Verify server type
        if server.type not in [s.value for s in ServerType]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid server type: {server.type}",
            )
            
        # Verify cluster exists if provided
        if server.cluster_id is not None:
            cluster = db_manager.get_cluster(server.cluster_id)
            if not cluster:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Cluster with ID {server.cluster_id} not found",
                )
                
        # Verify required fields based on server type
        if server.type == ServerType.STDIO.value:
            if not server.command:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Command is required for stdio servers",
                )
        elif server.type == ServerType.SSE.value:
            if not server.url:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="URL is required for SSE servers",
                )
        
        # Create server
        server_id = await mcp_hub.manager.add_server(
            name=server.name,
            server_type=server.type,
            command=server.command,
            args=server.args,
            url=server.url,
            cluster_id=server.cluster_id,
            is_local=server.is_local,
            is_default=server.is_default,
            env=server.env,
        )
        
        created_server = db_manager.get_mcp_server(server_id)
        if not created_server:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create MCP server",
            )
            
        return created_server
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create MCP server: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create MCP server: {str(e)}",
        )


@router.get("/mcp-servers", response_model=List[MCPServer])
async def get_mcp_servers(
    cluster_id: Optional[int] = None,
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Get all MCP servers, optionally filtered by cluster ID."""
    try:
        if cluster_id is not None:
            return db_manager.get_mcp_servers_by_cluster(cluster_id)
        else:
            return db_manager.get_all_mcp_servers()
            
    except Exception as e:
        logger.error(f"Failed to get MCP servers: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get MCP servers: {str(e)}",
        )


@router.get("/mcp-servers/{server_id}", response_model=MCPServer)
async def get_mcp_server(
    server_id: int,
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Get an MCP server by ID."""
    try:
        server = db_manager.get_mcp_server(server_id)
        if not server:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP server with ID {server_id} not found",
            )
            
        return server
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get MCP server: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get MCP server: {str(e)}",
        )


@router.put("/mcp-servers/{server_id}", response_model=MCPServer)
async def update_mcp_server(
    server_id: int,
    server: MCPServerUpdate,
    db_manager: DatabaseManager = Depends(get_db_manager),
    mcp_hub: MCPHub = Depends(get_mcp_hub),
):
    """Update an MCP server."""
    try:
        # Verify server exists
        existing_server = db_manager.get_mcp_server(server_id)
        if not existing_server:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP server with ID {server_id} not found",
            )
            
        # Verify cluster exists if provided
        if server.cluster_id is not None:
            cluster = db_manager.get_cluster(server.cluster_id)
            if not cluster:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Cluster with ID {server.cluster_id} not found",
                )
        
        # Update server
        success = await mcp_hub.manager.update_server(
            server_id=server_id,
            name=server.name,
            server_type=server.type,
            command=server.command,
            args=server.args,
            url=server.url,
            cluster_id=server.cluster_id,
            is_local=server.is_local,
            is_default=server.is_default,
            env=server.env,
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update MCP server",
            )
            
        updated_server = db_manager.get_mcp_server(server_id)
        if not updated_server:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get updated MCP server",
            )
            
        return updated_server
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update MCP server: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update MCP server: {str(e)}",
        )


@router.delete("/mcp-servers/{server_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_mcp_server(
    server_id: int,
    db_manager: DatabaseManager = Depends(get_db_manager),
    mcp_hub: MCPHub = Depends(get_mcp_hub),
):
    """Delete an MCP server."""
    try:
        # Verify server exists
        existing_server = db_manager.get_mcp_server(server_id)
        if not existing_server:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP server with ID {server_id} not found",
            )
            
        # Delete server
        success = await mcp_hub.manager.remove_server(server_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete MCP server",
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete MCP server: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete MCP server: {str(e)}",
        )


@router.post("/mcp-servers/{server_id}/connect", status_code=status.HTTP_200_OK)
async def connect_mcp_server(
    server_id: int,
    db_manager: DatabaseManager = Depends(get_db_manager),
    mcp_hub: MCPHub = Depends(get_mcp_hub),
):
    """Connect to an MCP server."""
    try:
        # Verify server exists
        existing_server = db_manager.get_mcp_server(server_id)
        if not existing_server:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP server with ID {server_id} not found",
            )
            
        # Connect to server
        success = await mcp_hub.manager.connect_server(existing_server["name"])
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to connect to MCP server {existing_server['name']}",
            )
            
        return {"message": f"Connected to MCP server {existing_server['name']}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to connect to MCP server: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to connect to MCP server: {str(e)}",
        )


@router.post("/mcp-servers/{server_id}/disconnect", status_code=status.HTTP_200_OK)
async def disconnect_mcp_server(
    server_id: int,
    db_manager: DatabaseManager = Depends(get_db_manager),
    mcp_hub: MCPHub = Depends(get_mcp_hub),
):
    """Disconnect from an MCP server."""
    try:
        # Verify server exists
        existing_server = db_manager.get_mcp_server(server_id)
        if not existing_server:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP server with ID {server_id} not found",
            )
            
        # Disconnect from server
        success = await mcp_hub.manager.disconnect_server(existing_server["name"])
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to disconnect from MCP server {existing_server['name']}",
            )
            
        return {"message": f"Disconnected from MCP server {existing_server['name']}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disconnect from MCP server: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to disconnect from MCP server: {str(e)}",
        )


# Chat routes
@router.post("/chat", response_model=ChatResponse)
async def chat(
    chat_message: ChatMessage,
    db_manager: DatabaseManager = Depends(get_db_manager),
    react_loop: ReactLoop = Depends(get_react_loop),
    config: Configuration = Depends(get_config)
):
    """Process a chat message and return the response."""
    # If streaming is requested, use the streaming endpoint
    if chat_message.stream:
        return await chat_stream(chat_message, db_manager, react_loop, config)
        
    try:
        # Get the current cluster context if specified
        current_cluster = None
        if chat_message.cluster_id is not None:
            cluster = db_manager.get_cluster(chat_message.cluster_id)
            if not cluster:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Cluster with ID {chat_message.cluster_id} not found",
                )
            current_cluster = cluster["name"]
            
        # Get chat history
        chat_history = db_manager.get_chat_history(limit=10, cluster_id=chat_message.cluster_id)
        conversation_history = [
            (msg["user_message"], msg["assistant_message"])
            for msg in reversed(chat_history)  # Reverse to get oldest messages first
        ]
        
        # Process the message through the ReAct loop
        response, chat_id, _ = await react_loop.run(
            user_message=chat_message.message,
            conversation_history=None,
            current_cluster=current_cluster,
        )
        
        return {"id": chat_id, "message": response}
        
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat message: {str(e)}",
        )


@router.post("/chat/stream")
async def chat_stream(
    chat_message: ChatMessage,
    db_manager: DatabaseManager = Depends(get_db_manager),
    react_loop: ReactLoop = Depends(get_react_loop),
    config: Configuration = Depends(get_config)
):
    """Process a chat message and stream the response using Server-Sent Events (SSE)."""
    
    async def event_generator():
        """Generate SSE events from the ReactLoop."""
        try:
            # Get the current cluster context if specified
            current_cluster = None
            if chat_message.cluster_id is not None:
                cluster = db_manager.get_cluster(chat_message.cluster_id)
                if not cluster:
                    # Send error as an event and return
                    error_data = json.dumps({
                        "type": "error",
                        "data": {
                            "message": f"Cluster with ID {chat_message.cluster_id} not found"
                        },
                        "timestamp": time.time()
                    })
                    yield f"data: {error_data}\n\n"
                    return
                current_cluster = cluster["name"]
                
            # Get chat history
            chat_history = db_manager.get_chat_history(limit=10, cluster_id=chat_message.cluster_id)
            conversation_history = [
                (msg["user_message"], msg["assistant_message"])
                for msg in reversed(chat_history)  # Reverse to get oldest messages first
            ]
            
            # Stream events from the ReAct loop
            final_response = None
            
            async for event in react_loop.run_stream(
                user_message=chat_message.message,
                conversation_history=conversation_history,
                current_cluster=current_cluster,
            ):
                # Convert the event to a JSON string and send as SSE data
                yield f"data: {json.dumps(event)}\n\n"
                
                # Keep track of final response for the chat history
                if event["type"] == "completion":
                    final_response = event["data"].get("result", "")
                elif event["type"] == "response" and event["data"].get("is_final", False):
                    final_response = event["data"].get("content", "")
            
            # Save the chat message to the database if we have a final response
            if final_response:
                chat_id = await react_loop._save_chat_message(
                    user_message=chat_message.message,
                    assistant_message=final_response,
                    current_cluster=current_cluster,
                )
                
                # Send the chat ID as a final event
                yield f"data: {json.dumps({'type': 'chat_id', 'data': {'id': chat_id}, 'timestamp': time.time()})}\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming chat: {str(e)}")
            error_data = json.dumps({
                "type": "error",
                "data": {
                    "message": f"Error processing chat message: {str(e)}"
                },
                "timestamp": time.time()
            })
            yield f"data: {error_data}\n\n"
    
    # Return a streaming response
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )


@router.get("/chat-history", response_model=List[Dict[str, Any]])
async def get_chat_history(
    limit: int = Query(20, ge=1, le=100),
    cluster_id: Optional[int] = None,
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Get chat history, optionally filtered by cluster ID."""
    try:
        return db_manager.get_chat_history(limit=limit, cluster_id=cluster_id)
        
    except Exception as e:
        logger.error(f"Failed to get chat history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get chat history: {str(e)}",
        )


@router.delete("/chat-history", status_code=status.HTTP_204_NO_CONTENT)
async def clear_chat_history(
    cluster_id: Optional[int] = None,
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Clear chat history, optionally filtered by cluster ID."""
    try:
        db_manager.clear_chat_history(cluster_id=cluster_id)
        
    except Exception as e:
        logger.error(f"Failed to clear chat history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear chat history: {str(e)}",
        )


# MCP server status endpoints
@router.get("/mcp-servers/status", response_model=List[ServerStatus])
async def get_all_mcp_servers_status(
    mcp_hub: MCPHub = Depends(get_mcp_hub),
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Get status of all MCP servers."""
    try:
        servers = db_manager.get_all_mcp_servers()
        result = []
        
        for server in servers:
            is_connected = mcp_hub.manager.is_server_connected(server["name"])
            # Explicitly convert types to ensure they match the Pydantic model
            server_status = ServerStatus(
                id=int(server["id"]),
                name=str(server["name"]),
                is_connected=bool(is_connected),
            )
            result.append(server_status.dict())

        return result
        
    except Exception as e:
        logger.error(f"Failed to get MCP server status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get MCP server status: {str(e)}",
        )


@router.get("/mcp-servers/{server_id}/status", response_model=ServerStatus)
async def get_mcp_server_status(
    server_id: int,
    mcp_hub: MCPHub = Depends(get_mcp_hub),
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Get status of a specific MCP server."""
    try:
        server = db_manager.get_mcp_server(server_id)
        if not server:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP server with ID {server_id} not found",
            )
            
        is_connected = mcp_hub.manager.is_server_connected(server["name"])
        
        return ServerStatus(
            id=server["id"],
            name=server["name"],
            is_connected=is_connected,
        ).dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get MCP server status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get MCP server status: {str(e)}",
        )


# MCP server tools and resources endpoints
@router.get("/mcp-servers/{server_id}/tools", response_model=List[Dict[str, Any]])
async def get_mcp_server_tools(
    server_id: int,
    mcp_hub: MCPHub = Depends(get_mcp_hub),
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Get available tools for a specific MCP server."""
    try:
        server = db_manager.get_mcp_server(server_id)
        if not server:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP server with ID {server_id} not found",
            )
            
        # Get server instance from the hub
        server_instance = mcp_hub.manager.get_server_by_name(server["name"])
        if not server_instance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP server instance '{server['name']}' not found in hub",
            )
            
        # Check if server is connected
        if not mcp_hub.manager.is_server_connected(server["name"]):
            return []  # Return empty list if server is not connected
            
        # Get tools from the server using the tools property
        tools = []
        for tool_name in [tool.name for tool in server_instance.tools]:
            tool = server_instance.get_tool(tool_name)
            if tool:
                tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema
                })
                
        return tools
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get MCP server tools: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get MCP server tools: {str(e)}",
        )


@router.get("/mcp-servers/{server_id}/resources", response_model=List[Dict[str, Any]])
async def get_mcp_server_resources(
    server_id: int,
    mcp_hub: MCPHub = Depends(get_mcp_hub),
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Get available resources for a specific MCP server."""
    try:
        server = db_manager.get_mcp_server(server_id)
        if not server:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP server with ID {server_id} not found",
            )
            
        # Get server instance from the hub
        server_instance = mcp_hub.manager.get_server_by_name(server["name"])
        if not server_instance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP server instance '{server['name']}' not found in hub",
            )
            
        # Check if server is connected
        if not mcp_hub.manager.is_server_connected(server["name"]):
            return []  # Return empty list if server is not connected
            
        # Get resources from the server using the resources property
        resources_list = []
        for resource in server_instance.resources:
            resources_list.append({
                "uri": resource.uri,
                "name": resource.name,
                "mime_type": resource.mime_type,
                "description": resource.description
            })
            
        return resources_list
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get MCP server resources: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get MCP server resources: {str(e)}",
        )


# Cluster metrics endpoints
class ClusterMetrics(BaseModel):
    """Base model for cluster metrics."""
    
    cluster_id: int = Field(..., description="Cluster ID")
    cluster_name: str = Field(..., description="Cluster name")
    timestamp: str = Field(..., description="Timestamp of the metrics")


class PerformanceMetrics(ClusterMetrics):
    """Model for cluster performance metrics."""
    
    cpu_usage: List[Dict[str, Any]] = Field(..., description="CPU usage over time")
    memory_usage: List[Dict[str, Any]] = Field(..., description="Memory usage over time")


class HealthMetrics(ClusterMetrics):
    """Model for cluster health metrics."""
    
    node_status: str = Field(..., description="Overall node status")
    pod_health_percentage: float = Field(..., description="Percentage of healthy pods")
    services_count: Dict[str, int] = Field(..., description="Count of services by status")
    pod_status: Dict[str, int] = Field(..., description="Count of pods by status")


class StorageMetrics(ClusterMetrics):
    """Model for cluster storage metrics."""
    
    storage_usage: List[Dict[str, Any]] = Field(..., description="Storage usage by persistent volume")


@router.get("/clusters/{cluster_id}/metrics/performance", response_model=PerformanceMetrics)
async def get_cluster_performance_metrics(
    cluster_id: int,
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Get performance metrics for a specific cluster."""
    try:
        cluster = db_manager.get_cluster(cluster_id)
        if not cluster:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Cluster with ID {cluster_id} not found",
            )
            
        # In a real implementation, this would fetch actual metrics from the Kubernetes cluster
        # For now, return example metrics
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Example CPU usage data
        cpu_data = [
            {"time": f"T-{i}", "usage": round(40 + 30 * (0.5 + 0.5 * (i % 3 - 1)), 2)}
            for i in range(10, 0, -1)
        ]
        
        # Example memory usage data
        memory_data = [
            {"time": f"T-{i}", "usage_gb": round(4 + 2 * (0.5 + 0.5 * (i % 5 - 2)), 2)}
            for i in range(10, 0, -1)
        ]
        
        return {
            "cluster_id": cluster_id,
            "cluster_name": cluster["name"],
            "timestamp": current_time,
            "cpu_usage": cpu_data,
            "memory_usage": memory_data,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cluster performance metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cluster performance metrics: {str(e)}",
        )


@router.get("/clusters/{cluster_id}/metrics/health", response_model=HealthMetrics)
async def get_cluster_health_metrics(
    cluster_id: int,
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Get health metrics for a specific cluster."""
    try:
        cluster = db_manager.get_cluster(cluster_id)
        if not cluster:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Cluster with ID {cluster_id} not found",
            )
            
        # In a real implementation, this would fetch actual metrics from the Kubernetes cluster
        # For now, return example metrics
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Example pod status
        pod_status = {
            "Running": 25,
            "Pending": 2,
            "Failed": 0,
            "Succeeded": 5,
            "Unknown": 0,
        }
        
        return {
            "cluster_id": cluster_id,
            "cluster_name": cluster["name"],
            "timestamp": current_time,
            "node_status": "Healthy",
            "pod_health_percentage": 98.0,
            "services_count": {"Total": 15, "Healthy": 15},
            "pod_status": pod_status,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cluster health metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cluster health metrics: {str(e)}",
        )


@router.get("/clusters/{cluster_id}/metrics/storage", response_model=StorageMetrics)
async def get_cluster_storage_metrics(
    cluster_id: int,
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Get storage metrics for a specific cluster."""
    try:
        cluster = db_manager.get_cluster(cluster_id)
        if not cluster:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Cluster with ID {cluster_id} not found",
            )
            
        # In a real implementation, this would fetch actual metrics from the Kubernetes cluster
        # For now, return example metrics
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Example storage data
        storage_data = [
            {
                "pv_name": f"pv-{i}",
                "capacity_gb": capacity,
                "used_gb": used,
                "available_gb": capacity - used,
                "usage_percentage": round((used / capacity) * 100, 1),
            }
            for i, (capacity, used) in enumerate(
                [(100, 78), (50, 32), (200, 150), (75, 60), (150, 25)], 1
            )
        ]
        
        return {
            "cluster_id": cluster_id,
            "cluster_name": cluster["name"],
            "timestamp": current_time,
            "storage_usage": storage_data,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cluster storage metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cluster storage metrics: {str(e)}",
        )


# Kubernetes resources endpoints
@router.get("/clusters/{cluster_id}/nodes", response_model=List[Dict[str, Any]])
async def get_cluster_nodes(
    cluster_id: int,
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Get nodes for a specific cluster."""
    try:
        cluster = db_manager.get_cluster(cluster_id)
        if not cluster:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Cluster with ID {cluster_id} not found",
            )
            
        # In a real implementation, this would fetch actual nodes from the Kubernetes cluster
        # For now, return example nodes
        nodes = [
            {
                "name": f"node-{i}",
                "status": "Ready",
                "role": "worker" if i > 1 else "master",
                "cpu": "4",
                "memory": "16Gi",
                "kubernetes_version": "1.26.0",
                "created_at": "2024-04-01T12:00:00Z",
            }
            for i in range(1, 4)
        ]
        
        return nodes
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cluster nodes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cluster nodes: {str(e)}",
        )


@router.get("/clusters/{cluster_id}/pods", response_model=List[Dict[str, Any]])
async def get_cluster_pods(
    cluster_id: int,
    namespace: Optional[str] = None,
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Get pods for a specific cluster, optionally filtered by namespace."""
    try:
        cluster = db_manager.get_cluster(cluster_id)
        if not cluster:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Cluster with ID {cluster_id} not found",
            )
            
        # In a real implementation, this would fetch actual pods from the Kubernetes cluster
        # For now, return example pods
        namespaces = ["default", "kube-system", "monitoring"] if namespace is None else [namespace]
        
        pods = []
        for ns in namespaces:
            ns_pods = [
                {
                    "name": f"pod-{ns}-{i}",
                    "namespace": ns,
                    "status": "Running" if i % 5 != 0 else "Pending",
                    "node": f"node-{(i % 3) + 1}",
                    "ip": f"10.0.{ns.replace('-', '.')}.{i}",
                    "created_at": "2024-04-01T12:00:00Z",
                }
                for i in range(1, 6)
            ]
            pods.extend(ns_pods)
        
        return pods
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cluster pods: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cluster pods: {str(e)}",
        )


@router.get("/clusters/{cluster_id}/services", response_model=List[Dict[str, Any]])
async def get_cluster_services(
    cluster_id: int,
    namespace: Optional[str] = None,
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Get services for a specific cluster, optionally filtered by namespace."""
    try:
        cluster = db_manager.get_cluster(cluster_id)
        if not cluster:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Cluster with ID {cluster_id} not found",
            )
            
        # In a real implementation, this would fetch actual services from the Kubernetes cluster
        # For now, return example services
        namespaces = ["default", "kube-system", "monitoring"] if namespace is None else [namespace]
        
        services = []
        for ns in namespaces:
            ns_services = [
                {
                    "name": f"service-{ns}-{i}",
                    "namespace": ns,
                    "type": "ClusterIP" if i % 3 != 0 else "LoadBalancer",
                    "cluster_ip": f"10.1.{ns.replace('-', '.')}.{i}",
                    "external_ip": f"192.168.1.{i}" if i % 3 == 0 else None,
                    "ports": [
                        {"port": 80, "target_port": 8080, "protocol": "TCP"}
                    ],
                    "created_at": "2024-04-01T12:00:00Z",
                }
                for i in range(1, 4)
            ]
            services.extend(ns_services)
        
        return services
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cluster services: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cluster services: {str(e)}",
        )


@router.get("/clusters/{cluster_id}/persistent-volumes", response_model=List[Dict[str, Any]])
async def get_cluster_persistent_volumes(
    cluster_id: int,
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Get persistent volumes for a specific cluster."""
    try:
        cluster = db_manager.get_cluster(cluster_id)
        if not cluster:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Cluster with ID {cluster_id} not found",
            )
            
        # In a real implementation, this would fetch actual PVs from the Kubernetes cluster
        # For now, return example PVs
        pvs = [
            {
                "name": f"pv-{i}",
                "capacity": f"{capacity}Gi",
                "access_modes": ["ReadWriteOnce"],
                "reclaim_policy": "Retain",
                "status": "Bound",
                "claim": f"default/pvc-{i}",
                "storage_class": "standard",
                "created_at": "2024-04-01T12:00:00Z",
            }
            for i, capacity in enumerate([100, 50, 200, 75, 150], 1)
        ]
        
        return pvs
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cluster persistent volumes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cluster persistent volumes: {str(e)}",
        )


# LLM configuration routes
@router.post("/llm-config", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_llm_config(
    config_data: LLMConfig,
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Create a new LLM configuration."""
    try:
        # Verify provider is valid
        if config_data.provider not in [p.value for p in LLMProvider]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid LLM provider: {config_data.provider}",
            )
            
        # Create config
        config_id = db_manager.add_llm_config(
            provider=config_data.provider,
            model=config_data.model,
            api_key=config_data.api_key,
            base_url=config_data.base_url,
            parameters=config_data.parameters,
            is_default=config_data.is_default,
        )
        
        created_config = db_manager.get_llm_config(config_id)
        if not created_config:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create LLM configuration",
            )
            
        return created_config
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create LLM configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create LLM configuration: {str(e)}",
        )


@router.get("/llm-config", response_model=List[Dict[str, Any]])
async def get_llm_configs(
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Get all LLM configurations."""
    try:
        return db_manager.get_all_llm_configs()
        
    except Exception as e:
        logger.error(f"Failed to get LLM configurations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get LLM configurations: {str(e)}",
        )


@router.get("/llm-config/{config_id}", response_model=Dict[str, Any])
async def get_llm_config(
    config_id: int,
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Get an LLM configuration by ID."""
    try:
        config = db_manager.get_llm_config(config_id)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"LLM configuration with ID {config_id} not found",
            )
            
        return config
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get LLM configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get LLM configuration: {str(e)}",
        )


@router.put("/llm-config/{config_id}", response_model=Dict[str, Any])
async def update_llm_config(
    config_id: int,
    config_data: LLMConfig,
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Update an LLM configuration."""
    try:
        # Verify config exists
        existing_config = db_manager.get_llm_config(config_id)
        if not existing_config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"LLM configuration with ID {config_id} not found",
            )
            
        # Update config
        success = db_manager.update_llm_config(
            config_id=config_id,
            provider=config_data.provider,
            model=config_data.model,
            api_key=config_data.api_key,
            base_url=config_data.base_url,
            parameters=config_data.parameters,
            is_default=config_data.is_default,
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update LLM configuration",
            )
            
        updated_config = db_manager.get_llm_config(config_id)
        if not updated_config:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get updated LLM configuration",
            )
            
        return updated_config
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update LLM configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update LLM configuration: {str(e)}",
        )


@router.delete("/llm-config/{config_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_llm_config(
    config_id: int,
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Delete an LLM configuration."""
    try:
        # Verify config exists
        existing_config = db_manager.get_llm_config(config_id)
        if not existing_config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"LLM configuration with ID {config_id} not found",
            )
            
        # Delete config
        success = db_manager.delete_llm_config(config_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete LLM configuration",
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete LLM configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete LLM configuration: {str(e)}",
        )


# Register router with app
app.include_router(router, prefix="/api")
