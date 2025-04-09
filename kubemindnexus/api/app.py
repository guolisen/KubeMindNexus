"""FastAPI application for KubeMindNexus."""
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from ..config.config import get_config
from ..utils.logger import LoggerMixin
from . import routes


class APIServer(LoggerMixin):
    """FastAPI server for KubeMindNexus."""
    
    def __init__(self) -> None:
        """Initialize the API server."""
        self.app = FastAPI(
            title="KubeMindNexus API",
            description="API for managing Kubernetes clusters with MCP and LLM capabilities",
            version="0.1.0",
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, this should be restricted
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Include routers
        self.app.include_router(routes.clusters_router, prefix="/api/clusters", tags=["clusters"])
        self.app.include_router(routes.chat_router, prefix="/api/chat", tags=["chat"])
        self.app.include_router(routes.mcp_router, prefix="/api/mcp", tags=["mcp"])
        self.app.include_router(routes.llm_router, prefix="/api/llm", tags=["llm"])
        self.app.include_router(routes.health_router, prefix="/api/health", tags=["health"])
        
        # Add startup and shutdown events
        self.setup_events()
    
    def setup_events(self) -> None:
        """Set up startup and shutdown events."""
        
        @self.app.on_event("startup")
        async def startup_event() -> None:
            """Event handler for application startup."""
            self.logger.info("Starting KubeMindNexus API server")
        
        @self.app.on_event("shutdown")
        async def shutdown_event() -> None:
            """Event handler for application shutdown."""
            self.logger.info("Shutting down KubeMindNexus API server")
    
    def start(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        """Start the API server.
        
        Args:
            host: The host to bind to. If None, uses the one from config.
            port: The port to bind to. If None, uses the one from config.
        """
        import uvicorn
        
        config = get_config()
        host = host or config.get_api_host()
        port = port or config.get_api_port()
        
        self.logger.info(f"Starting KubeMindNexus API server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Singleton instance
api_server = APIServer()
