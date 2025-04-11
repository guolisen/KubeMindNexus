#!/usr/bin/env python
"""KubeMindNexus API Server - Kubernetes clusters management with Model Context Protocol.

This is the API server for the KubeMindNexus application. It initializes
the configuration, database, MCP hub, and serves the REST API.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI

from kubemindnexus.api.routes import app as api_app
from kubemindnexus.config import Configuration
from kubemindnexus.constants import Colors, DEFAULT_API_HOST, DEFAULT_API_PORT
from kubemindnexus.database import DatabaseManager
from kubemindnexus.llm.base import LLMFactory
from kubemindnexus.llm.react import ReactLoop
from kubemindnexus.mcp.hub import MCPHub

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Suppress watchdog debug logs
logging.getLogger('watchdog').setLevel(logging.WARNING)

logger = logging.getLogger("kubemindnexus.api")


def setup_api_app(
    config: Configuration,
    db_manager: DatabaseManager,
    mcp_hub: MCPHub,
    react_loop: ReactLoop,
) -> None:
    """Set up the FastAPI application with required dependencies.
    
    Args:
        config: Configuration instance.
        db_manager: Database manager instance.
        mcp_hub: MCP hub instance.
        react_loop: ReactLoop instance.
    """
    # Set up application state
    api_app.state.config = config
    api_app.state.db_manager = db_manager
    api_app.state.mcp_hub = mcp_hub
    api_app.state.react_loop = react_loop


async def startup_mcp_servers(
    config: Configuration,
    mcp_hub: MCPHub,
) -> None:
    """Connect to default MCP servers on startup.
    
    Args:
        config: Configuration instance.
        mcp_hub: MCP hub instance.
    """
    default_servers = config.get_default_mcp_servers()
    
    if not default_servers:
        logger.info("No default MCP servers configured.")
        return
        
    logger.info(f"Connecting to {len(default_servers)} default MCP servers...")
    
    for server_name in default_servers:
        try:
            success = await mcp_hub.manager.connect_server(server_name)
            
            if success:
                logger.info(f"Connected to MCP server: {server_name}")
            else:
                logger.warning(f"Failed to connect to MCP server: {server_name}")
                
        except Exception as e:
            logger.error(f"Error connecting to MCP server {server_name}: {str(e)}")


async def run_api_server(
    config: Configuration,
    host: str,
    port: int,
) -> None:
    """Run the FastAPI server.
    
    Args:
        config: Configuration instance.
        host: API server host.
        port: API server port.
    """
    try:
        logger.info(f"Starting API server on {host}:{port}...")
        
        # Use uvicorn.Server for async handling
        uvicorn_config = uvicorn.Config(
            "kubemindnexus.api.routes:app",
            host=host,
            port=port,
            log_level="info",
        )
        server = uvicorn.Server(uvicorn_config)
        await server.serve()
        
    except Exception as e:
        logger.error(f"Error running API server: {str(e)}")


async def run_server(
    config : Configuration,
    api_host: str,
    api_port: int
) -> None:
    """Run the KubeMindNexus API server.
    
    Args:
        config: Configuration instance.
        api_host: API server host.
        api_port: API server port.
    """
    try:
        # Initialize database
        db_manager = DatabaseManager(config)
        logger.info("Database initialized.")
        
        # Initialize MCP hub
        mcp_hub = MCPHub(config, db_manager)
        logger.info("MCP hub initialized.")
        
        # Initialize LLM factory
        llm_factory = LLMFactory(config, db_manager)
        logger.info("LLM factory initialized.")
        
        # Initialize default LLM for ReactLoop
        # Try to get LLM config from database first
        default_llm_config = None #db_manager.get_default_llm_config()
        
        # If no LLM config in database, use the one from config file
        if not default_llm_config:
            logger.info("No LLM configuration found in database. Using configuration from config file.")
            llm_config = config.config.llm
            provider = llm_config.get("default_provider", "openai")
            provider_config = llm_config.get("providers", {}).get(provider, {})
            
            # Get API key from config or environment variable
            api_key = provider_config.get("api_key")
            if not api_key and provider in ["openai", "deepseek", "openrouter"]:
                env_var_name = f"{provider.upper()}_API_KEY"
                api_key = os.environ.get(env_var_name)
                if api_key:
                    logger.info(f"Using API key from environment variable {env_var_name}")
            
            default_llm = llm_factory.create_llm(
                provider=provider,
                model=provider_config.get("model", "gpt-4o"),
                api_key=api_key,
                base_url=provider_config.get("base_url"),
                parameters=provider_config.get("parameters", {})
            )
        else:
            default_llm = llm_factory.create_llm(
                provider=default_llm_config["provider"],
                model=default_llm_config["model"],
                api_key=default_llm_config.get("api_key"),
                base_url=default_llm_config.get("base_url"),
                parameters=default_llm_config.get("parameters", {})
            )
        
        # Initialize ReactLoop with correct parameter order
        react_loop = ReactLoop(config, db_manager, mcp_hub, default_llm)
        #logger.info("ReactLoop initialized.")
        
        # Set up API app
        setup_api_app(config, db_manager, mcp_hub, react_loop)
        logger.info("API app set up.")
        
        # Connect to default MCP servers
        await startup_mcp_servers(config, mcp_hub)
        
        # Start API server
        logger.info(f"Starting API server...")
        await run_api_server(config, api_host, api_port)
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested...")
        
    except Exception as e:
        logger.error(f"Error running server: {str(e)}")
        
    finally:
        # Clean up resources
        if 'db_manager' in locals():
            db_manager.close()
            
        logger.info("Server shutdown complete.")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="KubeMindNexus API Server - Kubernetes clusters management with MCP"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_API_HOST,
        help=f"API server host (default: {DEFAULT_API_HOST})",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_API_PORT,
        help=f"API server port (default: {DEFAULT_API_PORT})",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    return parser.parse_args()


def print_banner() -> None:
    """Print application banner."""
    banner = f"""
{Colors.OKBLUE}
 ██ ▄█▀ █    ██  ▄▄▄▄   ▓█████  ███▄ ▄███▓ ██▓ ███▄    █ ▓█████▄  ███▄    █ ▓█████ ▒██   ██▒█    ██   ██████ 
 ██▄█▒  ██  ▓██▒▓█████▄ ▓█   ▀ ▓██▒▀█▀ ██▒▓██▒ ██ ▀█   █ ▒██▀ ██▌ ██ ▀█   █ ▓█   ▀ ▒▒ █ █ ▒░██  ▓██▒▒██    ▒ 
▓███▄░ ▓██  ▒██░▒██▒ ▄██▒███   ▓██    ▓██░▒██▒▓██  ▀█ ██▒░██   █▌▓██  ▀█ ██▒▒███   ░░  █   ░▓██  ▒██░░ ▓██▄   
▓██ █▄ ▓▓█  ░██░▒██░█▀  ▒▓█  ▄ ▒██    ▒██ ░██░▓██▒  ▐▌██▒░▓█▄   ▌▓██▒  ▐▌██▒▒▓█  ▄  ░ █ █ ▒ ▓▓█  ░██░  ▒   ██▒
▒██▒ █▄▒▒█████▓ ░▓█  ▀█▓░▒████▒▒██▒   ░██▒░██░▒██░   ▓██░░▒████▓ ▒██░   ▓██░░▒████▒▒██▒ ▒██▒▒▒█████▓ ▒██████▒▒
▒ ▒▒ ▓▒░▒▓▒ ▒ ▒ ░▒▓███▀▒░░ ▒░ ░░ ▒░   ░  ░░▓  ░ ▒░   ▒ ▒  ▒▒▓  ▒ ░ ▒░   ▒ ▒ ░░ ▒░ ░▒▒ ░ ░▓ ░░▒▓▒ ▒ ▒ ▒ ▒▓▒ ▒ ░
░ ░▒ ▒░░░▒░ ░ ░ ▒░▒   ░  ░ ░  ░░  ░      ░ ▒ ░░ ░░   ░ ▒░ ░ ▒  ▒ ░ ░░   ░ ▒░ ░ ░  ░░░   ░▒ ░░░▒░ ░ ░ ░ ░▒  ░ ░
░ ░░ ░  ░░░ ░ ░  ░    ░    ░   ░      ░    ▒ ░   ░   ░ ░  ░ ░  ░    ░   ░ ░    ░    ░    ░   ░░░ ░ ░ ░  ░  ░  
░  ░      ░      ░         ░  ░       ░    ░           ░    ░             ░    ░  ░ ░    ░     ░           ░  
                      ░                                    ░                                                   
{Colors.ENDC}
{Colors.OKGREEN}KubeMindNexus API Server - Kubernetes clusters management with Model Context Protocol{Colors.ENDC}
{Colors.HEADER}Version: 0.1.0{Colors.ENDC}
"""
    print(banner)


def main() -> None:
    """Main entry point for the API server."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize configuration
    config = Configuration(args.config)
    logger.info("Configuration initialized.")
    
    # Run API server in the main thread
    logger.info("Starting API server...")
    asyncio.run(
        run_server(config, args.host, args.port)
    )


if __name__ == "__main__":
    main()
