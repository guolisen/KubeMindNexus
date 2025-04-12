#!/usr/bin/env python
"""
Script to directly run run_ui_server() and run_app() and wait for completion.
"""

import logging
import asyncio
import threading
from kubemindnexus.config import Configuration
from kubemindnexus.database import DatabaseManager
from kubemindnexus.mcp.hub import MCPHub
from kubemindnexus.llm.base import LLMFactory
from kubemindnexus.llm.react import ReactLoop
from main import run_ui_server, startup_mcp_servers
from kubemindnexus.ui.app import run_app
from kubemindnexus.constants import DEFAULT_UI_PORT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("direct_run")

def main():
    """Main entry point to directly run the UI server and app."""
    try:
        # Initialize configuration
        config = Configuration()
        logger.info("Configuration initialized.")
        
        # Initialize database
        db_manager = DatabaseManager(config)
        logger.info("Database initialized.")
        
        # Initialize MCP hub
        mcp_hub = MCPHub(config, db_manager)
        logger.info("MCP hub initialized.")
        
        # Connect to default MCP servers
        asyncio.run(startup_mcp_servers(config, mcp_hub))
        
        # Initialize LLM factory
        llm_factory = LLMFactory(config, db_manager)
        logger.info("LLM factory initialized.")
        
        # Initialize default LLM
        llm_config = config.config.llm
        provider = llm_config.get("default_provider", "openai")
        provider_config = llm_config.get("providers", {}).get(provider, {})
        
        default_llm = llm_factory.create_llm(
            provider=provider,
            model=provider_config.get("model", "gpt-4o"),
            api_key=provider_config.get("api_key"),
            base_url=provider_config.get("base_url"),
            parameters=provider_config.get("parameters", {})
        )
        
        # Initialize ReactLoop
        react_loop = ReactLoop(config, db_manager, mcp_hub, default_llm)
        logger.info("ReactLoop initialized.")
        
        # Get UI port from config or use default
        port = config.config.ui_port if hasattr(config.config, 'ui_port') else DEFAULT_UI_PORT
        
        # Looking at the original implementation, run_ui_server() runs Streamlit and does not return until closed
        # So to truly run both functions directly and hang waiting for run_app, we need to:
        
        # 1. Create StreamlitApp instance directly (skipping run_ui_server's bootstrap)
        logger.info("Directly running run_app() to handle the UI...")
        app = StreamlitApp(config, db_manager, mcp_hub, react_loop)
        
        # 2. Run the app directly - this will hang/wait until completed
        logger.info("Waiting for run_app() to complete...")
        app.run()
        
        # Code will only reach here after run_app() completes
        logger.info("run_app() has completed.")
        
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user.")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        # Clean up resources
        if 'db_manager' in locals():
            db_manager.close()
        logger.info("Execution completed.")

if __name__ == "__main__":
    main()
