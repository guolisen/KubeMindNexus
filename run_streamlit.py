#!/usr/bin/env python
"""Streamlit launcher for KubeMindNexus.

This script properly initializes the KubeMindNexus app with Streamlit,
ensuring that relative imports work correctly.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import and use the run_app function from kubemindnexus.ui.app
from kubemindnexus.config import Configuration
from kubemindnexus.database import DatabaseManager
from kubemindnexus.mcp.hub import MCPHub
from kubemindnexus.llm.base import LLMFactory
from kubemindnexus.llm.react import ReactLoop
from kubemindnexus.ui.app import run_app

# Initialize components
config = Configuration()
db_manager = DatabaseManager(config)
mcp_hub = MCPHub(config, db_manager)

# Initialize LLM for ReactLoop
llm_factory = LLMFactory(config, db_manager)
provider = config.config.llm.get("default_provider", "openai")
provider_config = config.config.llm.get("providers", {}).get(provider, {})
default_llm = llm_factory.create_llm(
    provider=provider,
    model=provider_config.get("model", "gpt-4o"),
    api_key=provider_config.get("api_key"),
    base_url=provider_config.get("base_url"),
    parameters=provider_config.get("parameters", {})
)

# Initialize ReactLoop
react_loop = ReactLoop(config, db_manager, mcp_hub, default_llm)

# Run the Streamlit app
if __name__ == "__main__":
    # Call the run_app function with our initialized components
    print("Starting KubeMindNexus UI with Streamlit...")
    run_app(config, db_manager, mcp_hub, react_loop)
