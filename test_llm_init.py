#!/usr/bin/env python
"""Test script for LLM initialization."""

import os
import logging
import asyncio
from kubemindnexus.config import Configuration
from kubemindnexus.llm.base import LLMFactory
from kubemindnexus.constants import LLMProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger("kubemindnexus.test")

async def test_llm_initialization():
    """Test the LLM initialization with the new configuration system."""
    # Initialize configuration
    config = Configuration()
    logger.info("Configuration initialized.")
    
    # Get LLM configuration from config
    llm_config = config.config.llm
    provider = llm_config.get("default_provider", "openai")
    provider_config = llm_config.get("providers", {}).get(provider, {})
    
    # Initialize LLM factory
    llm_factory = LLMFactory(config)
    logger.info("LLM factory initialized.")
    
    # Get API key from config or environment variable
    api_key = provider_config.get("api_key")
    if not api_key and provider in ["openai", "deepseek", "openrouter"]:
        env_var_name = f"{provider.upper()}_API_KEY"
        api_key = os.environ.get(env_var_name)
        if api_key:
            logger.info(f"Using API key from environment variable {env_var_name}")
        else:
            logger.warning(f"No API key found for {provider} provider.")
            
            # Try with Ollama as fallback if no API key is available
            if "ollama" in llm_config.get("providers", {}):
                logger.info("Falling back to Ollama provider.")
                provider = "ollama"
                provider_config = llm_config.get("providers", {}).get(provider, {})

    logger.info(f"Initializing {provider} LLM with model {provider_config.get('model')}")
    
    try:
        # Initialize LLM
        llm = llm_factory.create_llm(
            provider=provider,
            model=provider_config.get("model"),
            api_key=api_key,
            base_url=provider_config.get("base_url"),
            parameters=provider_config.get("parameters", {})
        )
        logger.info("LLM initialization successful!")
        return True
    except Exception as e:
        logger.error(f"LLM initialization failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Run the test
    result = asyncio.run(test_llm_initialization())
    
    # Print result
    if result:
        print("\nSUCCESS: LLM initialization completed successfully.")
    else:
        print("\nFAILURE: LLM initialization failed.")
