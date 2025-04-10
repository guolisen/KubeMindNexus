#!/usr/bin/env python
"""Command-line tool to set LLM API keys in KubeMindNexus config."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger("kubemindnexus.set_api_key")

def set_api_key(provider, api_key):
    """Set API key for a specific LLM provider in the config file.
    
    Args:
        provider: LLM provider (openai, deepseek, openrouter, ollama)
        api_key: API key to set
        
    Returns:
        True if successful, False otherwise
    """
    # Check for valid provider
    if provider not in ["openai", "deepseek", "openrouter", "ollama"]:
        logger.error(f"Invalid provider: {provider}")
        return False
        
    try:
        # Find config file path
        home_dir = Path.home()
        config_dir = home_dir / ".config" / "kubemindnexus"
        config_file = config_dir / "config.json"
        
        # If config file doesn't exist, check if default config exists
        if not config_file.exists():
            default_config = Path(__file__).parent / "kubemindnexus" / "config" / "default_config.json"
            if not default_config.exists():
                logger.error("No configuration file found.")
                return False
                
            # Create directory if it doesn't exist
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy default config
            with open(default_config, "r") as f:
                config_data = json.load(f)
                
            # Write config to user config directory
            with open(config_file, "w") as f:
                json.dump(config_data, f, indent=2)
                
            logger.info(f"Created new config file at {config_file}")
        
        # Read current config
        with open(config_file, "r") as f:
            config_data = json.load(f)
            
        # Make sure llm config exists
        if "llm" not in config_data:
            config_data["llm"] = {
                "default_provider": provider,
                "providers": {}
            }
            
        # Make sure providers config exists
        if "providers" not in config_data["llm"]:
            config_data["llm"]["providers"] = {}
            
        # Make sure provider config exists
        if provider not in config_data["llm"]["providers"]:
            config_data["llm"]["providers"][provider] = {
                "model": "gpt-4o" if provider == "openai" else 
                         "deepseek-chat" if provider == "deepseek" else
                         "anthropic/claude-3-opus" if provider == "openrouter" else
                         "llama3",
                "api_key": "",
                "base_url": "https://api.openai.com/v1" if provider == "openai" else
                            "https://api.deepseek.com/v1" if provider == "deepseek" else
                            "https://openrouter.ai/api/v1" if provider == "openrouter" else
                            "http://localhost:11434",
                "parameters": {"temperature": 0.7, "max_tokens": 1000}
            }
            
        # Set API key
        config_data["llm"]["providers"][provider]["api_key"] = api_key
        
        # Write updated config
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)
            
        logger.info(f"Successfully set API key for {provider} in {config_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error setting API key: {str(e)}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Set API key for a specific LLM provider in KubeMindNexus"
    )
    
    parser.add_argument(
        "provider",
        choices=["openai", "deepseek", "openrouter", "ollama"],
        help="LLM provider to set API key for",
    )
    
    parser.add_argument(
        "api_key",
        help="API key to set (use empty string to clear)",
    )
    
    args = parser.parse_args()
    
    # Set API key
    success = set_api_key(args.provider, args.api_key)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
