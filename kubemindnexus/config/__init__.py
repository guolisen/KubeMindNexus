"""Configuration module for KubeMindNexus."""

import json
import os
import pkg_resources
from typing import Dict, Any

# Default configuration path
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "default_config.json")

# Load default configuration
def get_default_config() -> Dict[str, Any]:
    """Get default configuration.
    
    Returns:
        Default configuration dictionary.
    """
    try:
        with open(DEFAULT_CONFIG_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        # Fallback to empty configuration
        return {}
