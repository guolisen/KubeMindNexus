"""Configuration module for KubeMindNexus."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Reference parent module constants directly
from ..constants import (DEFAULT_API_HOST, DEFAULT_API_PORT, DEFAULT_DB_PATH,
                       DEFAULT_SYSTEM_PROMPT, DEFAULT_UI_PORT,
                       REACT_MAX_ITERATIONS, REACT_SAFETY_TIMEOUT)

logger = logging.getLogger(__name__)


class Config:
    """Configuration data class."""
    
    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        system_prompt_template: str = DEFAULT_SYSTEM_PROMPT,
        react_max_iterations: int = REACT_MAX_ITERATIONS,
        react_safety_timeout: int = REACT_SAFETY_TIMEOUT,
        ui_port: int = DEFAULT_UI_PORT,
        api_host: str = DEFAULT_API_HOST,
        api_port: int = DEFAULT_API_PORT,
        default_mcp_servers: Optional[List[str]] = None,
        llm: Optional[Dict[str, Any]] = None,
    ):
        """Initialize configuration.
        
        Args:
            db_path: SQLite database path.
            system_prompt_template: System prompt template for LLM.
            react_max_iterations: Maximum iterations for ReAct loop.
            react_safety_timeout: Safety timeout for ReAct loop (seconds).
            ui_port: UI server port.
            api_host: API server host.
            api_port: API server port.
            default_mcp_servers: Default MCP servers to connect on startup.
        """
        self.db_path = db_path
        self.system_prompt_template = system_prompt_template
        self.react_max_iterations = react_max_iterations
        self.react_safety_timeout = react_safety_timeout
        self.ui_port = ui_port
        self.api_host = api_host
        self.api_port = api_port
        self.default_mcp_servers = default_mcp_servers or []
        self.llm = llm or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration.
        """
        return {
            "db_path": self.db_path,
            "system_prompt_template": self.system_prompt_template,
            "react_max_iterations": self.react_max_iterations,
            "react_safety_timeout": self.react_safety_timeout,
            "ui_port": self.ui_port,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "default_mcp_servers": self.default_mcp_servers,
            "llm": self.llm,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary.
        
        Args:
            data: Dictionary representation of configuration.
            
        Returns:
            Configuration instance.
        """
        return cls(
            db_path=data.get("db_path", DEFAULT_DB_PATH),
            system_prompt_template=data.get("system_prompt_template", DEFAULT_SYSTEM_PROMPT),
            react_max_iterations=data.get("react_max_iterations", REACT_MAX_ITERATIONS),
            react_safety_timeout=data.get("react_safety_timeout", REACT_SAFETY_TIMEOUT),
            ui_port=data.get("ui_port", DEFAULT_UI_PORT),
            api_host=data.get("api_host", DEFAULT_API_HOST),
            api_port=data.get("api_port", DEFAULT_API_PORT),
            llm=data.get("llm", {}),
            default_mcp_servers=data.get("default_mcp_servers", []),

        )


class Configuration:
    """Configuration manager for KubeMindNexus."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Optional path to configuration file.
                If not provided, default paths will be checked.
        """
        self.config_path = self._resolve_config_path(config_path)
        self.config = self._load_config()
    
    def _resolve_config_path(self, config_path: Optional[str] = None) -> Path:
        """Resolve configuration file path.
        
        Args:
            config_path: Optional explicit configuration file path.
            
        Returns:
            Path to configuration file.
        """
        if config_path:
            return Path(config_path)
            
        # Check common configuration locations
        home_dir = Path.home()
        config_dir = home_dir / ".config" / "kubemindnexus"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Default config file path
        return config_dir / "config.json"
    
    def _load_config(self) -> Config:
        """Load configuration from file.
        
        Returns:
            Configuration instance.
        """
        try:
            if self.config_path.exists():
                with open(self.config_path, "r") as f:
                    data = json.load(f)
                    
                logger.info(f"Loaded configuration from {self.config_path}")
                return Config.from_dict(data)
            else:
                # Create default configuration
                default_config = Config()
                
                # Save default configuration
                self.save_config(default_config)
                
                logger.info(f"Created default configuration at {self.config_path}")
                return default_config
                
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            
            # Fallback to default configuration
            return Config()
    
    def save_config(self, config: Optional[Config] = None) -> bool:
        """Save configuration to file.
        
        Args:
            config: Configuration to save. If not provided, current configuration will be saved.
            
        Returns:
            True if save was successful, False otherwise.
        """
        config_to_save = config or self.config
        
        try:
            # Create parent directory if it doesn't exist
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            with open(self.config_path, "w") as f:
                json.dump(config_to_save.to_dict(), f, indent=2)
                
            if config:
                self.config = config
                
            logger.info(f"Saved configuration to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def get_default_mcp_servers(self) -> List[str]:
        """Get default MCP servers.
        
        Returns:
            List of default MCP server names.
        """
        return self.config.default_mcp_servers
    
    def add_default_mcp_server(self, server_name: str) -> bool:
        """Add a default MCP server.
        
        Args:
            server_name: Server name.
            
        Returns:
            True if server was added, False otherwise.
        """
        if server_name in self.config.default_mcp_servers:
            return True
            
        self.config.default_mcp_servers.append(server_name)
        return self.save_config()
    
    def remove_default_mcp_server(self, server_name: str) -> bool:
        """Remove a default MCP server.
        
        Args:
            server_name: Server name.
            
        Returns:
            True if server was removed, False otherwise.
        """
        if server_name not in self.config.default_mcp_servers:
            return True
            
        self.config.default_mcp_servers.remove(server_name)
        return self.save_config()
    
    def get_db_path(self) -> str:
        """Get database path.
        
        Returns:
            Database path.
        """
        # Expand user directory if path starts with "~"
        db_path = self.config.db_path
        if db_path.startswith("~"):
            db_path = os.path.expanduser(db_path)
            
        return db_path

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
