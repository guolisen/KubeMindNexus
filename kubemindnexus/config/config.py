"""Configuration module for KubeMindNexus."""
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .settings import (
    ConfigModel, DEFAULT_CONFIG_PATH, DATA_DIR,
    ServerConfigModel, ClusterConfigModel,
    APIConfigModel, UIConfigModel, LLMConfigModel,
    MCPConfigModel, LoggerConfigModel, ReActConfigModel
)

logger = logging.getLogger(__name__)

# Global configuration instance
_config_instance = None


def get_config(config_path: Optional[str] = None) -> 'Configuration':
    """Get the global configuration instance.
    
    If the global instance has not been initialized, this will initialize it.
    
    Args:
        config_path: Optional path to the configuration file to use for initialization.
            This is only used if the instance has not been initialized yet.
            
    Returns:
        The global configuration instance.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Configuration(config_path)
    return _config_instance


def initialize_config(config_path: Optional[str] = None) -> 'Configuration':
    """Initialize the global configuration instance.
    
    This will create a new configuration instance even if one already exists.
    
    Args:
        config_path: Optional path to the configuration file.
            
    Returns:
        The newly created global configuration instance.
    """
    global _config_instance
    _config_instance = Configuration(config_path)
    return _config_instance


class Configuration:
    """Configuration manager for KubeMindNexus.
    
    This class handles loading, saving, and managing configuration settings.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize Configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses the default path.
        """
        self._config_path = config_path or DEFAULT_CONFIG_PATH
        self._config = self._load_or_create_config()
    
    @property
    def config_path(self) -> str:
        """Get the configuration file path."""
        return self._config_path
    
    @property
    def config(self) -> ConfigModel:
        """Get the current configuration."""
        return self._config
    
    def _load_or_create_config(self) -> ConfigModel:
        """Load configuration from file or create default if it doesn't exist."""
        path = Path(self._config_path)
        if path.exists():
            try:
                with open(path, "r") as f:
                    config_data = json.load(f)
                return ConfigModel.model_validate(config_data)
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                logger.info("Loading default configuration")
                return self._create_default_config()
        else:
            logger.info(f"Configuration file not found. Creating default at {self._config_path}")
            default_config = self._create_default_config()
            self.save_config(default_config)
            return default_config
    
    def _create_default_config(self) -> ConfigModel:
        """Create default configuration."""
        return ConfigModel()
    
    def save_config(self, config: Union[ConfigModel, Dict[str, Any]] = None) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration to save, either as a ConfigModel object
                   or a dictionary. If None, uses the current configuration.
        """
        if config is None:
            config_to_save = self._config
        elif isinstance(config, ConfigModel):
            config_to_save = config
            # Update the current config
            self._config = config
        else:
            # Handle dictionary input
            self._config = ConfigModel.model_validate(config)
            config_to_save = self._config
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
        
        try:
            with open(self._config_path, "w") as f:
                json.dump(config_to_save.model_dump(), f, indent=2)
            
            logger.info(f"Configuration saved to {self._config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def reload(self) -> None:
        """Reload configuration from file."""
        path = Path(self._config_path)
        if path.exists():
            try:
                with open(path, "r") as f:
                    config_data = json.load(f)
                    self._config = ConfigModel.model_validate(config_data)
                logger.info(f"Configuration reloaded from {self._config_path}")
            except Exception as e:
                logger.error(f"Error reloading configuration: {e}")
                raise
        else:
            raise FileNotFoundError(f"Configuration file not found: {self._config_path}")
    
    # Cluster management methods
    def add_cluster(self, cluster_name: str, ip: str, port: int, 
                   description: Optional[str] = None) -> bool:
        """Add a new cluster to the configuration."""
        if cluster_name in self._config.clusters:
            logger.warning(f"Cluster {cluster_name} already exists. Updating.")
        
        try:
            self._config.clusters[cluster_name] = ClusterConfigModel(
                name=cluster_name,
                ip=ip,
                port=port,
                description=description,
                servers={}
            )
            
            self.save_config()
            logger.info(f"Cluster {cluster_name} added to configuration")
            return True
        
        except Exception as e:
            logger.error(f"Error adding cluster {cluster_name}: {e}")
            return False
    
    def remove_cluster(self, cluster_name: str) -> bool:
        """Remove a cluster from the configuration."""
        if cluster_name not in self._config.clusters:
            logger.warning(f"Cluster {cluster_name} not found")
            return False
        
        try:
            del self._config.clusters[cluster_name]
            
            # If the default cluster is being removed, clear it
            if self._config.default_cluster == cluster_name:
                self._config.default_cluster = None
            
            self.save_config()
            logger.info(f"Cluster {cluster_name} removed from configuration")
            return True
        
        except Exception as e:
            logger.error(f"Error removing cluster {cluster_name}: {e}")
            return False
    
    # Server management methods
    def add_server_to_cluster(self, cluster_name: str, server_name: str, 
                             server_type: str, command: Optional[str] = None,
                             args: Optional[list] = None, 
                             env: Optional[Dict[str, str]] = None,
                             url: Optional[str] = None) -> bool:
        """Add a server to a cluster."""
        if cluster_name not in self._config.clusters:
            logger.error(f"Cluster {cluster_name} not found")
            return False
        
        try:
            self._config.clusters[cluster_name].servers[server_name] = ServerConfigModel(
                name=server_name,
                type=server_type,
                command=command,
                args=args or [],
                env=env or {},
                url=url
            )
            
            self.save_config()
            logger.info(f"Server {server_name} added to cluster {cluster_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error adding server {server_name} to cluster {cluster_name}: {e}")
            return False
    
    def add_local_server(self, server_name: str, server_type: str,
                        command: Optional[str] = None, args: Optional[list] = None,
                        env: Optional[Dict[str, str]] = None,
                        url: Optional[str] = None) -> bool:
        """Add a local server to the configuration."""
        try:
            self._config.local_servers[server_name] = ServerConfigModel(
                name=server_name,
                type=server_type,
                command=command,
                args=args or [],
                env=env or {},
                url=url
            )
            
            self.save_config()
            logger.info(f"Local server {server_name} added to configuration")
            return True
        
        except Exception as e:
            logger.error(f"Error adding local server {server_name}: {e}")
            return False
    
    def remove_server_from_cluster(self, cluster_name: str, server_name: str) -> bool:
        """Remove a server from a cluster."""
        if cluster_name not in self._config.clusters:
            logger.error(f"Cluster {cluster_name} not found")
            return False
        
        if server_name not in self._config.clusters[cluster_name].servers:
            logger.warning(f"Server {server_name} not found in cluster {cluster_name}")
            return False
        
        try:
            del self._config.clusters[cluster_name].servers[server_name]
            
            self.save_config()
            logger.info(f"Server {server_name} removed from cluster {cluster_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error removing server {server_name} from cluster {cluster_name}: {e}")
            return False
    
    def remove_local_server(self, server_name: str) -> bool:
        """Remove a local server from the configuration."""
        if server_name not in self._config.local_servers:
            logger.warning(f"Local server {server_name} not found")
            return False
        
        try:
            del self._config.local_servers[server_name]
            
            self.save_config()
            logger.info(f"Local server {server_name} removed from configuration")
            return True
        
        except Exception as e:
            logger.error(f"Error removing local server {server_name}: {e}")
            return False
    
    def set_default_cluster(self, cluster_name: Optional[str]) -> bool:
        """Set the default cluster."""
        if cluster_name is not None and cluster_name not in self._config.clusters:
            logger.error(f"Cluster {cluster_name} not found")
            return False
        
        try:
            self._config.default_cluster = cluster_name
            
            self.save_config()
            logger.info(f"Default cluster set to {cluster_name or 'None'}")
            return True
        
        except Exception as e:
            logger.error(f"Error setting default cluster: {e}")
            return False
    
    # Helper methods to get configuration values
    
    # API settings
    def get_api_host(self) -> str:
        """Get the API host address."""
        return self._config.api.host
    
    def get_api_port(self) -> int:
        """Get the API port."""
        return self._config.api.port
    
    def get_api_base_url(self) -> str:
        """Get the API base URL."""
        return f"http://{self._config.api.host}:{self._config.api.port}"
    
    # UI settings
    def get_ui_host(self) -> str:
        """Get the UI host address."""
        return self._config.ui.host
    
    def get_ui_port(self) -> int:
        """Get the UI port."""
        return self._config.ui.port
    
    def get_ui_title(self) -> str:
        """Get the UI title."""
        return self._config.ui.title
    
    # LLM settings
    def get_openai_api_key(self) -> str:
        """Get the OpenAI API key."""
        return self._config.llm.openai_api_key
    
    def get_openai_model(self) -> str:
        """Get the OpenAI model."""
        return self._config.llm.openai_model
    
    def get_deepseek_api_key(self) -> str:
        """Get the DeepSeek API key."""
        return self._config.llm.deepseek_api_key
    
    def get_openrouter_api_key(self) -> str:
        """Get the OpenRouter API key."""
        return self._config.llm.openrouter_api_key
    
    def get_ollama_base_url(self) -> str:
        """Get the Ollama base URL."""
        return self._config.llm.ollama_base_url
    
    def get_default_llm_provider(self) -> str:
        """Get the default LLM provider."""
        return self._config.llm.provider
    
    def set_default_llm_provider(self, provider: str) -> bool:
        """Set the default LLM provider."""
        try:
            self._config.llm.provider = provider
            
            self.save_config()
            logger.info(f"Default LLM provider set to {provider}")
            return True
        
        except Exception as e:
            logger.error(f"Error setting default LLM provider: {e}")
            return False
    
    # MCP settings
    def get_max_mcp_retries(self) -> int:
        """Get the maximum MCP connection retries."""
        return self._config.mcp.max_retries
    
    def get_mcp_retry_delay(self) -> float:
        """Get the delay between MCP connection retries."""
        return self._config.mcp.retry_delay
    
    # Logger settings
    def get_log_level(self) -> str:
        """Get the logging level."""
        return self._config.logger.level
    
    # ReAct settings
    def get_react_max_iterations(self) -> int:
        """Get the maximum ReAct iterations."""
        return self._config.react.max_iterations
    
    # Cluster and server information
    def get_all_clusters(self) -> Dict[str, ClusterConfigModel]:
        """Get all clusters in the configuration."""
        return self._config.clusters
    
    def get_all_local_servers(self) -> Dict[str, ServerConfigModel]:
        """Get all local servers in the configuration."""
        return self._config.local_servers
    
    def get_cluster(self, cluster_name: str) -> Optional[ClusterConfigModel]:
        """Get a specific cluster configuration."""
        return self._config.clusters.get(cluster_name)
    
    def get_local_server(self, server_name: str) -> Optional[ServerConfigModel]:
        """Get a specific local server configuration."""
        return self._config.local_servers.get(server_name)
