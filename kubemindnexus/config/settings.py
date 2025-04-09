"""Settings module for KubeMindNexus."""
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.getenv("KUBEMINDNEXUS_DATA_DIR", os.path.join(str(Path.home()), ".kubemindnexus")))
SQLITE_DB_PATH = os.path.join(DATA_DIR, "kubemindnexus.db")
DEFAULT_CONFIG_PATH = os.path.join(DATA_DIR, "config.json")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Logging variables used by logger.py
LOG_LEVEL = os.getenv("KUBEMINDNEXUS_LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(DATA_DIR, "kubemindnexus.log")

# Ensure logs directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


class ServerConfigModel(BaseModel):
    """Model for MCP server configuration."""
    type: str = Field(..., description="Server type, either 'stdio' or 'sse'")
    name: Optional[str] = Field(None, description="Server name")
    command: Optional[str] = Field(None, description="Command to run for stdio servers")
    args: List[str] = Field(default_factory=list, description="Arguments for the command")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    url: Optional[str] = Field(None, description="URL for SSE servers")


class ClusterConfigModel(BaseModel):
    """Model for cluster configuration."""
    name: str = Field(..., description="Cluster name")
    ip: str = Field(..., description="Cluster IP address")
    port: int = Field(..., description="Cluster port")
    description: Optional[str] = Field(None, description="Cluster description")
    servers: Dict[str, ServerConfigModel] = Field(
        default_factory=dict, description="MCP servers for this cluster"
    )


class APIConfigModel(BaseModel):
    """Model for API configuration."""
    host: str = Field(
        default=os.getenv("KUBEMINDNEXUS_API_HOST", "127.0.0.1"),
        description="API host address"
    )
    port: int = Field(
        default=int(os.getenv("KUBEMINDNEXUS_API_PORT", "8000")),
        description="API port"
    )


class UIConfigModel(BaseModel):
    """Model for UI configuration."""
    host: str = Field(
        default=os.getenv("KUBEMINDNEXUS_UI_HOST", "127.0.0.1"),
        description="UI host address"
    )
    port: int = Field(
        default=int(os.getenv("KUBEMINDNEXUS_UI_PORT", "8501")),
        description="UI port"
    )
    title: str = Field(
        default="KubeMindNexus - K8s Cluster Management",
        description="UI title"
    )


class LLMConfigModel(BaseModel):
    """Model for LLM configuration."""
    provider: str = Field(
        default=os.getenv("DEFAULT_LLM_PROVIDER", "ollama"),
        description="Default LLM provider"
    )
    openai_api_key: Optional[str] = Field(
        default=os.getenv("OPENAI_API_KEY", ""),
        description="OpenAI API key"
    )
    openai_model: Optional[str] = Field(
        default=os.getenv("OPENAI_MODEL", "gpt-4o"),
        description="OpenAI model"
    )
    deepseek_api_key: Optional[str] = Field(
        default=os.getenv("DEEPSEEK_API_KEY", ""),
        description="DeepSeek API key"
    )
    openrouter_api_key: Optional[str] = Field(
        default=os.getenv("OPENROUTER_API_KEY", ""),
        description="OpenRouter API key"
    )
    ollama_base_url: Optional[str] = Field(
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        description="Ollama base URL"
    )
    other_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional LLM parameters"
    )


class MCPConfigModel(BaseModel):
    """Model for MCP configuration."""
    max_retries: int = Field(
        default=int(os.getenv("KUBEMINDNEXUS_MAX_MCP_RETRIES", "3")),
        description="Maximum MCP connection retries"
    )
    retry_delay: float = Field(
        default=float(os.getenv("KUBEMINDNEXUS_MCP_RETRY_DELAY", "1.0")),
        description="Delay between MCP connection retries"
    )

# MCP retry constants
MAX_MCP_RETRIES = int(os.getenv("KUBEMINDNEXUS_MAX_MCP_RETRIES", "3"))
MCP_RETRY_DELAY = float(os.getenv("KUBEMINDNEXUS_MCP_RETRY_DELAY", "1.0"))


class LoggerConfigModel(BaseModel):
    """Model for logger configuration."""
    level: str = Field(
        default=os.getenv("KUBEMINDNEXUS_LOG_LEVEL", "INFO"),
        description="Logging level"
    )
    file: str = Field(
        default=os.path.join(DATA_DIR, "kubemindnexus.log"),
        description="Log file path"
    )


class ReActConfigModel(BaseModel):
    """Model for ReAct configuration."""
    max_iterations: int = Field(
        default=int(os.getenv("KUBEMINDNEXUS_REACT_MAX_ITERATIONS", "5")),
        description="Maximum ReAct iterations"
    )

# ReAct constants
REACT_MAX_ITERATIONS = int(os.getenv("KUBEMINDNEXUS_REACT_MAX_ITERATIONS", "5"))


class ConfigModel(BaseModel):
    """Model for application configuration."""
    # Cluster and server configuration
    clusters: Dict[str, ClusterConfigModel] = Field(
        default_factory=dict, 
        description="Kubernetes clusters configuration"
    )
    local_servers: Dict[str, ServerConfigModel] = Field(
        default_factory=dict, 
        description="Local MCP servers configuration"
    )
    default_cluster: Optional[str] = Field(
        None, 
        description="Default cluster name"
    )
    
    # Grouped configuration models
    api: APIConfigModel = Field(
        default_factory=APIConfigModel,
        description="API configuration"
    )
    ui: UIConfigModel = Field(
        default_factory=UIConfigModel,
        description="UI configuration"
    )
    llm: LLMConfigModel = Field(
        default_factory=LLMConfigModel,
        description="LLM configuration"
    )
    mcp: MCPConfigModel = Field(
        default_factory=MCPConfigModel,
        description="MCP configuration"
    )
    logger: LoggerConfigModel = Field(
        default_factory=LoggerConfigModel,
        description="Logger configuration"
    )
    react: ReActConfigModel = Field(
        default_factory=ReActConfigModel,
        description="ReAct configuration"
    )
