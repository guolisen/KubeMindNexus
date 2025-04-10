"""Constants for KubeMindNexus."""

from enum import Enum, auto


class ServerType(str, Enum):
    """MCP server types."""
    
    STDIO = "stdio"
    SSE = "sse"


class LLMProvider(str, Enum):
    """LLM providers."""
    
    OPENAI = "openai"
    OLLAMA = "ollama"
    DEEPSEEK = "deepseek"
    OPENROUTER = "openrouter"


# ReAct loop constants
REACT_MAX_ITERATIONS = 5
REACT_SAFETY_TIMEOUT = 60  # seconds

# Default system prompt template
DEFAULT_SYSTEM_PROMPT = """
You are KubeMindNexus, an AI assistant specialized in Kubernetes cluster management.
You have access to the following tools:

{available_tools}

Current cluster context: {cluster_context}

Always respond in a helpful, concise manner focused on Kubernetes management tasks.
"""

# Default database path
DEFAULT_DB_PATH = "~/.config/kubemindnexus/kubemindnexus.db"

# Default UI settings
DEFAULT_UI_PORT = 8501

# Default API settings
DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000

# Console colors for logging
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
