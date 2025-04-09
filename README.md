# KubeMindNexus

A comprehensive Kubernetes clusters management MCP client with LLM-powered capabilities.

## Features

- **Kubernetes Cluster Management**: Register, monitor, and manage multiple Kubernetes clusters
- **LLM-Powered Interaction**: Chat with your clusters using natural language through multiple LLM providers
- **REST API**: Comprehensive API for integration with other tools
- **Streamlit UI**: User-friendly interface for cluster management and chatting
- **MCP Integration**: Supports multiple MCP servers per cluster
- **SQLite Storage**: Persistent storage for clusters, chat history, and settings
- **Monitoring**: Real-time health checks and performance monitoring with LLM-powered insights

## Supported LLM Providers

- Ollama
- Deepseek
- OpenAI
- OpenRouter
- Extensible framework for adding additional providers

## Installation

### Using uv

```bash
uv venv
uv pip install -e .
```

### From source

```bash
git clone https://github.com/yourusername/KubeMindNexus.git
cd KubeMindNexus
pip install -e .
```

## Usage

### Starting the application

#### Command Line Interface

```bash
# Start both REST API and Streamlit UI
kubemindnexus

# Start only the REST API
kubemindnexus api

# Start only the Streamlit UI (requires API to be running)
kubemindnexus ui
```

#### Running from Python

You can also run components directly from Python files:

```python
# Run the API server
from kubemindnexus.api.app import main
main()  # Starts the API server using the configuration

# Run the UI server
from kubemindnexus.ui.app import main
main()  # Starts the Streamlit UI using the configuration

# Run both API and UI together
from kubemindnexus.__main__ import main
main()  # Starts both API and UI servers
```

If you need to customize settings before starting the servers:

```python
# API Server with custom settings
from kubemindnexus.api.app import api_server
from kubemindnexus.config.config import get_config

config = get_config()
# Modify configuration if needed
config.config.api.host = "0.0.0.0"  # Allow external connections
config.config.api.port = 9000       # Use a different port
config.save_config()                # Save changes

# Start the API server with custom host/port
api_server.start(host="0.0.0.0", port=9000)

# UI Server (runs with Streamlit)
from kubemindnexus.ui.app import ui
ui.run()  # Streamlit handling is automatic
```

### Configuration

KubeMindNexus uses a flexible configuration system that can be managed via:

1. Environment variables
2. Configuration file (JSON)
3. UI settings panel
4. Python API

#### Environment Variables

Create a `.env` file in your working directory:

```
# LLM API Keys
OPENAI_API_KEY=your_api_key
OPENROUTER_API_KEY=your_api_key
DEEPSEEK_API_KEY=your_api_key
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_LLM_PROVIDER=ollama

# API Settings
KUBEMINDNEXUS_API_HOST=127.0.0.1
KUBEMINDNEXUS_API_PORT=8000

# UI Settings
KUBEMINDNEXUS_UI_HOST=127.0.0.1
KUBEMINDNEXUS_UI_PORT=8501

# MCP Settings
KUBEMINDNEXUS_MAX_MCP_RETRIES=3
KUBEMINDNEXUS_MCP_RETRY_DELAY=1.0

# Logger Settings
KUBEMINDNEXUS_LOG_LEVEL=INFO

# ReAct Settings
KUBEMINDNEXUS_REACT_MAX_ITERATIONS=5

# Data Directory
KUBEMINDNEXUS_DATA_DIR=~/.kubemindnexus
```

#### Configuration File

KubeMindNexus uses a JSON configuration file located at `~/.kubemindnexus/config.json` by default. You can specify a different location using the `--config` command-line option.

The configuration file follows this structure:

```json
{
  "clusters": {
    "cluster-name": {
      "name": "cluster-name",
      "ip": "192.168.1.100",
      "port": 8000,
      "description": "Kubernetes cluster description",
      "servers": {
        "server-name": {
          "type": "sse",
          "name": "server-name",
          "url": "http://192.168.1.100:8000/sse",
          "command": null,
          "args": [],
          "env": {}
        }
      }
    }
  },
  "local_servers": {
    "local-server": {
      "type": "stdio",
      "name": "local-server",
      "command": "python",
      "args": ["-m", "server_module"],
      "env": {
        "ENV_VAR": "value"
      },
      "url": null
    }
  },
  "default_cluster": "cluster-name",
  "api": {
    "host": "127.0.0.1",
    "port": 8000
  },
  "ui": {
    "host": "127.0.0.1",
    "port": 8501,
    "title": "KubeMindNexus - K8s Cluster Management"
  },
  "llm": {
    "provider": "ollama",
    "openai_api_key": "",
    "openai_model": "gpt-4o",
    "deepseek_api_key": "",
    "openrouter_api_key": "",
    "ollama_base_url": "http://localhost:11434",
    "other_params": {
      "temperature": 0.7,
      "max_tokens": 4096
    }
  },
  "mcp": {
    "max_retries": 3,
    "retry_delay": 1.0
  },
  "logger": {
    "level": "INFO",
    "file": "~/.kubemindnexus/kubemindnexus.log"
  },
  "react": {
    "max_iterations": 5
  }
}
```

A sample configuration file is included at `config.json.example`.

#### Python API

You can programmatically access and modify the configuration:

```python
from kubemindnexus.config import get_config

# Get configuration instance
config = get_config()

# Access configuration properties
api_host = config.get_api_host()
api_port = config.get_api_port()

# Add a cluster
config.add_cluster(
    cluster_name="my-cluster",
    ip="192.168.1.100",
    port=8000,
    description="My Kubernetes cluster"
)

# Add a server to a cluster
config.add_server_to_cluster(
    cluster_name="my-cluster",
    server_name="k8s-server",
    server_type="sse",
    url="http://192.168.1.100:8000/sse"
)

# Add a local server
config.add_local_server(
    server_name="local-server",
    server_type="stdio",
    command="python",
    args=["-m", "server_module"],
    env={"ENV_VAR": "value"}
)

# Set default cluster
config.set_default_cluster("my-cluster")

# Set default LLM provider
config.set_default_llm_provider("openai")

# Save configuration to file
config.save_config()

# Reload configuration from file
config.reload()
```

## Database

KubeMindNexus uses SQLite for persistent storage. The database is automatically created and initialized when you first run the application, but you can also manually initialize it using the provided SQL file.

### Database Initialization

#### Automatic Initialization

By default, the database is automatically initialized when the application starts:

```python
from kubemindnexus.database.client import DatabaseClient

# Initialize the database with default settings
db_client = DatabaseClient()
```

#### Manual Initialization

You can manually initialize the database using the provided SQL file:

```bash
# Create an empty database file
touch ~/.kubemindnexus/kubemindnexus.db

# Initialize the database with the SQL file
sqlite3 ~/.kubemindnexus/kubemindnexus.db < initialize_db.sql
```

#### Custom Database Path

You can specify a custom database path:

```python
from kubemindnexus.database.client import DatabaseClient

# Initialize the database with a custom path
db_client = DatabaseClient(db_path="/path/to/your/database.db")
```

### Database Schema

The database includes the following tables:

- **clusters**: Kubernetes cluster configurations
- **mcp_servers**: MCP server configurations
- **chat_history**: History of LLM interactions
- **health_checks**: Cluster health check results
- **performance_metrics**: Cluster performance metrics
- **config**: Application configuration key-value pairs

## Architecture

KubeMindNexus is built with a modular architecture:

- **API Module**: FastAPI REST API for cluster management and chat
- **UI Module**: Streamlit-based user interface
- **Database Module**: SQLite database for persistent storage
- **LLM Module**: Abstraction layer for different LLM providers
- **MCP Module**: MCP server and cluster management
- **Config Module**: Configuration management

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
