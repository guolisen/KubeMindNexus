# KubeMindNexus API Architecture

This document provides an overview of the KubeMindNexus API architecture, explaining how the different components interact with each other.

## Architecture Overview

```mermaid
graph TD
    UI[Streamlit UI] --> |HTTP Requests| API[FastAPI REST API]
    API --> |Queries| DB[SQLite Database]
    API --> |LLM Requests| LLM[LLM Integration]
    API --> |MCP Commands| MCP[MCP Hub]
    MCP --> |Connect/Disconnect| S1[MCP Server 1]
    MCP --> |Connect/Disconnect| S2[MCP Server 2]
    MCP --> |Connect/Disconnect| S3[MCP Server N]
    LLM --> |ReAct Loop| REACT[ReAct Engine]
    REACT --> |Tool Use| MCP
    
    subgraph External LLMs
        LLM --> |API Calls| OPENAI[OpenAI]
        LLM --> |API Calls| OLLAMA[Ollama]
        LLM --> |API Calls| DEEPSEEK[Deepseek]
        LLM --> |API Calls| OPENROUTER[OpenRouter]
    end
    
    subgraph Kubernetes Clusters
        S1 --> |K8s API| C1[Cluster 1]
        S2 --> |K8s API| C2[Cluster 2]
        S3 --> |K8s API| C3[Cluster N]
    end
```

## API Layer Components

1. **REST API (FastAPI)**: Exposes HTTP endpoints for managing clusters, MCP servers, and chat interactions.
2. **Database Layer**: Stores configuration, chat history, and relationships.
3. **MCP Hub**: Manages connections to MCP servers.
4. **LLM Integration**: Provides a unified interface to multiple LLM providers.
5. **ReAct Engine**: Implements the Reasoning and Acting pattern for LLMs.

## Endpoint Categories

The API endpoints are organized into the following categories:

### Cluster Management
Endpoints for registering, configuring, and managing Kubernetes clusters.

### MCP Server Management
Endpoints for configuring and connecting to MCP servers.

### MCP Server Status
Endpoints for checking the connection status of MCP servers.

### MCP Server Tools and Resources
Endpoints for accessing tools and resources provided by connected MCP servers.

### Cluster Metrics
Endpoints for retrieving performance, health, and storage metrics from clusters.

### Kubernetes Resources
Endpoints for accessing Kubernetes resources (nodes, pods, services, PVs).

### LLM Configuration
Endpoints for configuring LLM providers and models.

### Chat Integration
Endpoints for sending messages and retrieving chat history.

## Data Flow

```mermaid
sequenceDiagram
    participant UI as Streamlit UI
    participant API as REST API
    participant DB as Database
    participant MCP as MCP Hub
    participant LLM as LLM Service
    participant K8s as Kubernetes Cluster

    UI->>API: HTTP Request (e.g., /api/clusters)
    API->>DB: Query Data
    DB->>API: Return Data
    API->>UI: HTTP Response (JSON)

    UI->>API: Send Chat Message
    API->>LLM: Process Message
    LLM->>MCP: Execute Tool (e.g., list pods)
    MCP->>K8s: Execute K8s API Call
    K8s->>MCP: Return Results
    MCP->>LLM: Return Tool Results
    LLM->>API: Generate Response
    API->>UI: Return Chat Response
```

## REST API Structure

The API follows RESTful design principles with consistent endpoint patterns:

- **GET** for retrieving data
- **POST** for creating resources
- **PUT** for updating resources
- **DELETE** for removing resources

### Example: Cluster Resource Lifecycle

```mermaid
graph LR
    A[GET /api/clusters] -->|List all clusters| B[POST /api/clusters]
    B -->|Create new cluster| C[GET /api/clusters/:id]
    C -->|Get cluster details| D[PUT /api/clusters/:id]
    D -->|Update cluster| E[DELETE /api/clusters/:id]
    E -->|Delete cluster| A
```

## MCP Server Connection Flow

```mermaid
sequenceDiagram
    participant UI as UI
    participant API as API
    participant MCP as MCP Hub
    participant Server as MCP Server
    participant K8s as Kubernetes Cluster

    UI->>API: POST /api/mcp-servers/:id/connect
    API->>MCP: Connect to server
    MCP->>Server: Establish connection
    Server->>K8s: Connect to K8s API
    K8s->>Server: Authentication success
    Server->>MCP: Connection established
    MCP->>API: Return connection status
    API->>UI: Connection status response
```

## Metrics Data Flow

```mermaid
sequenceDiagram
    participant UI as UI
    participant API as API
    participant MCP as MCP Hub
    participant Server as MCP Server
    participant K8s as Kubernetes Cluster

    UI->>API: GET /api/clusters/:id/metrics/performance
    API->>MCP: Request metrics
    MCP->>Server: Execute metrics tool
    Server->>K8s: Query metrics server
    K8s->>Server: Return metrics data
    Server->>MCP: Process metrics
    MCP->>API: Format metrics response
    API->>UI: Return formatted metrics
    UI->>UI: Render charts and visualizations
```

## API Authentication and Security

Authentication is currently not implemented but planned for future versions. The typical authentication flow will be:

```mermaid
sequenceDiagram
    participant User as User
    participant UI as UI
    participant API as API
    participant Auth as Auth Service

    User->>UI: Access UI
    UI->>API: Request authentication
    API->>Auth: Validate credentials
    Auth->>API: Auth token
    API->>UI: Return auth token
    UI->>User: Authentication complete
```

## Error Handling

The API implements consistent error handling with appropriate HTTP status codes:

- **400**: Bad Request - Invalid parameters
- **404**: Not Found - Resource doesn't exist
- **500**: Internal Server Error - Server-side errors

Each error response follows a standard format:

```json
{
  "detail": "Error message describing the issue"
}
```

## API Versioning Strategy

While versioning is not currently implemented, future API versions will follow this pattern:

```
/api/v1/clusters
/api/v2/clusters
```

This allows for backward compatibility while introducing new features.
