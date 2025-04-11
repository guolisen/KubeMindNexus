# KubeMindNexus API Reference

This document provides a comprehensive reference for the KubeMindNexus REST API endpoints.

## Base URL

All API endpoints are relative to the base URL: `http://localhost:8000`

## Authentication

Authentication is not currently implemented. Future versions may include authentication mechanisms.

## Common Response Codes

- `200 OK`: Request succeeded
- `201 Created`: Resource was successfully created
- `204 No Content`: Request succeeded with no response body
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

## Endpoints

- [Cluster Management](#cluster-management)
- [MCP Server Management](#mcp-server-management)
- [MCP Server Status](#mcp-server-status)
- [MCP Server Tools and Resources](#mcp-server-tools-and-resources)
- [Cluster Metrics](#cluster-metrics)
- [Kubernetes Resources](#kubernetes-resources)
- [LLM Configuration](#llm-configuration)
- [Chat Integration](#chat-integration)

---

## Cluster Management

### List All Clusters

```
GET /api/clusters
```

Returns a list of all registered Kubernetes clusters.

**Response**

```json
[
  {
    "id": 1,
    "name": "production-cluster",
    "ip": "192.168.1.100",
    "port": 6443,
    "description": "Production Kubernetes cluster",
    "created_at": "2025-04-10T12:34:56Z"
  },
  {
    "id": 2,
    "name": "development-cluster",
    "ip": "192.168.1.101",
    "port": 6443,
    "description": "Development Kubernetes cluster",
    "created_at": "2025-04-10T14:23:45Z"
  }
]
```

### Get Cluster by ID

```
GET /api/clusters/{cluster_id}
```

Returns details for a specific cluster.

**Parameters**

- `cluster_id`: ID of the cluster (integer)

**Response**

```json
{
  "id": 1,
  "name": "production-cluster",
  "ip": "192.168.1.100",
  "port": 6443,
  "description": "Production Kubernetes cluster",
  "created_at": "2025-04-10T12:34:56Z"
}
```

### Create Cluster

```
POST /api/clusters
```

Creates a new Kubernetes cluster configuration.

**Request Body**

```json
{
  "name": "staging-cluster",
  "ip": "192.168.1.102",
  "port": 6443,
  "description": "Staging Kubernetes cluster"
}
```

**Response**

```json
{
  "id": 3,
  "name": "staging-cluster",
  "ip": "192.168.1.102",
  "port": 6443,
  "description": "Staging Kubernetes cluster",
  "created_at": "2025-04-11T09:12:34Z"
}
```

### Update Cluster

```
PUT /api/clusters/{cluster_id}
```

Updates an existing cluster configuration.

**Parameters**

- `cluster_id`: ID of the cluster to update (integer)

**Request Body**

```json
{
  "name": "updated-cluster-name",
  "ip": "192.168.1.105",
  "port": 6443,
  "description": "Updated description"
}
```

All fields are optional. Only provided fields will be updated.

**Response**

```json
{
  "id": 1,
  "name": "updated-cluster-name",
  "ip": "192.168.1.105",
  "port": 6443,
  "description": "Updated description",
  "created_at": "2025-04-10T12:34:56Z"
}
```

### Delete Cluster

```
DELETE /api/clusters/{cluster_id}
```

Deletes a cluster and disconnects all associated MCP servers.

**Parameters**

- `cluster_id`: ID of the cluster to delete (integer)

**Response**

```
204 No Content
```

---

## MCP Server Management

### List All MCP Servers

```
GET /api/mcp-servers
```

Returns a list of all MCP servers.

**Query Parameters**

- `cluster_id` (optional): Filter servers by cluster ID

**Response**

```json
[
  {
    "id": 1,
    "name": "weather-mcp",
    "type": "stdio",
    "command": "node",
    "args": ["/path/to/server.js"],
    "url": null,
    "cluster_id": 1,
    "is_local": true,
    "is_default": false,
    "env": {
      "API_KEY": "xxxxx"
    },
    "created_at": "2025-04-10T15:00:00Z"
  },
  {
    "id": 2,
    "name": "github-mcp",
    "type": "sse",
    "command": null,
    "args": null,
    "url": "https://github-mcp.example.com",
    "cluster_id": null,
    "is_local": false,
    "is_default": true,
    "env": {},
    "created_at": "2025-04-10T16:30:00Z"
  }
]
```

### Get MCP Server by ID

```
GET /api/mcp-servers/{server_id}
```

Returns details for a specific MCP server.

**Parameters**

- `server_id`: ID of the MCP server (integer)

**Response**

```json
{
  "id": 1,
  "name": "weather-mcp",
  "type": "stdio",
  "command": "node",
  "args": ["/path/to/server.js"],
  "url": null,
  "cluster_id": 1,
  "is_local": true,
  "is_default": false,
  "env": {
    "API_KEY": "xxxxx"
  },
  "created_at": "2025-04-10T15:00:00Z"
}
```

### Create MCP Server

```
POST /api/mcp-servers
```

Creates a new MCP server configuration.

**Request Body**

```json
{
  "name": "new-mcp-server",
  "type": "stdio",
  "command": "python",
  "args": ["-m", "my_mcp_server"],
  "cluster_id": 1,
  "is_local": true,
  "is_default": false,
  "env": {
    "DEBUG": "true"
  }
}
```

**Response**

```json
{
  "id": 3,
  "name": "new-mcp-server",
  "type": "stdio",
  "command": "python",
  "args": ["-m", "my_mcp_server"],
  "url": null,
  "cluster_id": 1,
  "is_local": true,
  "is_default": false,
  "env": {
    "DEBUG": "true"
  },
  "created_at": "2025-04-11T10:15:00Z"
}
```

### Update MCP Server

```
PUT /api/mcp-servers/{server_id}
```

Updates an existing MCP server configuration.

**Parameters**

- `server_id`: ID of the MCP server to update (integer)

**Request Body**

```json
{
  "name": "updated-mcp-name",
  "env": {
    "DEBUG": "false",
    "LOG_LEVEL": "info"
  }
}
```

All fields are optional. Only provided fields will be updated.

**Response**

```json
{
  "id": 1,
  "name": "updated-mcp-name",
  "type": "stdio",
  "command": "node",
  "args": ["/path/to/server.js"],
  "url": null,
  "cluster_id": 1,
  "is_local": true,
  "is_default": false,
  "env": {
    "API_KEY": "xxxxx",
    "DEBUG": "false",
    "LOG_LEVEL": "info"
  },
  "created_at": "2025-04-10T15:00:00Z"
}
```

### Delete MCP Server

```
DELETE /api/mcp-servers/{server_id}
```

Deletes an MCP server configuration.

**Parameters**

- `server_id`: ID of the MCP server to delete (integer)

**Response**

```
204 No Content
```

### Connect to MCP Server

```
POST /api/mcp-servers/{server_id}/connect
```

Connects to an MCP server.

**Parameters**

- `server_id`: ID of the MCP server to connect to (integer)

**Response**

```json
{
  "message": "Connected to MCP server server-name"
}
```

### Disconnect from MCP Server

```
POST /api/mcp-servers/{server_id}/disconnect
```

Disconnects from an MCP server.

**Parameters**

- `server_id`: ID of the MCP server to disconnect from (integer)

**Response**

```json
{
  "message": "Disconnected from MCP server server-name"
}
```

---

## MCP Server Status

### Get All MCP Servers Status

```
GET /api/mcp-servers/status
```

Returns connection status for all MCP servers.

**Response**

```json
[
  {
    "id": 1,
    "name": "weather-mcp",
    "is_connected": true
  },
  {
    "id": 2,
    "name": "github-mcp",
    "is_connected": false
  }
]
```

### Get MCP Server Status

```
GET /api/mcp-servers/{server_id}/status
```

Returns connection status for a specific MCP server.

**Parameters**

- `server_id`: ID of the MCP server (integer)

**Response**

```json
{
  "id": 1,
  "name": "weather-mcp",
  "is_connected": true
}
```

---

## MCP Server Tools and Resources

### Get MCP Server Tools

```
GET /api/mcp-servers/{server_id}/tools
```

Returns available tools for a specific MCP server.

**Parameters**

- `server_id`: ID of the MCP server (integer)

**Response**

```json
[
  {
    "name": "weather_forecast",
    "description": "Get weather forecast for a location",
    "inputSchema": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "City name or coordinates"
        },
        "days": {
          "type": "integer",
          "description": "Number of days to forecast",
          "minimum": 1,
          "maximum": 10
        }
      },
      "required": ["location"]
    }
  }
]
```

### Get MCP Server Resources

```
GET /api/mcp-servers/{server_id}/resources
```

Returns available resources for a specific MCP server.

**Parameters**

- `server_id`: ID of the MCP server (integer)

**Response**

```json
[
  {
    "uri": "weather://San Francisco/current",
    "name": "Current weather in San Francisco",
    "mimeType": "application/json",
    "description": "Real-time weather data for San Francisco"
  }
]
```

---

## Cluster Metrics

### Get Cluster Performance Metrics

```
GET /api/clusters/{cluster_id}/metrics/performance
```

Returns performance metrics for a specific cluster.

**Parameters**

- `cluster_id`: ID of the cluster (integer)

**Response**

```json
{
  "cluster_id": 1,
  "cluster_name": "production-cluster",
  "timestamp": "2025-04-11T14:30:00",
  "cpu_usage": [
    {"time": "T-10", "usage": 45.2},
    {"time": "T-9", "usage": 52.8},
    {"time": "T-8", "usage": 38.1},
    {"time": "T-7", "usage": 42.5},
    {"time": "T-6", "usage": 56.3},
    {"time": "T-5", "usage": 61.7},
    {"time": "T-4", "usage": 55.0},
    {"time": "T-3", "usage": 48.9},
    {"time": "T-2", "usage": 52.4},
    {"time": "T-1", "usage": 47.6}
  ],
  "memory_usage": [
    {"time": "T-10", "usage_gb": 5.2},
    {"time": "T-9", "usage_gb": 5.5},
    {"time": "T-8", "usage_gb": 5.3},
    {"time": "T-7", "usage_gb": 4.9},
    {"time": "T-6", "usage_gb": 4.7},
    {"time": "T-5", "usage_gb": 4.8},
    {"time": "T-4", "usage_gb": 5.0},
    {"time": "T-3", "usage_gb": 5.4},
    {"time": "T-2", "usage_gb": 5.7},
    {"time": "T-1", "usage_gb": 5.5}
  ]
}
```

### Get Cluster Health Metrics

```
GET /api/clusters/{cluster_id}/metrics/health
```

Returns health metrics for a specific cluster.

**Parameters**

- `cluster_id`: ID of the cluster (integer)

**Response**

```json
{
  "cluster_id": 1,
  "cluster_name": "production-cluster",
  "timestamp": "2025-04-11T14:30:00",
  "node_status": "Healthy",
  "pod_health_percentage": 98.0,
  "services_count": {"Total": 15, "Healthy": 15},
  "pod_status": {
    "Running": 25,
    "Pending": 2,
    "Failed": 0,
    "Succeeded": 5,
    "Unknown": 0
  }
}
```

### Get Cluster Storage Metrics

```
GET /api/clusters/{cluster_id}/metrics/storage
```

Returns storage metrics for a specific cluster.

**Parameters**

- `cluster_id`: ID of the cluster (integer)

**Response**

```json
{
  "cluster_id": 1,
  "cluster_name": "production-cluster",
  "timestamp": "2025-04-11T14:30:00",
  "storage_usage": [
    {
      "pv_name": "pv-1",
      "capacity_gb": 100,
      "used_gb": 78,
      "available_gb": 22,
      "usage_percentage": 78.0
    },
    {
      "pv_name": "pv-2",
      "capacity_gb": 50,
      "used_gb": 32,
      "available_gb": 18,
      "usage_percentage": 64.0
    },
    {
      "pv_name": "pv-3",
      "capacity_gb": 200,
      "used_gb": 150,
      "available_gb": 50,
      "usage_percentage": 75.0
    }
  ]
}
```

---

## Kubernetes Resources

### Get Cluster Nodes

```
GET /api/clusters/{cluster_id}/nodes
```

Returns nodes for a specific cluster.

**Parameters**

- `cluster_id`: ID of the cluster (integer)

**Response**

```json
[
  {
    "name": "node-1",
    "status": "Ready",
    "role": "master",
    "cpu": "4",
    "memory": "16Gi",
    "kubernetes_version": "1.26.0",
    "created_at": "2024-04-01T12:00:00Z"
  },
  {
    "name": "node-2",
    "status": "Ready",
    "role": "worker",
    "cpu": "4",
    "memory": "16Gi",
    "kubernetes_version": "1.26.0",
    "created_at": "2024-04-01T12:00:00Z"
  },
  {
    "name": "node-3",
    "status": "Ready",
    "role": "worker",
    "cpu": "4",
    "memory": "16Gi",
    "kubernetes_version": "1.26.0",
    "created_at": "2024-04-01T12:00:00Z"
  }
]
```

### Get Cluster Pods

```
GET /api/clusters/{cluster_id}/pods
```

Returns pods for a specific cluster, optionally filtered by namespace.

**Parameters**

- `cluster_id`: ID of the cluster (integer)
- `namespace` (optional): Namespace to filter by (string)

**Response**

```json
[
  {
    "name": "pod-default-1",
    "namespace": "default",
    "status": "Running",
    "node": "node-1",
    "ip": "10.0.default.1",
    "created_at": "2024-04-01T12:00:00Z"
  },
  {
    "name": "pod-kube-system-1",
    "namespace": "kube-system",
    "status": "Running",
    "node": "node-2",
    "ip": "10.0.kube.system.1",
    "created_at": "2024-04-01T12:00:00Z"
  }
]
```

### Get Cluster Services

```
GET /api/clusters/{cluster_id}/services
```

Returns services for a specific cluster, optionally filtered by namespace.

**Parameters**

- `cluster_id`: ID of the cluster (integer)
- `namespace` (optional): Namespace to filter by (string)

**Response**

```json
[
  {
    "name": "service-default-1",
    "namespace": "default",
    "type": "ClusterIP",
    "cluster_ip": "10.1.default.1",
    "external_ip": null,
    "ports": [
      {"port": 80, "target_port": 8080, "protocol": "TCP"}
    ],
    "created_at": "2024-04-01T12:00:00Z"
  },
  {
    "name": "service-default-3",
    "namespace": "default",
    "type": "LoadBalancer",
    "cluster_ip": "10.1.default.3",
    "external_ip": "192.168.1.3",
    "ports": [
      {"port": 80, "target_port": 8080, "protocol": "TCP"}
    ],
    "created_at": "2024-04-01T12:00:00Z"
  }
]
```

### Get Cluster Persistent Volumes

```
GET /api/clusters/{cluster_id}/persistent-volumes
```

Returns persistent volumes for a specific cluster.

**Parameters**

- `cluster_id`: ID of the cluster (integer)

**Response**

```json
[
  {
    "name": "pv-1",
    "capacity": "100Gi",
    "access_modes": ["ReadWriteOnce"],
    "reclaim_policy": "Retain",
    "status": "Bound",
    "claim": "default/pvc-1",
    "storage_class": "standard",
    "created_at": "2024-04-01T12:00:00Z"
  },
  {
    "name": "pv-2",
    "capacity": "50Gi",
    "access_modes": ["ReadWriteOnce"],
    "reclaim_policy": "Retain",
    "status": "Bound",
    "claim": "default/pvc-2",
    "storage_class": "standard",
    "created_at": "2024-04-01T12:00:00Z"
  }
]
```

---

## LLM Configuration

### List All LLM Configurations

```
GET /api/llm-config
```

Returns a list of all LLM configurations.

**Response**

```json
[
  {
    "id": 1,
    "provider": "openai",
    "model": "gpt-4o",
    "api_key": "sk-xxxx",
    "base_url": null,
    "parameters": {
      "temperature": 0.7,
      "top_p": 0.9
    },
    "is_default": true,
    "created_at": "2025-04-10T12:00:00Z"
  },
  {
    "id": 2,
    "provider": "ollama",
    "model": "llama3:8b",
    "api_key": null,
    "base_url": "http://localhost:11434",
    "parameters": {},
    "is_default": false,
    "created_at": "2025-04-10T14:00:00Z"
  }
]
```

### Get LLM Configuration by ID

```
GET /api/llm-config/{config_id}
```

Returns details for a specific LLM configuration.

**Parameters**

- `config_id`: ID of the LLM configuration (integer)

**Response**

```json
{
  "id": 1,
  "provider": "openai",
  "model": "gpt-4o",
  "api_key": "sk-xxxx",
  "base_url": null,
  "parameters": {
    "temperature": 0.7,
    "top_p": 0.9
  },
  "is_default": true,
  "created_at": "2025-04-10T12:00:00Z"
}
```

### Create LLM Configuration

```
POST /api/llm-config
```

Creates a new LLM configuration.

**Request Body**

```json
{
  "provider": "deepseek",
  "model": "deepseek-coder",
  "api_key": "sk-xxxx",
  "base_url": "https://api.deepseek.com",
  "parameters": {
    "temperature": 0.5,
    "max_tokens": 4096
  },
  "is_default": false
}
```

**Response**

```json
{
  "id": 3,
  "provider": "deepseek",
  "model": "deepseek-coder",
  "api_key": "sk-xxxx",
  "base_url": "https://api.deepseek.com",
  "parameters": {
    "temperature": 0.5,
    "max_tokens": 4096
  },
  "is_default": false,
  "created_at": "2025-04-11T15:00:00Z"
}
```

### Update LLM Configuration

```
PUT /api/llm-config/{config_id}
```

Updates an existing LLM configuration.

**Parameters**

- `config_id`: ID of the LLM configuration to update (integer)

**Request Body**

```json
{
  "model": "gpt-4o-mini",
  "parameters": {
    "temperature": 0.8
  }
}
```

All fields are optional. Only provided fields will be updated.

**Response**

```json
{
  "id": 1,
  "provider": "openai",
  "model": "gpt-4o-mini",
  "api_key": "sk-xxxx",
  "base_url": null,
  "parameters": {
    "temperature": 0.8,
    "top_p": 0.9
  },
  "is_default": true,
  "created_at": "2025-04-10T12:00:00Z"
}
```

### Delete LLM Configuration

```
DELETE /api/llm-config/{config_id}
```

Deletes an LLM configuration.

**Parameters**

- `config_id`: ID of the LLM configuration to delete (integer)

**Response**

```
204 No Content
```

---

## Chat Integration

### Send Chat Message

```
POST /api/chat
```

Sends a chat message and returns the AI assistant's response.

**Request Body**

```json
{
  "message": "How many nodes are in the production cluster?",
  "cluster_id": 1
}
```

**Response**

```json
{
  "id": 123,
  "message": "The production cluster currently has 3 nodes: 1 master node and 2 worker nodes."
}
```

### Get Chat History

```
GET /api/chat-history
```

Returns chat history, optionally filtered by cluster ID.

**Query Parameters**

- `limit` (optional): Maximum number of messages to retrieve (default: 20, max: 100)
- `cluster_id` (optional): Cluster ID to filter by

**Response**

```json
[
  {
    "id": 123,
    "user_message": "How many nodes are in the production cluster?",
    "assistant_message": "The production cluster currently has 3 nodes: 1 master node and 2 worker nodes.",
    "cluster_id": 1,
    "created_at": "2025-04-11T15:30:00Z"
  },
  {
    "id": 122,
    "user_message": "Show me CPU usage for the production cluster",
    "assistant_message": "Here is the current CPU usage for the production cluster...",
    "cluster_id": 1,
    "created_at": "2025-04-11T15:25:00Z"
  }
]
```

### Clear Chat History

```
DELETE /api/chat-history
```

Clears chat history, optionally filtered by cluster ID.

**Query Parameters**

- `cluster_id` (optional): Cluster ID to filter by

**Response**

```
204 No Content
