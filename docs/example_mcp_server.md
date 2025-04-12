# Example MCP Server for Kubernetes Management

This document provides an example of how to create a Model Context Protocol (MCP) server for KubeMindNexus that works effectively with the enhanced system prompt architecture.

## Overview

The example MCP server provides tools for interacting with a Kubernetes cluster, demonstrating how to leverage the enhanced system prompt architecture for better tool documentation and usage.

## Prerequisites

- Node.js 14+
- MCP SDK (`npm install @modelcontextprotocol/sdk`)
- Kubernetes cluster access configured via `kubectl`

## Server Implementation

Create a new directory for your MCP server:

```bash
mkdir -p ~/Documents/k8s-mcp-server
cd ~/Documents/k8s-mcp-server
npm init -y
npm install @modelcontextprotocol/sdk @kubernetes/client-node
```

Create a file named `index.js` with the following content:

```javascript
#!/usr/bin/env node
const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
} = require('@modelcontextprotocol/sdk/types.js');
const k8s = require('@kubernetes/client-node');

// Initialize Kubernetes client
const kc = new k8s.KubeConfig();
kc.loadFromDefault();
const k8sAppsV1Api = kc.makeApiClient(k8s.AppsV1Api);
const k8sCoreV1Api = kc.makeApiClient(k8s.CoreV1Api);

class KubernetesServer {
  constructor() {
    this.server = new Server(
      {
        name: 'kubernetes-server',
        version: '0.1.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupToolHandlers();
    
    // Error handling
    this.server.onerror = (error) => console.error('[MCP Error]', error);
    process.on('SIGINT', async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  setupToolHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'list_deployments',
          description: 'List deployments in a Kubernetes namespace',
          inputSchema: {
            type: 'object',
            properties: {
              namespace: {
                type: 'string',
                description: 'Kubernetes namespace (default: "default")',
              },
            },
          },
        },
        {
          name: 'list_pods',
          description: 'List pods in a Kubernetes namespace',
          inputSchema: {
            type: 'object',
            properties: {
              namespace: {
                type: 'string',
                description: 'Kubernetes namespace (default: "default")',
              },
              labelSelector: {
                type: 'string',
                description: 'Label selector to filter pods',
              },
            },
          },
        },
        {
          name: 'get_pod_logs',
          description: 'Get logs from a specific pod',
          inputSchema: {
            type: 'object',
            properties: {
              namespace: {
                type: 'string',
                description: 'Kubernetes namespace (default: "default")',
              },
              podName: {
                type: 'string',
                description: 'Name of the pod',
              },
              container: {
                type: 'string',
                description: 'Container name (if pod has multiple containers)',
              },
              tailLines: {
                type: 'number',
                description: 'Number of lines to retrieve from the end of the logs',
              },
            },
            required: ['podName'],
          },
        },
        {
          name: 'scale_deployment',
          description: 'Scale a deployment to a specified number of replicas',
          inputSchema: {
            type: 'object',
            properties: {
              namespace: {
                type: 'string',
                description: 'Kubernetes namespace (default: "default")',
              },
              deploymentName: {
                type: 'string',
                description: 'Name of the deployment',
              },
              replicas: {
                type: 'number',
                description: 'Number of replicas',
              },
            },
            required: ['deploymentName', 'replicas'],
          },
        },
      ],
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const toolName = request.params.name;
      const args = request.params.arguments || {};

      try {
        let result;
        switch (toolName) {
          case 'list_deployments':
            result = await this.listDeployments(args.namespace || 'default');
            break;
          case 'list_pods':
            result = await this.listPods(args.namespace || 'default', args.labelSelector);
            break;
          case 'get_pod_logs':
            if (!args.podName) {
              throw new McpError(ErrorCode.InvalidParams, 'Pod name is required');
            }
            result = await this.getPodLogs(
              args.namespace || 'default',
              args.podName,
              args.container,
              args.tailLines
            );
            break;
          case 'scale_deployment':
            if (!args.deploymentName || args.replicas === undefined) {
              throw new McpError(
                ErrorCode.InvalidParams,
                'Deployment name and replicas are required'
              );
            }
            result = await this.scaleDeployment(
              args.namespace || 'default',
              args.deploymentName,
              args.replicas
            );
            break;
          default:
            throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${toolName}`);
        }

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      } catch (error) {
        console.error(`Error executing tool ${toolName}:`, error);
        return {
          content: [
            {
              type: 'text',
              text: `Error: ${error.message || 'Unknown error'}`,
            },
          ],
          isError: true,
        };
      }
    });
  }

  async listDeployments(namespace) {
    try {
      const res = await k8sAppsV1Api.listNamespacedDeployment(namespace);
      return res.body.items.map((deployment) => ({
        name: deployment.metadata.name,
        replicas: deployment.spec.replicas,
        availableReplicas: deployment.status.availableReplicas || 0,
        creationTimestamp: deployment.metadata.creationTimestamp,
      }));
    } catch (err) {
      throw new Error(`Failed to list deployments: ${err.message}`);
    }
  }

  async listPods(namespace, labelSelector) {
    try {
      const res = await k8sCoreV1Api.listNamespacedPod(
        namespace,
        undefined,
        undefined,
        undefined,
        undefined,
        labelSelector
      );
      return res.body.items.map((pod) => ({
        name: pod.metadata.name,
        status: pod.status.phase,
        podIP: pod.status.podIP,
        nodeName: pod.spec.nodeName,
        creationTimestamp: pod.metadata.creationTimestamp,
      }));
    } catch (err) {
      throw new Error(`Failed to list pods: ${err.message}`);
    }
  }

  async getPodLogs(namespace, podName, container, tailLines) {
    try {
      const res = await k8sCoreV1Api.readNamespacedPodLog(
        podName,
        namespace,
        container,
        undefined,
        false,
        undefined,
        undefined,
        undefined,
        tailLines,
        undefined
      );
      return { logs: res.body };
    } catch (err) {
      throw new Error(`Failed to get pod logs: ${err.message}`);
    }
  }

  async scaleDeployment(namespace, deploymentName, replicas) {
    try {
      // First, get the current deployment
      const deployment = await k8sAppsV1Api.readNamespacedDeployment(deploymentName, namespace);
      
      // Update the replicas
      deployment.body.spec.replicas = replicas;
      
      // Apply the update
      const res = await k8sAppsV1Api.replaceNamespacedDeployment(
        deploymentName,
        namespace,
        deployment.body
      );
      
      return {
        name: res.body.metadata.name,
        namespace: res.body.metadata.namespace,
        replicas: res.body.spec.replicas,
        message: `Deployment ${deploymentName} scaled to ${replicas} replicas`,
      };
    } catch (err) {
      throw new Error(`Failed to scale deployment: ${err.message}`);
    }
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Kubernetes MCP server running on stdio');
  }
}

// Run the server
const server = new KubernetesServer();
server.run().catch(console.error);
```

Make the script executable:

```bash
chmod +x index.js
```

## Adding the Server to KubeMindNexus

1. Edit the KubeMindNexus configuration file:

```bash
vim ~/.config/kubemindnexus/config.json
```

2. Add the MCP server to the configuration:

```json
{
  "db_path": "~/.config/kubemindnexus/kubemindnexus.db",
  "system_prompt_template": "use_enhanced",
  "system_prompt_options": {
    "include_react_guidance": true,
    "include_mcp_guidance": true,
    "include_kubernetes_guidance": true
  },
  "default_mcp_servers": ["kubernetes-server"],
  "llm": {
    // existing LLM configuration
  },
  "mcpServers": {
    "kubernetes-server": {
      "command": "node",
      "args": ["~/Documents/k8s-mcp-server/index.js"],
      "type": "stdio",
      "is_local": true,
      "is_default": true
    }
  }
}
```

## Testing the Server

Start KubeMindNexus, and you should see the Kubernetes MCP server connected. You can now use the tools provided by the server to interact with your Kubernetes cluster through KubeMindNexus.

Example queries:

- "Show me all deployments in the default namespace"
- "Scale the nginx-deployment to 3 replicas"
- "Get logs from the pod web-frontend-57f9df7b7d-xz2wq"

## Benefits of the Enhanced System Prompt

With the enhanced system prompt architecture, KubeMindNexus will:

1. Provide better tool documentation with examples
2. Offer clearer guidance on the ReAct process for complex operations
3. Include Kubernetes-specific best practices
4. Format tool responses more effectively

The MCP server leverages these enhancements to provide a more effective user experience when working with Kubernetes clusters.

## Further Extensions

This basic example could be extended with more sophisticated tools:

- Deploying applications from templates
- Troubleshooting resources
- Managing ConfigMaps and Secrets
- Analyzing cluster health
- Implementing canary deployments

Each of these would benefit from the enhanced system prompt's ability to provide detailed guidance and examples.
