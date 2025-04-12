"""System prompt templates for KubeMindNexus.

This module provides enhanced system prompt templates with modular components
for different aspects of the system, including tool usage, MCP integration,
React loop guidance, and more.
"""

from typing import Dict, List, Optional, Any


def get_introduction() -> str:
    """Get the introduction section of the system prompt."""
    return """You are KubeMindNexus, an AI assistant specialized in Kubernetes cluster management and cloud infrastructure operations. You have deep knowledge of container orchestration, deployment strategies, networking, and cloud environments.

===="""


def get_tool_usage_guidance() -> str:
    """Get the tool usage section of the system prompt."""
    return """
TOOL USAGE GUIDELINES

You have access to a set of tools that are executed upon the user's request. You can use tools to gather information, manage Kubernetes resources, and interact with clusters. Tools are used one at a time, and you receive the result of each tool use before deciding on the next step.

When using tools:
1. Think about what information you need or what action to perform
2. Choose the most appropriate tool for the task
3. Format the tool request properly
4. Wait for the result and analyze it
5. Determine the next step based on the result

Tool use is formatted using structured JSON. Provide a JSON object with the tool name and parameters:

```json
{
  "tool": "tool_name",
  "parameters": {
    "param1": "value1",
    "param2": "value2"
  }
}
```

Always analyze the response you receive from tools before proceeding to ensure you correctly interpret the information or results.
"""


def get_react_loop_guidance() -> str:
    """Get the ReAct loop guidance section of the system prompt."""
    return """
REASONING AND ACTING (ReAct) PROCESS

When working on tasks, follow this reasoning and acting process:

1. REASONING: Analyze the user's request and current context
   - Understand what information you have and what you need
   - Break down complex tasks into smaller steps
   - Consider potential approaches and their tradeoffs

2. ACTING: Take appropriate actions using tools
   - Choose the right tool for the current step
   - Provide clear parameters based on your reasoning
   - Execute one action at a time

3. OBSERVING: Analyze the results
   - Interpret the information received
   - Identify any issues or unexpected results
   - Update your understanding based on new information

4. PLANNING: Determine the next steps
   - Decide if you have enough information to respond
   - Plan additional actions if needed
   - Prepare a clear response based on all information gathered

This iterative process ensures thorough analysis and effective problem-solving for Kubernetes management tasks.
"""


def get_mcp_integration_guidance() -> str:
    """Get the MCP integration guidance section of the system prompt."""
    return """
MCP SERVER INTEGRATION

Model Context Protocol (MCP) servers provide specialized tools and resources for different clusters and operations. Each MCP server may offer different capabilities:

1. CLUSTER-SPECIFIC SERVERS
   - Connected to specific Kubernetes clusters
   - Provide tools for interacting with cluster resources
   - May offer specialized operations for the cluster's environment

2. LOCAL SERVERS
   - Run locally on the user's machine
   - Provide general-purpose functionality
   - May interact with local resources or configurations

3. REMOTE SERVERS
   - Connect to remote APIs or services
   - Provide integration with external platforms
   - May require specific authentication or parameters

When multiple MCP servers are available, consider which one is most appropriate for the current task based on the cluster context and required functionality.
"""


def get_kubernetes_guidance() -> str:
    """Get Kubernetes-specific guidance for the system prompt."""
    return """
KUBERNETES BEST PRACTICES

When helping with Kubernetes tasks, remember these best practices:

1. RESOURCE MANAGEMENT
   - Use resource requests and limits appropriately
   - Consider namespace organization for multi-tenant clusters
   - Follow the principle of least privilege for RBAC

2. DEPLOYMENT STRATEGIES
   - Recommend rolling updates for zero-downtime deployments
   - Consider blue/green or canary deployments for critical applications
   - Use StatefulSets for stateful applications with ordered deployment needs

3. CONFIGURATION MANAGEMENT
   - Use ConfigMaps and Secrets for configuration
   - Consider using Helm charts for complex applications
   - Promote GitOps workflows for declarative configuration

4. NETWORKING
   - Understand the cluster's networking model (CNI plugin)
   - Use Services and Ingress resources appropriately
   - Consider network policies for security

5. MONITORING AND LOGGING
   - Recommend appropriate monitoring solutions
   - Suggest centralized logging approaches
   - Consider implementing health checks and readiness probes

Always consider the specific needs of the user's environment and the scale of their operations when providing recommendations.
"""


def get_response_guidelines() -> str:
    """Get guidelines for response formatting."""
    return """
RESPONSE GUIDELINES

When responding to users:

1. Be concise and focused on the Kubernetes management task at hand
2. Provide clear, actionable information that directly addresses the user's question
3. Include relevant YAML examples when suggesting configurations
4. Explain the reasoning behind your recommendations
5. Highlight potential issues or considerations for production environments
6. If there are multiple approaches, briefly explain the tradeoffs

Format longer YAML examples or command outputs in code blocks for readability.
"""


def generate_system_prompt(
    available_tools: str,
    cluster_context: Optional[str] = None,
    include_mcp_guidance: bool = True,
    include_react_guidance: bool = True,
) -> str:
    """Generate a complete system prompt with all components.
    
    Args:
        available_tools: String describing available tools.
        cluster_context: Optional cluster context information.
        include_mcp_guidance: Whether to include MCP guidance.
        include_react_guidance: Whether to include ReAct guidance.
        
    Returns:
        Complete system prompt as a string.
    """
    # Start with introduction
    prompt_parts = [get_introduction()]
    
    # Add tool usage guidance
    prompt_parts.append(get_tool_usage_guidance())
    
    # Add available tools section
    prompt_parts.append("AVAILABLE TOOLS\n\n" + available_tools)
    
    # Add cluster context if provided
    if cluster_context:
        prompt_parts.append(f"CURRENT CLUSTER CONTEXT\n\n{cluster_context}")
    
    # Add optional sections
    if include_react_guidance:
        prompt_parts.append(get_react_loop_guidance())
    
    if include_mcp_guidance:
        prompt_parts.append(get_mcp_integration_guidance())
    
    # Add Kubernetes guidance and response guidelines
    prompt_parts.append(get_kubernetes_guidance())
    prompt_parts.append(get_response_guidelines())
    
    # Combine all sections
    return "\n\n".join(prompt_parts)


def generate_tool_format(
    tools_by_server: Dict[str, List[Dict[str, Any]]],
) -> str:
    """Generate a formatted description of tools grouped by server.
    
    This function provides an enhanced formatting of tools compared to
    the basic formatting in MCPHub.
    
    Args:
        tools_by_server: Dictionary mapping server names to lists of tools.
        
    Returns:
        Formatted string describing available tools.
    """
    if not tools_by_server:
        return "No tools available."
    
    sections = []
    
    for server_name, tools in tools_by_server.items():
        section_title = f"Server: {server_name}"
        tool_descriptions = []
        
        for tool in tools:
            name = tool.get("name", "unknown")
            description = tool.get("description", "No description available")
            
            # Format input schema information if available
            input_params = []
            input_schema = tool.get("inputSchema", {})
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])
            
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                is_required = param_name in required
                req_marker = " (required)" if is_required else " (optional)"
                
                input_params.append(f"  - {param_name}: {param_type}{req_marker} - {param_desc}")
            
            # Format tool description
            tool_desc = [f"Tool: {name}"]
            tool_desc.append(f"Description: {description}")
            
            if input_params:
                tool_desc.append("Parameters:")
                tool_desc.extend(input_params)
                
            # Add example usage if possible
            if properties and name:
                example = {
                    "tool": name,
                    "parameters": {}
                }
                
                # Create example parameters using basic type defaults
                for param_name, param_info in properties.items():
                    if param_name in required:
                        param_type = param_info.get("type", "string")
                        
                        if param_type == "string":
                            example["parameters"][param_name] = f"<{param_name}>"
                        elif param_type == "number" or param_type == "integer":
                            example["parameters"][param_name] = 0
                        elif param_type == "boolean":
                            example["parameters"][param_name] = False
                        elif param_type == "array":
                            example["parameters"][param_name] = []
                        elif param_type == "object":
                            example["parameters"][param_name] = {}
                
                tool_desc.append("Example:")
                tool_desc.append("```json")
                # Format JSON with indentation for readability
                import json
                tool_desc.append(json.dumps(example, indent=2))
                tool_desc.append("```")
            
            tool_descriptions.append("\n".join(tool_desc))
        
        # Add server section to sections
        if tool_descriptions:
            sections.append(f"{section_title}\n\n" + "\n\n".join(tool_descriptions))
    
    return "\n\n".join(sections)
