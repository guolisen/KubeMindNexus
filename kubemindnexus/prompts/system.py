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

<< MUST IMPORTANT NOTICE >>:
When calling MCP tools, you MUST strictly follow these rules:
    - Return ONLY a valid JSON object formatted as a tool call request
    - Absolutely NO explanations, comments, or extra text
    - Do NOT include any reasoning or thought process

Format longer YAML examples or command outputs in code blocks for readability.
"""

def get_tool_usage_guidance() -> str:
    """Get test for response formatting."""
    return """
\n\n
==========

Choose the appropriate tool based on the user's question. 
    - If no tool is needed, reply directly.
    - If cannot find the parameters from current context, ask user for more information. 
IMPORTANT: When you need to use a tool, you must ONLY respond with
the exact JSON object format below, nothing else:\n

{
  "server": "server_name",  // Must include the exact server name from the tool description
  "tool": "tool_name",      // Must be the exact tool name from the description
  "parameters": {
    "param1": "value1",
    "param2": "value2",
    "param3": "",
  }
}

Note that the server name is critical - it must be the exact server name 
from either the ACTIVE CLUSTER section or LOCAL TOOLS section of the available tools.
This ensures the API server can correctly route your request to the appropriate MCP server.

*<< IMPORTANT AFTER RECEIVING A TOOL'S RESPONSE >>*:\n
When you receive a tool's response, follow these steps:\n
1. Transform the raw data into a natural, conversational response\n
2. Keep responses concise but informative\n
3. Focus on the most relevant information\n
4. Use appropriate context from the user's question\n
5. Avoid simply repeating the raw data\n
6. If current information also cannot meet the requirement of user, try continue to call tools.\n
7. If no need to call tools, summerize all of message and give the final response according to user's query\n

*<<TOOL USAGE GUIDELINES>>*
*<< MUST IMPORTANT NOTICE >>*:
When calling MCP tools, you MUST strictly follow these rules:
    - Return ONLY a valid JSON object formatted as a tool call request
    - Absolutely NO explanations, comments, or extra text
    - Do NOT include any reasoning or thought process
    - Do NOT respond with any other text, just the JSON object\n\n
    - If you want to return none property in JSON, just return "", Do NOT use 'None'
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
    
    # Add cluster context if provided
    #if cluster_context:
    prompt_parts.append(f"CURRENT CLUSTER CONTEXT\n\n{cluster_context}")
    
    # Add optional sections
    if include_react_guidance:
        prompt_parts.append(get_react_loop_guidance())
    
    if include_mcp_guidance:
        prompt_parts.append(get_mcp_integration_guidance())
    
    # Add Kubernetes guidance and response guidelines
    prompt_parts.append(get_kubernetes_guidance())
    #prompt_parts.append(get_response_guidelines())
    prompt_parts.append("\n=============\n")
    # Add tool usage guidance
    prompt_parts.append("AVAILABLE TOOLS:\n" + available_tools)
    prompt_parts.append(get_tool_usage_guidance())

    # Combine all sections
    return "\n\n".join(prompt_parts)


def generate_tool_format(
    tools_by_cluster: Dict[str, List[Dict[str, Any]]],
    tools_by_local: Dict[str, List[Dict[str, Any]]],
    cluster_context: Optional[str] = None
) -> str:
    """Generate a formatted description of tools grouped by server.
    
    This function produces a more detailed format that clearly distinguishes
    between cluster-specific tools and local tools, with improved organization.
    
    Args:
        tools_by_server: Dictionary mapping server names to lists of tools.
        cluster_context: Optional name of the active cluster for highlighting.
        
    Returns:
        Formatted string describing available tools.
    """
    if not tools_by_cluster and not tools_by_local:
        return "No tools available."

    # Organize servers into cluster-specific and local categories
    cluster_servers = {}
    local_servers = {}
    other_servers = {}
    
    for server_name, tools in tools_by_cluster.items():
        cluster_servers[server_name] = tools
    for server_name, tools in tools_by_local.items():
        local_servers[server_name] = tools

    sections = []
    
    # Add active cluster section if applicable
    if cluster_context and cluster_servers:
        sections.append(f"## ACTIVE CLUSTER: {cluster_context}")
        for server_name, tools in cluster_servers.items():
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
                    param_desc = param_info.get("description", "No description")
                    param_type = param_info.get("type", "any")
                    is_required = param_name in required
                    req_marker = " (required)" if is_required else ""
                    
                    input_params.append(f"- {param_name}: {param_desc}{req_marker}")
                
                # Format tool description
                tool_desc = [f"\nTool: {name}"]
                tool_desc.append(f"Description: {description}")
                
                if input_params:
                    tool_desc.append("Arguments:")
                    tool_desc.extend(input_params)
                
                tool_descriptions.append("\n".join(tool_desc))
            
            # Add server section
            if tool_descriptions:
                sections.append(f"{section_title}\n\n" + "\n\n".join(tool_descriptions))
    
    # Add local servers section
    if local_servers:
        sections.append(f"\n## LOCAL TOOLS")
        for server_name, tools in local_servers.items():
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
                    param_desc = param_info.get("description", "No description")
                    param_type = param_info.get("type", "any")
                    is_required = param_name in required
                    req_marker = " (required)" if is_required else ""
                    
                    input_params.append(f"- {param_name}: {param_desc}{req_marker}")
                
                # Format tool description
                tool_desc = [f"\nTool: {name}"]
                tool_desc.append(f"Description: {description}")
                
                if input_params:
                    tool_desc.append("Arguments:")
                    tool_desc.extend(input_params)
                
                tool_descriptions.append("\n".join(tool_desc))
            
            # Add server section
            if tool_descriptions:
                sections.append(f"{section_title}\n\n" + "\n\n".join(tool_descriptions))
    
    # Add other servers section (if any)
    if other_servers:
        sections.append(f"\n## OTHER TOOLS")
        for server_name, tools in other_servers.items():
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
                    param_desc = param_info.get("description", "No description")
                    param_type = param_info.get("type", "any")
                    is_required = param_name in required
                    req_marker = " (required)" if is_required else ""
                    
                    input_params.append(f"- {param_name}: {param_desc}{req_marker}")
                
                # Format tool description
                tool_desc = [f"\nTool: {name}"]
                tool_desc.append(f"Description: {description}")
                
                if input_params:
                    tool_desc.append("Arguments:")
                    tool_desc.extend(input_params)
                
                tool_descriptions.append("\n".join(tool_desc))
            
            # Add server section
            if tool_descriptions:
                sections.append(f"{section_title}\n\n" + "\n\n".join(tool_descriptions))
    
    return "\n\n".join(sections)
