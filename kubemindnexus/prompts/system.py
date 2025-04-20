"""System prompt templates for KubeMindNexus.

This module provides enhanced system prompt templates with modular components
for different aspects of the system, including tool usage, MCP integration,
React loop guidance, and more.
"""
import json
from typing import Dict, List, Optional, Any
from .attempt_completion import get_attempt_completion_tool


def get_introduction() -> str:
    """Get the introduction section of the system prompt."""
    return """You are KubeMindNexus, an AI assistant specialized in Kubernetes cluster management and cloud infrastructure operations.
You have deep knowledge of container orchestration, deployment strategies, networking, and cloud environments. You can call mcp tools to sovle the user's request.
You are capable of reasoning, acting, and observing the results of your actions. You can also plan your next steps based on the results you observe.
if no need to call tools, summerize all of message and give the final response according to user's query.
===="""


def get_react_loop_guidance() -> str:
    """Get the ReAct loop guidance section of the system prompt."""
    return """
REASONING AND ACTING (ReAct) PROCESS

When working on tasks, follow this reasoning and acting process:

0. TASK COMPLETION: Signal when the task is complete
   - If no need to call tools, summarize all of the message and call the attempt_completion tool, give the final response according to user's query
   - **If you have completed all steps in the ReAct process, call the attempt_completion tool**
   - Use the attempt_completion tool ONLY after confirming all previous tool uses were successful
   - Before using attempt_completion, verify that you've received user confirmation for all previous actions
   - Failure to confirm previous tool successes could result in code corruption or system failure
   - Formulate your final result in a way that is complete and doesn't require further input
   - Don't end your result with questions or offers for further assistance
   - If providing a demonstration command, ensure it's properly formatted and appropriate for the user's system

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

5. ERROR HANDLING: Respond appropriately to tool execution errors
   - When a tool execution fails, analyze whether the error is fatal to task completion
   - For non-fatal errors, propose alternative approaches or tools to achieve the goal
   - For fatal errors that prevent task completion, use the attempt_completion tool to explain
     the issues encountered and provide any partial results or alternative suggestions
   - Be specific about why an error occurred and how it impacts the overall task

6. FINAL RESPONSE: Provide a clear and concise response
   - Summarize what was accomplished
   - Present the results in a user-friendly format
   - Include any relevant next steps or considerations

<< MUST IMPORTANT NOTICE >>
    - Each step in the ReAct process must be only run one tool at a time
    - Each LLM response must be only run one tool at a time
   
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
7. Avoid unnecessary jargon or overly technical language unless required
"""

def get_tool_usage_guidance() -> str:
    """Get test for response formatting."""
    return """
\n\n
==========

Choose the appropriate tool based on the user's question. 
    - If no need to call tools, summarize all of the message and call the 'attempt_completion' tool, give the final response according to user's query
    - If cannot find the parameters from current context, ask user for more information. 
IMPORTANT: When you need to use a tool, you must ONLY respond with
the exact JSON object format below, nothing else.

{
  "tool": "tool_name",      // Must be the exact tool name from the description
  "parameters": {
    "param1": "value1",     // Must be the exact parameter name from the description
    "param2": "value2",
    "param3": "",
  }
}

Note that the server name is critical - it must be the exact server name 
from either the ACTIVE CLUSTER section or LOCAL TOOLS section of the available tools.
This ensures the API server can correctly route your request to the appropriate MCP server.

*<<TOOL USAGE GUIDELINES>>*\n
*<< MUST IMPORTANT NOTICE >>*:\n
When calling MCP tools, you MUST strictly follow these rules:\n
    - if LLM need to call tools, each LLM response must be only run one tool at a time
    - DO NOT set the tool name outside the JSON object. 
    - !! Return ONLY a valid JSON object formatted as a tool call request !!\n
    - !! Absolutely NO explanations, comments, or extra text !!\n
    - Do NOT include any reasoning or thought process\n
    - Do NOT respond with any other text, just the formated JSON object, all of message should be in formated JSON\n
    - If you want to return none property in formated JSON, just return "", Do NOT use 'None'\n

*<< IMPORTANT AFTER RECEIVING A TOOL'S RESPONSE >>*:\n
When you receive a tool's response, follow these steps:\n
1. Transform the raw data into a natural, conversational response\n
2. Keep responses concise but informative\n
3. Focus on the most relevant information\n
4. Use appropriate context from the user's question\n
5. Avoid simply repeating the raw data\n
6. If no need to call tools, summarize all of the message and call the 'attempt_completion' tool, give the final response according to user's query\n

*<<TOOL USAGE GUIDELINES>>*\n
*<< MUST IMPORTANT NOTICE >>*:\n
When calling MCP tools, you MUST strictly follow these rules:\n
    - Return ONLY a valid JSON object as a tool call request\n
    - Absolutely NO explanations, comments, or extra text\n
    - Do NOT include any reasoning or thought process\n
    - Do NOT respond with any other text, just the JSON object\n
    - If you want to return none property in JSON, just return "", Do NOT use 'None'\n
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
    #if include_react_guidance:
    #    prompt_parts.append(get_react_loop_guidance())
    
    #if include_mcp_guidance:
    #    prompt_parts.append(get_mcp_integration_guidance())
    
    # Add Kubernetes guidance and response guidelines
    #prompt_parts.append(get_kubernetes_guidance())
    

    prompt_parts.append("\n=============\n")
    # Add attempt_completion tool to available tools
    attempt_completion_tool = get_attempt_completion_tool()
    available_tools_with_completion = ""
    available_tools_with_completion += "\n\n\n"

    attempt_completion_tool_json = {
        "name": attempt_completion_tool['name'],
        "description": attempt_completion_tool['description'],
        "server": "local",
        "parameters": {}
    }
    
    # Add parameters information
    input_schema = attempt_completion_tool.get("inputSchema", {})
    required = input_schema.get("required", [])
    
    for param_name, param_info in attempt_completion_tool['inputSchema']['properties'].items():
        is_required = param_name in required
        attempt_completion_tool_json["parameters"][param_name] = {
            "type": param_info.get("type", "string"),
            "description": param_info.get("description", "No description"),
            "required": is_required
        }
    available_tools_with_completion += str("### Attempt Completion Tool\n")
    available_tools_with_completion += str(f"\n### Tool: {attempt_completion_tool['name']}\n")
    available_tools_with_completion += str(json.dumps(attempt_completion_tool_json, indent=2))
    available_tools_with_completion += "\n\n\n"
    
    # Add available tools
    available_tools_with_completion = available_tools

    # Add tool usage guidance
    prompt_parts.append("AVAILABLE TOOLS:\n" + available_tools_with_completion)

    prompt_parts.append("\n=============\n")
    prompt_parts.append(get_tool_usage_guidance())
    prompt_parts.append("\n=============\n")
    prompt_parts.append(get_response_guidelines())
    prompt_parts.append("\n=============\n")
    #prompt_parts.append(get_kubernetes_guidance())
    prompt_parts.append("\n=============\n")
    prompt_parts.append(get_mcp_integration_guidance())
    prompt_parts.append("\n=============\n")

    # Combine all sections
    return "\n\n".join(prompt_parts)


def _create_example_json(server_name: str, tool_name: str, properties: Dict[str, Any], required: List[str]) -> str:
    """Create a JSON example for a tool based on its properties.
    
    Args:
        server_name: Name of the server providing the tool.
        tool_name: Name of the tool.
        properties: Dictionary of tool parameters and their schemas.
        required: List of required parameter names.
        
    Returns:
        Formatted JSON example as a string.
    """
    # Create parameters with example values based on their types
    params = {}
    for param_name, param_info in properties.items():
        param_type = param_info.get("type", "string")
        
        # Generate appropriate example value based on type
        if param_type == "string":
            # Use enum value if available, otherwise generic example
            if "enum" in param_info and param_info["enum"]:
                params[param_name] = param_info["enum"][0]
            else:
                params[param_name] = f"example_{param_name}"
        elif param_type == "integer" or param_type == "number":
            params[param_name] = 42
        elif param_type == "boolean":
            params[param_name] = True
        elif param_type == "array":
            params[param_name] = ["item1", "item2"]
        elif param_type == "object":
            params[param_name] = {"key": "value"}
        else:
            params[param_name] = "example_value"
    
    # Format the JSON example with proper indentation
    example = {
        "server": server_name,
        "tool": tool_name,
        "parameters": params
    }
    
    # Format the JSON as a string with indentation
    json_lines = [
        "{",
        f'  "server": "{server_name}",',
        f'  "tool": "{tool_name}",',
        '  "parameters": {'
    ]
    
    # Add parameters
    param_lines = []
    for param_name, param_value in params.items():
        if isinstance(param_value, str):
            param_lines.append(f'    "{param_name}": "{param_value}"')
        elif isinstance(param_value, bool):
            param_lines.append(f'    "{param_name}": {str(param_value).lower()}')
        else:
            param_lines.append(f'    "{param_name}": {param_value}')
    
    # Join parameters with commas
    json_lines.extend([f"{line}," for line in param_lines[:-1]])
    if param_lines:
        json_lines.append(param_lines[-1])
    
    # Close the JSON
    json_lines.append("  }")
    json_lines.append("}")
    
    return "\n".join(json_lines)


def generate_tool_format(
    tools_by_cluster: Dict[str, List[Dict[str, Any]]],
    tools_by_local: Dict[str, List[Dict[str, Any]]],
    cluster_context: Optional[str] = None
) -> str:
    """Generate a formatted description of tools grouped by server.
    
    This function produces a JSON format for each tool that clearly shows
    the LLM how to structure tool calls.
    
    Args:
        tools_by_cluster: Dictionary mapping cluster server names to lists of tools.
        tools_by_local: Dictionary mapping local server names to lists of tools.
        cluster_context: Optional name of the active cluster for highlighting.
        
    Returns:
        Formatted string describing available tools with JSON representations.
    """
    if not tools_by_cluster and not tools_by_local:
        return "No tools available."

    import json
    sections = []
    
    # Add usage instructions
    sections.append("""TOOL USAGE FORMAT:
When calling a tool, respond ONLY with a JSON object in this format:

{
  "server": "server_name",
  "tool": "tool_name",
  "parameters": {
    "param1": "value1",
    "param2": "value2"
  }
}

IMPORTANT: Return ONLY the JSON object, with NO additional text or explanations.""")
    
    # Add active cluster section if applicable
    if cluster_context and tools_by_cluster:
        sections.append(f"\n## ACTIVE CLUSTER: {cluster_context}")
        for server_name, tools in tools_by_cluster.items():
            section_tools = []
            
            for tool in tools:
                # Create a JSON representation of the tool
                tool_json = {
                    "name": tool.get("name", "unknown"),
                    "description": tool.get("description", "No description available"),
                    "server": server_name,
                    "parameters": {}
                }
                
                # Add parameters information
                input_schema = tool.get("inputSchema", {})
                properties = input_schema.get("properties", {})
                required = input_schema.get("required", [])
                
                for param_name, param_info in properties.items():
                    is_required = param_name in required
                    tool_json["parameters"][param_name] = {
                        "type": param_info.get("type", "string"),
                        "description": param_info.get("description", "No description"),
                        "required": is_required
                    }
                
                # Create example call JSON
                example_params = {}
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "string")
                    
                    # Generate appropriate example value based on type
                    if param_type == "string":
                        if "enum" in param_info and param_info["enum"]:
                            example_params[param_name] = param_info["enum"][0]
                        else:
                            example_params[param_name] = f"example_{param_name}"
                    elif param_type == "integer" or param_type == "number":
                        example_params[param_name] = 42
                    elif param_type == "boolean":
                        example_params[param_name] = True
                    elif param_type == "array":
                        example_params[param_name] = ["item1", "item2"]
                    elif param_type == "object":
                        example_params[param_name] = {"key": "value"}
                    else:
                        example_params[param_name] = "example_value"
                
                example_call = {
                    "server": server_name,
                    "tool": tool.get("name", "unknown"),
                    "parameters": example_params
                }
                
                #tool_json["example_call"] = example_call
                
                # Add formatted JSON string to sections
                section_tools.append(f"\n### Tool: {tool.get('name', 'unknown')}")
                section_tools.append(json.dumps(tool_json, indent=2))
            
            if section_tools:
                sections.append("\n".join(section_tools))
    
    # Add local tools section
    if tools_by_local:
        sections.append(f"\n## LOCAL TOOLS")
        for server_name, tools in tools_by_local.items():
            section_tools = []
            
            for tool in tools:
                # Create a JSON representation of the tool
                tool_json = {
                    "name": tool.get("name", "unknown"),
                    "description": tool.get("description", "No description available"),
                    "server": server_name,
                    "parameters": {}
                }
                
                # Add parameters information
                input_schema = tool.get("inputSchema", {})
                properties = input_schema.get("properties", {})
                required = input_schema.get("required", [])
                
                for param_name, param_info in properties.items():
                    is_required = param_name in required
                    tool_json["parameters"][param_name] = {
                        "type": param_info.get("type", "string"),
                        "description": param_info.get("description", "No description"),
                        "required": is_required
                    }
                
                # Create example call JSON
                example_params = {}
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "string")
                    
                    # Generate appropriate example value based on type
                    if param_type == "string":
                        if "enum" in param_info and param_info["enum"]:
                            example_params[param_name] = param_info["enum"][0]
                        else:
                            example_params[param_name] = f"example_{param_name}"
                    elif param_type == "integer" or param_type == "number":
                        example_params[param_name] = 42
                    elif param_type == "boolean":
                        example_params[param_name] = True
                    elif param_type == "array":
                        example_params[param_name] = ["item1", "item2"]
                    elif param_type == "object":
                        example_params[param_name] = {"key": "value"}
                    else:
                        example_params[param_name] = "example_value"
                
                example_call = {
                    "server": server_name,
                    "tool": tool.get("name", "unknown"),
                    "parameters": example_params
                }
                
                #tool_json["example_call"] = example_call
                
                # Add formatted JSON string to sections
                section_tools.append(f"\n### Tool: {tool.get('name', 'unknown')}")
                section_tools.append(json.dumps(tool_json, indent=2))
            
            if section_tools:
                sections.append("\n".join(section_tools))
    
    return "\n\n".join(sections)
