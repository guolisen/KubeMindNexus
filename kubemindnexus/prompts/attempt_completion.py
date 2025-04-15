"""Attempt completion tool for KubeMindNexus.

This module defines the attempt_completion tool, which allows the LLM to signal
that it has completed the user's task and wants to provide a final response.
"""

from ..constants import ATTEMPT_COMPLETION_TOOL_NAME

# Tool definition for attempt_completion
ATTEMPT_COMPLETION_TOOL_DEFINITION = {
    "name": ATTEMPT_COMPLETION_TOOL_NAME,
    "description": "Use this tool when you've completed the user's request and want to provide a final response.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "result": {
                "type": "string",
                "description": "A comprehensive final response that summarizes what was accomplished."
            },
            "command": {
                "type": "string",
                "description": "Optional CLI command to demonstrate the result."
            }
        },
        "required": ["result"]
    }
}

def get_attempt_completion_tool() -> dict:
    """Get the attempt_completion tool definition.
    
    Returns:
        Tool definition as a dictionary.
    """
    return ATTEMPT_COMPLETION_TOOL_DEFINITION
