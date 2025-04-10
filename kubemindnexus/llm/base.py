"""Base LLM interface for KubeMindNexus."""

import abc
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from ..constants import LLMProvider

logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    """Message role in a conversation."""
    
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class LLMMessage:
    """Message for LLM APIs."""
    
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class BaseLLM(abc.ABC):
    """Base class for LLM integrations."""
    
    def __init__(
        self,
        provider: Union[str, LLMProvider],
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """Initialize LLM.
        
        Args:
            provider: LLM provider (e.g., OpenAI, Ollama).
            model: Model name.
            api_key: API key for authentication.
            base_url: Base URL for API requests.
            parameters: Additional parameters for the model.
        """
        self.provider = LLMProvider(provider) if isinstance(provider, str) else provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.parameters = parameters or {}
    
    @abc.abstractmethod
    async def generate(
        self, messages: List[LLMMessage], system_prompt: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate a response from the LLM.
        
        Args:
            messages: List of messages in the conversation.
            system_prompt: Optional system prompt to prepend to the conversation.
            
        Returns:
            Tuple of (response_text, tool_calls) where tool_calls is a list of tool calls
            that were extracted from the response.
        """
        pass
    
    @abc.abstractmethod
    async def generate_with_tools(
        self,
        messages: List[LLMMessage],
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate a response from the LLM with tool definitions.
        
        Args:
            messages: List of messages in the conversation.
            tools: List of tool definitions.
            system_prompt: Optional system prompt to prepend to the conversation.
            
        Returns:
            Tuple of (response_text, tool_calls) where tool_calls is a list of tool calls
            that were extracted from the response.
        """
        pass


class LLMFactory:
    """Factory for creating LLM instances."""
    
    def __init__(self, config=None, db_manager=None):
        """Initialize LLM factory.
        
        Args:
            config: Configuration instance.
            db_manager: Database manager instance.
        """
        self.config = config
        self.db_manager = db_manager
    
    def create_llm(
        self,
        provider: Union[str, LLMProvider],
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> BaseLLM:
        """Create an LLM instance.
        
        Args:
            provider: LLM provider (e.g., OpenAI, Ollama).
            model: Model name.
            api_key: API key for authentication.
            base_url: Base URL for API requests.
            parameters: Additional parameters for the model.
            
        Returns:
            LLM instance.
            
        Raises:
            ValueError: If the provider is not supported.
        """
        provider_enum = LLMProvider(provider) if isinstance(provider, str) else provider
        
        # Import implementations here to avoid circular imports
        from .ollama import OllamaLLM
        from .openai import OpenAILLM
        from .deepseek import DeepseekLLM
        from .openrouter import OpenRouterLLM
        
        if provider_enum == LLMProvider.OLLAMA:
            return OllamaLLM(model, api_key, base_url, parameters)
        elif provider_enum == LLMProvider.OPENAI:
            return OpenAILLM(model, api_key, base_url, parameters)
        elif provider_enum == LLMProvider.DEEPSEEK:
            return DeepseekLLM(model, api_key, base_url, parameters)
        elif provider_enum == LLMProvider.OPENROUTER:
            return OpenRouterLLM(model, api_key, base_url, parameters)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
