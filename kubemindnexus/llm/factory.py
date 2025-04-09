"""LLM factory module for KubeMindNexus."""
from typing import Dict, Optional, Type, Any

from ..config.config import get_config
from ..utils.logger import LoggerMixin
from .client import (
    LLMClient, OpenAIClient, OllamaClient, DeepseekClient, OpenRouterClient
)


class LLMFactory(LoggerMixin):
    """Factory for creating LLM clients."""
    
    def __init__(self) -> None:
        """Initialize the LLM factory."""
        self.providers: Dict[str, Type[LLMClient]] = {
            "openai": OpenAIClient,
            "ollama": OllamaClient,
            "deepseek": DeepseekClient,
            "openrouter": OpenRouterClient
        }
    
    def create_client(self, provider: Optional[str] = None, **kwargs) -> LLMClient:
        """Create an LLM client.
        
        Args:
            provider: The provider to use. If None, uses the default provider.
            **kwargs: Additional arguments to pass to the client constructor.
            
        Returns:
            An LLM client instance.
            
        Raises:
            ValueError: If the provider is not supported.
        """
        config = get_config()
        provider = provider or config.get_default_llm_provider()
        
        if provider not in self.providers:
            self.logger.error(f"Unsupported LLM provider: {provider}")
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        client_class = self.providers[provider]
        self.logger.info(f"Creating {provider} LLM client")
        return client_class(**kwargs)
    
    def register_provider(self, name: str, client_class: Type[LLMClient]) -> None:
        """Register a new LLM provider.
        
        Args:
            name: The name of the provider.
            client_class: The client class for the provider.
        """
        self.providers[name] = client_class
        self.logger.info(f"Registered new LLM provider: {name}")
    
    def get_available_providers(self) -> list[str]:
        """Get the list of available providers.
        
        Returns:
            The list of available providers.
        """
        return list(self.providers.keys())
    
    def is_provider_available(self, provider: str) -> bool:
        """Check if a provider is available.
        
        Args:
            provider: The provider to check.
            
        Returns:
            True if the provider is available, False otherwise.
        """
        return provider in self.providers


# Singleton instance
llm_factory = LLMFactory()
