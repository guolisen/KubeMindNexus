"""LLM client module for KubeMindNexus."""
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Literal

import httpx

from ..config.config import get_config
from ..utils.logger import LoggerMixin


class LLMClient(ABC, LoggerMixin):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def get_response(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """Get a response from the LLM.
        
        Args:
            messages: A list of message dictionaries.
            temperature: The temperature for sampling.
            max_tokens: The maximum number of tokens to generate.
            
        Returns:
            The response from the LLM.
        """
        pass
    
    @abstractmethod
    async def get_chat_completion(self, messages: List[Dict[str, str]], 
                                temperature: float = 0.7, max_tokens: int = 4096, 
                                model: Optional[str] = None) -> Dict[str, Any]:
        """Get a chat completion from the LLM.
        
        Args:
            messages: A list of message dictionaries.
            temperature: The temperature for sampling.
            max_tokens: The maximum number of tokens to generate.
            model: Optional model override.
            
        Returns:
            The complete response object from the LLM.
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the LLM provider.
        
        Returns:
            The name of the LLM provider.
        """
        pass


class OpenAIClient(LLMClient):
    """Client for OpenAI's API."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, 
                base_url: Optional[str] = None) -> None:
        """Initialize an OpenAI client.
        
        Args:
            api_key: Optional API key. If None, uses the one from config.
            model: Optional model override. If None, uses the one from config.
            base_url: Optional base URL override. If None, uses the default.
        """
        config = get_config()
        self.api_key = api_key or config.get_openai_api_key()
        self.model = model or config.get_openai_model()
        self.base_url = base_url or "https://api.openai.com/v1"
        
        if not self.api_key:
            self.logger.warning("OpenAI API key not provided")
    
    def get_name(self) -> str:
        """Get the name of the LLM provider.
        
        Returns:
            The name of the LLM provider.
        """
        return "openai"
    
    async def get_chat_completion(self, messages: List[Dict[str, str]], 
                                temperature: float = 0.7, max_tokens: int = 4096,
                                model: Optional[str] = None) -> Dict[str, Any]:
        """Get a chat completion from OpenAI.
        
        Args:
            messages: A list of message dictionaries.
            temperature: The temperature for sampling.
            max_tokens: The maximum number of tokens to generate.
            model: Optional model override.
            
        Returns:
            The response object from OpenAI.
            
        Raises:
            ValueError: If the API key is not set.
            httpx.RequestError: If the request to OpenAI fails.
        """
        if not self.api_key:
            raise ValueError("OpenAI API key not set")
        
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        payload = {
            "messages": messages,
            "model": model or self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1,
            "stream": False,
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload, timeout=60.0)
                response.raise_for_status()
                return response.json()
        
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP Error from OpenAI: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            self.logger.error(f"Error communicating with OpenAI API: {str(e)}")
            raise
    
    async def get_response(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """Get a response from OpenAI.
        
        Args:
            messages: A list of message dictionaries.
            temperature: The temperature for sampling.
            max_tokens: The maximum number of tokens to generate.
            
        Returns:
            The response from OpenAI.
        """
        try:
            data = await self.get_chat_completion(messages, temperature, max_tokens)
            return data["choices"][0]["message"]["content"]
        
        except Exception as e:
            error_message = f"Error getting response from OpenAI: {str(e)}"
            self.logger.error(error_message)
            return f"I encountered an error: {error_message}. Please try again or rephrase your request."


class OllamaClient(LLMClient):
    """Client for Ollama's API."""
    
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None) -> None:
        """Initialize an Ollama client.
        
        Args:
            base_url: Optional base URL override. If None, uses the one from config.
            model: Optional model override. If None, uses "llama3" as default.
        """
        config = get_config()
        self.base_url = base_url or config.get_ollama_base_url()
        # Get other_params from config to see if we have a specific model configured
        other_params = config.config.llm.other_params
        default_model = other_params.get("ollama_model", "llama3") if other_params else "llama3"
        self.model = model or default_model
    
    def get_name(self) -> str:
        """Get the name of the LLM provider.
        
        Returns:
            The name of the LLM provider.
        """
        return "ollama"
    
    async def get_chat_completion(self, messages: List[Dict[str, str]], 
                                temperature: float = 0.7, max_tokens: int = 4096,
                                model: Optional[str] = None) -> Dict[str, Any]:
        """Get a chat completion from Ollama.
        
        Args:
            messages: A list of message dictionaries.
            temperature: The temperature for sampling.
            max_tokens: The maximum number of tokens to generate.
            model: Optional model override.
            
        Returns:
            The response object from Ollama.
            
        Raises:
            httpx.RequestError: If the request to Ollama fails.
        """
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "messages": messages,
            "model": model or self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=90.0)
                response.raise_for_status()
                return response.json()
        
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP Error from Ollama: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            self.logger.error(f"Error communicating with Ollama API: {str(e)}")
            raise
    
    async def get_response(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """Get a response from Ollama.
        
        Args:
            messages: A list of message dictionaries.
            temperature: The temperature for sampling.
            max_tokens: The maximum number of tokens to generate.
            
        Returns:
            The response from Ollama.
        """
        try:
            data = await self.get_chat_completion(messages, temperature, max_tokens)
            return data["message"]["content"]
        
        except Exception as e:
            error_message = f"Error getting response from Ollama: {str(e)}"
            self.logger.error(error_message)
            return f"I encountered an error: {error_message}. Please try again or rephrase your request."


class DeepseekClient(LLMClient):
    """Client for Deepseek's API."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        """Initialize a Deepseek client.
        
        Args:
            api_key: Optional API key. If None, uses the one from config.
            model: Optional model override. If None, uses default model.
        """
        config = get_config()
        self.api_key = api_key or config.get_deepseek_api_key()
        # Get other_params from config to see if we have a specific model configured
        other_params = config.config.llm.other_params
        default_model = other_params.get("deepseek_model", "deepseek-chat") if other_params else "deepseek-chat"
        self.model = model or default_model
        
        if not self.api_key:
            self.logger.warning("Deepseek API key not provided")
    
    def get_name(self) -> str:
        """Get the name of the LLM provider.
        
        Returns:
            The name of the LLM provider.
        """
        return "deepseek"
    
    async def get_chat_completion(self, messages: List[Dict[str, str]], 
                                temperature: float = 0.7, max_tokens: int = 4096,
                                model: Optional[str] = None) -> Dict[str, Any]:
        """Get a chat completion from Deepseek.
        
        Args:
            messages: A list of message dictionaries.
            temperature: The temperature for sampling.
            max_tokens: The maximum number of tokens to generate.
            model: Optional model override.
            
        Returns:
            The response object from Deepseek.
            
        Raises:
            ValueError: If the API key is not set.
            httpx.RequestError: If the request to Deepseek fails.
        """
        if not self.api_key:
            raise ValueError("Deepseek API key not set")
        
        url = "https://api.deepseek.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        payload = {
            "messages": messages,
            "model": model or self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1,
            "stream": False,
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload, timeout=60.0)
                response.raise_for_status()
                return response.json()
        
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP Error from Deepseek: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            self.logger.error(f"Error communicating with Deepseek API: {str(e)}")
            raise
    
    async def get_response(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """Get a response from Deepseek.
        
        Args:
            messages: A list of message dictionaries.
            temperature: The temperature for sampling.
            max_tokens: The maximum number of tokens to generate.
            
        Returns:
            The response from Deepseek.
        """
        try:
            data = await self.get_chat_completion(messages, temperature, max_tokens)
            return data["choices"][0]["message"]["content"]
        
        except Exception as e:
            error_message = f"Error getting response from Deepseek: {str(e)}"
            self.logger.error(error_message)
            return f"I encountered an error: {error_message}. Please try again or rephrase your request."


class OpenRouterClient(LLMClient):
    """Client for OpenRouter's API."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                base_url: Optional[str] = None) -> None:
        """Initialize an OpenRouter client.
        
        Args:
            api_key: Optional API key. If None, uses the one from config.
            model: Optional model override. If None, uses default model.
            base_url: Optional base URL override. If None, uses default.
        """
        config = get_config()
        self.api_key = api_key or config.get_openrouter_api_key()
        # Get other_params from config to see if we have a specific model configured
        other_params = config.config.llm.other_params
        default_model = other_params.get("openrouter_model", "mistralai/mistral-7b") if other_params else "mistralai/mistral-7b"
        self.model = model or default_model
        self.base_url = base_url or "https://openrouter.ai/api/v1"
        
        if not self.api_key:
            self.logger.warning("OpenRouter API key not provided")
    
    def get_name(self) -> str:
        """Get the name of the LLM provider.
        
        Returns:
            The name of the LLM provider.
        """
        return "openrouter"
    
    async def get_chat_completion(self, messages: List[Dict[str, str]], 
                                temperature: float = 0.7, max_tokens: int = 4096,
                                model: Optional[str] = None) -> Dict[str, Any]:
        """Get a chat completion from OpenRouter.
        
        Args:
            messages: A list of message dictionaries.
            temperature: The temperature for sampling.
            max_tokens: The maximum number of tokens to generate.
            model: Optional model override.
            
        Returns:
            The response object from OpenRouter.
            
        Raises:
            ValueError: If the API key is not set.
            httpx.RequestError: If the request to OpenRouter fails.
        """
        if not self.api_key:
            raise ValueError("OpenRouter API key not set")
        
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://kubemindnexus.io",
            "X-Title": "KubeMindNexus"
        }
        
        payload = {
            "messages": messages,
            "model": model or self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1,
            "stream": False,
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload, timeout=60.0)
                response.raise_for_status()
                return response.json()
        
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP Error from OpenRouter: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            self.logger.error(f"Error communicating with OpenRouter API: {str(e)}")
            raise
    
    async def get_response(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """Get a response from OpenRouter.
        
        Args:
            messages: A list of message dictionaries.
            temperature: The temperature for sampling.
            max_tokens: The maximum number of tokens to generate.
            
        Returns:
            The response from OpenRouter.
        """
        try:
            data = await self.get_chat_completion(messages, temperature, max_tokens)
            return data["choices"][0]["message"]["content"]
        
        except Exception as e:
            error_message = f"Error getting response from OpenRouter: {str(e)}"
            self.logger.error(error_message)
            return f"I encountered an error: {error_message}. Please try again or rephrase your request."
