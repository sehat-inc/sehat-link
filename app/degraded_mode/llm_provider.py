"""
LLM Provider Abstraction

Easily swappable between OpenAI and local LLMs.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import os
from openai import AsyncOpenAI


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """Generate response from messages"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT-4o-mini provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """Generate response using OpenAI API"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")


class LocalLLMProvider(LLMProvider):
    """
    Local LLM provider (Ollama/llama.cpp)
    
    To use:
    1. Install: pip install ollama
    2. Run: ollama pull phi3:mini
    3. Set: DEGRADED_LLM_PROVIDER=local
    """
    
    def __init__(self, model: str = "phi3:mini", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        try:
            import ollama
            self.client = ollama.AsyncClient(host=base_url)
        except ImportError:
            raise RuntimeError(
                "Ollama not installed. Install with: pip install ollama"
            )
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """Generate response using local Ollama"""
        try:
            response = await self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            return response['message']['content']
        except Exception as e:
            raise RuntimeError(f"Local LLM error: {str(e)}")


def get_llm_provider() -> LLMProvider:
    """
    Factory function to get LLM provider based on environment.
    
    Set DEGRADED_LLM_PROVIDER to:
    - "openai" (default): Use OpenAI GPT-4o-mini
    - "local": Use local Ollama
    """
    provider_type = os.getenv("DEGRADED_LLM_PROVIDER", "openai").lower()
    
    if provider_type == "local":
        model = os.getenv("DEGRADED_LOCAL_MODEL", "phi3:mini")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return LocalLLMProvider(model=model, base_url=base_url)
    else:
        # Default to OpenAI
        model = os.getenv("DEGRADED_OPENAI_MODEL", "gpt-4o-mini")
        return OpenAIProvider(model=model)
