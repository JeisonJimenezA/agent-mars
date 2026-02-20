# llm/llm_client.py - Unified LLM Client (OpenRouter + Ollama)

import os
import time
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    """Unified LLM client supporting OpenRouter (all cloud models) and Ollama (local)."""

    # Supported providers
    PROVIDERS = ["openrouter", "ollama"]

    def __init__(self):
        # Check which provider to use
        provider = os.getenv("LLM_PROVIDER", "").lower()

        # Legacy flag support
        if not provider:
            if os.getenv("USE_OLLAMA", "false").lower() == "true":
                provider = "ollama"
            else:
                provider = "openrouter"  # Default

        # Initialize selected provider
        if provider == "ollama":
            self._init_ollama()
        else:
            self._init_openrouter()

        # Rate limiting
        self.last_request_time = 0
        self.min_interval = 0.1  # 100ms between requests

        # Statistics
        self.total_requests = 0
        self.total_tokens = 0

    def _init_ollama(self):
        """Initialize Ollama local LLM."""
        from llm.ollama_client import OllamaClient

        self._ollama_client = OllamaClient()
        self.model = self._ollama_client.model_name
        self.max_output_tokens = self._ollama_client.max_output_tokens
        self.provider = "ollama"
        self.client = self._ollama_client

    def _init_openrouter(self):
        """Initialize OpenRouter - unified gateway to all cloud LLM providers."""
        from openai import OpenAI

        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/agent-mars",
                "X-Title": "Agent MARS"
            }
        )
        self.model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324")
        self.max_output_tokens = int(os.getenv("OPENROUTER_MAX_TOKENS", "16000"))
        self.provider = "openrouter"

        print(f"Using OpenRouter: {self.model} (max_tokens: {self.max_output_tokens})")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Make chat completion request"""

        # Rate limiting
        self._rate_limit()

        if self.provider == "ollama":
            return self._ollama_completion(messages, temperature, max_tokens)
        else:
            return self._openrouter_completion(messages, temperature, max_tokens)

    def _ollama_completion(self, messages, temperature, max_tokens):
        """Ollama local LLM call - delegates to OllamaClient."""
        result = self._ollama_client.chat_completion(messages, temperature, max_tokens)

        self.total_requests += 1
        self.total_tokens += result.get("total_tokens", 0)

        return result

    def _openrouter_completion(self, messages, temperature, max_tokens):
        """OpenRouter API call with retry (OpenAI-compatible)."""

        retry_attempts = 3
        retry_delay = 1

        effective_max_tokens = min(max_tokens or self.max_output_tokens, self.max_output_tokens)

        for attempt in range(retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature or 0,
                    max_tokens=effective_max_tokens
                )

                content = response.choices[0].message.content
                total_tokens = response.usage.total_tokens if response.usage else 0

                self.total_requests += 1
                self.total_tokens += total_tokens

                return {
                    "content": content,
                    "total_tokens": total_tokens,
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0
                }

            except Exception as e:
                error_str = str(e)

                if "429" in error_str or "rate_limit" in error_str.lower():
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"OpenRouter rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                if attempt < retry_attempts - 1:
                    print(f"OpenRouter request failed (attempt {attempt + 1}/{retry_attempts}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise Exception(f"OpenRouter API request failed after {retry_attempts} attempts: {e}")

    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "provider": self.provider,
            "model": self.model,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "estimated_cost": self._estimate_cost()
        }

    def _estimate_cost(self) -> float:
        """Estimate API cost (approximate - varies by model on OpenRouter)"""
        if self.provider == "ollama":
            return 0.0  # Local, no cost
        # OpenRouter pricing varies by model; use a conservative estimate
        # Average across common models: ~$1.5/M input, ~$6/M output
        cost_per_m_input = 1.5
        cost_per_m_output = 6.0
        return (self.total_tokens / 1_000_000) * ((cost_per_m_input + cost_per_m_output) / 2)


# Global instance
_client = None

def get_client() -> LLMClient:
    """Get or create global client"""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
