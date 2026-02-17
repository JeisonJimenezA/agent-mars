# llm/deepseek_client.py - Unified LLM Client

import os
import time
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    """Unified LLM client supporting DeepSeek, Claude, Gemini, OpenAI, and Ollama."""

    # Supported providers
    PROVIDERS = ["ollama", "gemini", "anthropic", "openai", "deepseek"]

    def __init__(self):
        # Check which provider to use via LLM_PROVIDER env var or legacy flags
        provider = os.getenv("LLM_PROVIDER", "").lower()

        # Legacy support for USE_* flags
        if not provider:
            if os.getenv("USE_OLLAMA", "false").lower() == "true":
                provider = "ollama"
            elif os.getenv("USE_GEMINI", "false").lower() == "true":
                provider = "gemini"
            elif os.getenv("USE_ANTHROPIC", "false").lower() == "true":
                provider = "anthropic"
            elif os.getenv("USE_OPENAI", "false").lower() == "true":
                provider = "openai"
            else:
                provider = "deepseek"  # Default

        # Initialize selected provider
        if provider == "ollama":
            self._init_ollama()
        elif provider == "gemini":
            self._init_gemini()
        elif provider == "anthropic":
            self._init_anthropic()
        elif provider == "openai":
            self._init_openai()
        else:
            self._init_deepseek()

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

    def _init_gemini(self):
        """Initialize Google Gemini using dedicated module."""
        from llm.gemini_client import GeminiClient

        # Use dedicated Gemini client (new google.genai SDK)
        self._gemini_client = GeminiClient()
        self.model = self._gemini_client.model_name  # model_name in new client
        self.max_output_tokens = self._gemini_client.max_output_tokens
        self.provider = "gemini"
        self.client = self._gemini_client  # For compatibility

    def _init_anthropic(self):
        """Initialize Anthropic Claude"""
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        self.client = Anthropic(api_key=api_key)
        # Usar Sonnet 4.5 para mayor capacidad de tokens (hasta 16K en modo largo)
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
        self.max_output_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "16000"))
        self.provider = "anthropic"

        print(f"Using Claude: {self.model} (max_tokens: {self.max_output_tokens})")
    
    def _init_openai(self):
        """Initialize OpenAI"""
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.client = OpenAI(api_key=api_key)
        # GPT-4o, GPT-4-turbo, or GPT-3.5-turbo
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        # GPT-4o: 16K output, GPT-4-turbo: 4K output
        self.max_output_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "16000"))
        self.provider = "openai"

        print(f"Using OpenAI: {self.model} (max_tokens: {self.max_output_tokens})")

    def _init_deepseek(self):
        """Initialize DeepSeek"""
        from openai import OpenAI

        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        # DeepSeek V3 soporta hasta 8K output, deepseek-reasoner hasta 16K
        self.max_output_tokens = int(os.getenv("DEEPSEEK_MAX_TOKENS", "8192"))
        self.provider = "deepseek"

        print(f"Using DeepSeek: {self.model} (max_tokens: {self.max_output_tokens})")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Make chat completion request"""

        # Rate limiting
        self._rate_limit()

        # Use appropriate provider
        if self.provider == "ollama":
            return self._ollama_completion(messages, temperature, max_tokens)
        elif self.provider == "gemini":
            return self._gemini_completion(messages, temperature, max_tokens)
        elif self.provider == "anthropic":
            return self._anthropic_completion(messages, temperature, max_tokens)
        elif self.provider == "openai":
            return self._openai_completion(messages, temperature, max_tokens)
        else:
            return self._deepseek_completion(messages, temperature, max_tokens)
    
    def _ollama_completion(self, messages, temperature, max_tokens):
        """Ollama local LLM call - delegates to OllamaClient."""
        result = self._ollama_client.chat_completion(messages, temperature, max_tokens)

        # Update unified statistics
        self.total_requests += 1
        self.total_tokens += result.get("total_tokens", 0)

        return result

    def _gemini_completion(self, messages, temperature, max_tokens):
        """Gemini API call - delegates to dedicated GeminiClient."""
        result = self._gemini_client.chat_completion(messages, temperature, max_tokens)

        # Update unified statistics
        self.total_requests += 1
        self.total_tokens += result.get("total_tokens", 0)

        return result

    def _anthropic_completion(self, messages, temperature, max_tokens):
        """Claude API call"""

        # Convert messages format
        system_msg = ""
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)

        # Cap to provider's limit (Sonnet: 16K, Opus: 32K)
        tokens_to_use = min(max_tokens or self.max_output_tokens, self.max_output_tokens)

        # Call Claude
        response = self.client.messages.create(
            model=self.model,
            max_tokens=tokens_to_use,
            temperature=temperature or 0.5,
            system=system_msg,
            messages=user_messages
        )

        # Extract content
        content = response.content[0].text

        # Count tokens (approximate for Claude)
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens

        self.total_requests += 1
        self.total_tokens += total_tokens

        return {
            "content": content,
            "total_tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

    def _openai_completion(self, messages, temperature, max_tokens):
        """OpenAI API call with retry"""

        retry_attempts = 3
        retry_delay = 1

        # Cap max_tokens to OpenAI's limit
        effective_max_tokens = min(max_tokens or self.max_output_tokens, self.max_output_tokens)

        for attempt in range(retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature or 0.7,
                    max_tokens=effective_max_tokens
                )

                content = response.choices[0].message.content
                total_tokens = response.usage.total_tokens

                self.total_requests += 1
                self.total_tokens += total_tokens

                return {
                    "content": content,
                    "total_tokens": total_tokens,
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens
                }

            except Exception as e:
                error_str = str(e)

                # Handle rate limiting
                if "429" in error_str or "rate_limit" in error_str.lower():
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"OpenAI rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                if attempt < retry_attempts - 1:
                    print(f"OpenAI request failed (attempt {attempt + 1}/{retry_attempts}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise Exception(f"OpenAI API request failed after {retry_attempts} attempts: {e}")

    def _deepseek_completion(self, messages, temperature, max_tokens):
        """DeepSeek API call with retry"""

        retry_attempts = 3
        retry_delay = 1

        # Cap max_tokens to DeepSeek's limit (8192 for V3, 16384 for reasoner)
        effective_max_tokens = min(max_tokens or self.max_output_tokens, self.max_output_tokens)

        for attempt in range(retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature or 0.7,
                    max_tokens=effective_max_tokens
                )
                
                content = response.choices[0].message.content
                total_tokens = response.usage.total_tokens
                
                self.total_requests += 1
                self.total_tokens += total_tokens
                
                return {
                    "content": content,
                    "total_tokens": total_tokens,
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens
                }
                
            except Exception as e:
                if attempt < retry_attempts - 1:
                    print(f"API request failed (attempt {attempt + 1}/{retry_attempts}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise Exception(f"API request failed after {retry_attempts} attempts: {e}")
    
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
        """Estimate API cost"""
        if self.provider == "gemini":
            # Gemini 2.5 Pro pricing (per million tokens)
            # Input: $1.25/M (<200K), $2.50/M (>200K)
            # Output: $10.00/M (<200K), $15.00/M (>200K)
            cost_per_m_input = 1.25
            cost_per_m_output = 10.0
            return (self.total_tokens / 1_000_000) * ((cost_per_m_input + cost_per_m_output) / 2)
        elif self.provider == "anthropic":
            # Claude Sonnet 4.5 pricing
            cost_per_m_input = 3.0
            cost_per_m_output = 15.0
            return (self.total_tokens / 1_000_000) * ((cost_per_m_input + cost_per_m_output) / 2)
        elif self.provider == "openai":
            # GPT-4o pricing (per million tokens)
            # Input: $2.50/M, Output: $10.00/M
            cost_per_m_input = 2.50
            cost_per_m_output = 10.0
            return (self.total_tokens / 1_000_000) * ((cost_per_m_input + cost_per_m_output) / 2)
        else:
            # DeepSeek pricing
            cost_per_m_input = 0.27
            cost_per_m_output = 1.10
            return (self.total_tokens / 1_000_000) * ((cost_per_m_input + cost_per_m_output) / 2)


# Global instance
_client = None

def get_client() -> LLMClient:
    """Get or create global client"""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client