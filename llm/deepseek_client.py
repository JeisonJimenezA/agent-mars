# llm/deepseek_client.py - REEMPLAZAR TODO

import os
import time
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    """Unified LLM client supporting DeepSeek and Claude."""
    
    def __init__(self):
        # Check which provider to use
        self.use_anthropic = os.getenv("USE_ANTHROPIC", "false").lower() == "true"
        
        if self.use_anthropic:
            self._init_anthropic()
        else:
            self._init_deepseek()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_interval = 0.1  # 100ms between requests
        
        # Statistics
        self.total_requests = 0
        self.total_tokens = 0
    
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
        if self.provider == "anthropic":
            return self._anthropic_completion(messages, temperature, max_tokens)
        else:
            return self._deepseek_completion(messages, temperature, max_tokens)
    
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

        # Usar el máximo configurado (16K para Sonnet 4.5)
        tokens_to_use = max_tokens or self.max_output_tokens

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
    
    def _deepseek_completion(self, messages, temperature, max_tokens):
        """DeepSeek API call with retry"""
        
        retry_attempts = 3
        retry_delay = 1
        
        for attempt in range(retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature or 0.7,
                    max_tokens=max_tokens or self.max_output_tokens
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
        if self.provider == "anthropic":
            # Claude 3.5 Sonnet pricing
            cost_per_m_input = 3.0
            cost_per_m_output = 15.0
            # Rough approximation (50/50 split)
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