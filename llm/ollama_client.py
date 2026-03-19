# llm/ollama_client.py
"""
Ollama client for local LLM inference.
Supports any model available in Ollama (Qwen, Llama, Mistral, etc.)
"""

import os
import sys
import json
import time
import requests
from typing import Dict, List, Optional, Any


class OllamaClient:
    """Client for Ollama local LLM server."""

    def __init__(self):
        # Ollama server configuration
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b")
        self.max_output_tokens = int(os.getenv("OLLAMA_MAX_TOKENS", "16384"))

        # Verify server is running
        self._verify_server()

        # Statistics
        self.total_requests = 0
        self.total_tokens = 0

        print(f"Using Ollama: {self.model_name} (max_tokens: {self.max_output_tokens})")
        print(f"Server: {self.base_url}")

    # Known cloud-only model name prefixes (no local GGUF exists for these)
    CLOUD_MODEL_PREFIXES = (
        "gemini-",
        "claude-",
        "gpt-",
        "o1-",
        "o3-",
        "o4-",
    )

    def _is_cloud_model(self) -> bool:
        """
        Detect cloud models that don't carry the explicit ':cloud' suffix.
        Ollama cloud models backed by proprietary APIs (Gemini, Claude, GPT, etc.)
        do not accept llama.cpp options like num_predict and will return 500.
        """
        base = self.model_name.split(":")[0].lower()
        return any(base.startswith(prefix) for prefix in self.CLOUD_MODEL_PREFIXES)

    def _verify_server(self):
        """Verify Ollama server is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama server returned status {response.status_code}")

            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            # Cloud models (e.g. deepseek-v3.2:cloud) are not listed in /api/tags
            # They are served remotely after 'ollama signin' — skip local check
            is_cloud_model = self.model_name.endswith(":cloud")
            if not is_cloud_model:
                model_base = self.model_name.split(":")[0]
                model_available = any(model_base in name for name in model_names)

                if not model_available and models:
                    print(f"Warning: Model '{self.model_name}' not found locally.")
                    print(f"Available models: {model_names}")
                    print(f"Run: ollama pull {self.model_name}")

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama server at {self.base_url}\n"
                "Make sure Ollama is running: ollama serve"
            )
        except Exception as e:
            print(f"Warning: Could not verify Ollama server: {e}")

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool,
    ) -> Dict[str, Any]:
        """Build the request payload, handling cloud vs local model differences."""
        is_cloud = self.model_name.endswith(":cloud") or (
            ":" not in self.model_name and self._is_cloud_model()
        )
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
        }
        if not is_cloud:
            effective_tokens = min(
                max_tokens or self.max_output_tokens,
                self.max_output_tokens,
            )
            payload["options"] = {
                "num_predict": effective_tokens,
                "temperature": temperature if temperature is not None else 0,
            }
        return payload

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Make a chat completion request to Ollama.

        When stream=True, tokens are printed to stdout as they arrive so the
        user can see generation in real time. The full assembled response is
        returned the same way as the blocking mode.
        """
        if stream:
            return self._chat_completion_streaming(messages, temperature, max_tokens)
        return self._chat_completion_blocking(messages, temperature, max_tokens)

    def _chat_completion_blocking(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> Dict[str, Any]:
        """Blocking (non-streaming) completion — waits for the full response."""
        payload = self._build_payload(messages, temperature, max_tokens, stream=False)

        timeout_seconds = 900
        retry_attempts = 10
        retry_delay = 2

        for attempt in range(retry_attempts):
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=timeout_seconds,
                )

                if response.status_code != 200:
                    raise Exception(f"Ollama API error {response.status_code}: {response.text[:500]}")

                result = response.json()
                content = result.get("message", {}).get("content", "")
                eval_count = result.get("eval_count", len(content) // 4)
                prompt_eval_count = result.get("prompt_eval_count", 0)
                total_tokens = eval_count + prompt_eval_count

                self.total_requests += 1
                self.total_tokens += total_tokens

                return {
                    "content": content,
                    "total_tokens": total_tokens,
                    "input_tokens": prompt_eval_count,
                    "output_tokens": eval_count,
                    "model": result.get("model", self.model_name),
                    "eval_duration": result.get("eval_duration", 0) / 1e9,
                }

            except requests.exceptions.Timeout:
                if attempt < retry_attempts - 1:
                    print(f"Ollama request timed out, retrying ({attempt + 1}/{retry_attempts})...")
                    time.sleep(retry_delay)
                    continue
                raise Exception("Ollama request timed out after multiple attempts")

            except requests.exceptions.ConnectionError:
                if attempt < retry_attempts - 1:
                    print(f"Connection to Ollama lost, retrying ({attempt + 1}/{retry_attempts})...")
                    time.sleep(retry_delay)
                    continue
                raise Exception(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Is Ollama running? Try: ollama serve"
                )

            except Exception as e:
                if attempt < retry_attempts - 1:
                    print(f"Ollama request failed ({attempt + 1}/{retry_attempts}): {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

    def _chat_completion_streaming(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> Dict[str, Any]:
        """
        Streaming completion — prints each token to stdout as it arrives.
        Returns the same dict structure as the blocking version.
        """
        payload = self._build_payload(messages, temperature, max_tokens, stream=True)

        timeout_seconds = 900
        retry_attempts = 3
        retry_delay = 2

        for attempt in range(retry_attempts):
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    stream=True,
                    timeout=timeout_seconds,
                )

                if response.status_code != 200:
                    raise Exception(f"Ollama API error {response.status_code}: {response.text[:500]}")

                chunks = []
                eval_count = 0
                prompt_eval_count = 0

                print(f"\n[{self.model_name}] ", end="", flush=True)

                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        chunk = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        sys.stdout.write(token)
                        sys.stdout.flush()
                        chunks.append(token)

                    if chunk.get("done", False):
                        eval_count = chunk.get("eval_count", 0)
                        prompt_eval_count = chunk.get("prompt_eval_count", 0)
                        break

                # Newline after stream ends
                print()

                content = "".join(chunks)
                total_tokens = eval_count + prompt_eval_count

                self.total_requests += 1
                self.total_tokens += total_tokens

                return {
                    "content": content,
                    "total_tokens": total_tokens,
                    "input_tokens": prompt_eval_count,
                    "output_tokens": eval_count,
                    "model": self.model_name,
                    "eval_duration": 0,
                }

            except requests.exceptions.Timeout:
                if attempt < retry_attempts - 1:
                    print(f"\nOllama stream timed out, retrying ({attempt + 1}/{retry_attempts})...")
                    time.sleep(retry_delay)
                    continue
                raise Exception("Ollama streaming timed out after multiple attempts")

            except requests.exceptions.ConnectionError:
                if attempt < retry_attempts - 1:
                    print(f"\nConnection lost during stream, retrying ({attempt + 1}/{retry_attempts})...")
                    time.sleep(retry_delay)
                    continue
                raise Exception(f"Cannot connect to Ollama at {self.base_url}")

            except Exception as e:
                if attempt < retry_attempts - 1:
                    print(f"\nStream failed ({attempt + 1}/{retry_attempts}): {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Simple text generation (non-chat format).

        Args:
            prompt: The prompt text
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens or self.max_output_tokens,
                "temperature": temperature if temperature is not None else 0,
            }
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=600
        )

        if response.status_code != 200:
            raise Exception(f"Ollama generate error: {response.text}")

        return response.json().get("response", "")

    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            models = response.json().get("models", [])
            return [m.get("name", "") for m in models]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

    def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama registry.

        Args:
            model_name: Name of model to pull (e.g., 'qwen2.5-coder:14b')

        Returns:
            True if successful
        """
        print(f"Pulling model {model_name}... (this may take a while)")

        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name, "stream": False},
                timeout=5400  # 1.5 hours timeout for large models
            )

            if response.status_code == 200:
                print(f"Successfully pulled {model_name}")
                return True
            else:
                print(f"Failed to pull model: {response.text}")
                return False

        except Exception as e:
            print(f"Error pulling model: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "provider": "ollama",
            "model": self.model_name,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "server": self.base_url,
        }
