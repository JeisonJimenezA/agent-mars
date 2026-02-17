# llm/gemini_client.py
"""
Gemini client using the google-generativeai SDK.
Supports Gemini 2.5 Pro, Gemini 2.0 Flash, and Gemini 1.5 Pro.
"""

import os
import time
from typing import Dict, List, Optional, Any

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Run: pip install google-generativeai")


class GeminiClient:
    """Client for Google Gemini API using google-generativeai SDK."""

    def __init__(self):
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            )

        # Get API key from environment
        api_key = os.getenv("GOOGLE_GENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Set GOOGLE_GENAI_API_KEY or GEMINI_API_KEY"
            )

        # Configure the SDK
        genai.configure(api_key=api_key)

        # Get model configuration
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.max_output_tokens = int(os.getenv("GEMINI_MAX_TOKENS", "8192"))

        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "max_output_tokens": self.max_output_tokens,
                "temperature": 0.7,
            }
        )

        # Statistics
        self.total_requests = 0
        self.total_tokens = 0

        print(f"Using Gemini: {self.model_name} (max_tokens: {self.max_output_tokens})")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Make a chat completion request to Gemini.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum output tokens

        Returns:
            Dict with 'content', 'total_tokens', etc.
        """
        # Convert messages to Gemini format
        gemini_messages = self._convert_messages(messages)

        # Build generation config
        gen_config = {
            "max_output_tokens": max_tokens or self.max_output_tokens,
            "temperature": temperature if temperature is not None else 0.7,
        }

        # Create a new model instance with updated config if needed
        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=gen_config
        )

        # Make request with retry
        retry_attempts = 3
        retry_delay = 1

        for attempt in range(retry_attempts):
            try:
                # Start chat and send messages
                chat = model.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])

                # Send the last message
                last_message = gemini_messages[-1] if gemini_messages else {"parts": [{"text": "Hello"}]}
                response = chat.send_message(last_message["parts"][0]["text"])

                # Extract content
                content = response.text

                # Estimate tokens (Gemini doesn't always provide token counts)
                input_tokens = sum(len(m.get("parts", [{}])[0].get("text", "")) // 4
                                   for m in gemini_messages)
                output_tokens = len(content) // 4
                total_tokens = input_tokens + output_tokens

                self.total_requests += 1
                self.total_tokens += total_tokens

                return {
                    "content": content,
                    "total_tokens": total_tokens,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }

            except Exception as e:
                error_str = str(e).lower()

                # Handle rate limiting
                if "429" in str(e) or "quota" in error_str or "rate" in error_str:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Gemini rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                # Handle safety blocks
                if "safety" in error_str or "blocked" in error_str:
                    print(f"Gemini safety filter triggered, retrying with adjusted content...")
                    # Could modify the prompt here, but for now just retry
                    if attempt < retry_attempts - 1:
                        time.sleep(retry_delay)
                        continue

                if attempt < retry_attempts - 1:
                    print(f"Gemini request failed (attempt {attempt + 1}/{retry_attempts}): {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise Exception(f"Gemini API request failed after {retry_attempts} attempts: {e}")

    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[Dict]:
        """
        Convert OpenAI-style messages to Gemini format.

        Gemini uses:
        - 'user' and 'model' roles
        - 'parts' instead of 'content'
        - System messages become part of the first user message
        """
        gemini_messages = []
        system_content = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Accumulate system message to prepend to first user message
                system_content = content
            elif role == "assistant":
                # Gemini uses 'model' instead of 'assistant'
                gemini_messages.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })
            else:  # user
                # Prepend system content to first user message if present
                if system_content and not gemini_messages:
                    content = f"{system_content}\n\n{content}"
                    system_content = ""

                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })

        # Ensure we have at least one message
        if not gemini_messages:
            gemini_messages.append({
                "role": "user",
                "parts": [{"text": "Hello"}]
            })

        return gemini_messages

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "provider": "gemini",
            "model": self.model_name,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
        }
