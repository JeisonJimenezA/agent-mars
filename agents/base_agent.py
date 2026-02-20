# agents/base_agent.py
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import json

from llm.llm_client import get_client
from llm.prompt_manager import get_prompt_manager
from utils.code_parser import CodeParser
from utils.debug_logger import get_debug_logger

class BaseAgent(ABC):
    """
    Base class for all MARS agents.
    Provides common functionality for LLM interaction.
    """

    def __init__(self, name: str):
        self.name = name
        self.client = get_client()
        self.prompt_manager = get_prompt_manager()
        self.parser = CodeParser()
        self.debug_logger = get_debug_logger()

        # Statistics
        self.total_calls = 0
        self.total_tokens = 0
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent"""
        pass
    
    def call_llm(
        self,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Call LLM with user message.

        Args:
            user_message: User message/prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            system_prompt: Override default system prompt

        Returns:
            Response dict with 'content', 'tokens', etc.
        """
        system = system_prompt or self.get_system_prompt()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message}
        ]

        response = self.client.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Update statistics
        self.total_calls += 1
        self.total_tokens += response.get("total_tokens", 0)

        # Debug logging
        self.debug_logger.log_llm_call(
            agent_name=self.name,
            prompt=user_message,
            response=response.get("content", ""),
            system_prompt=system,
            metadata={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "total_tokens": response.get("total_tokens", 0),
                "call_number": self.total_calls,
            }
        )

        return response
    
    def extract_json_from_response(self, response: str) -> Optional[Dict]:
        """
        Extract JSON object from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed JSON dict or None
        """
        json_str = self.parser.extract_json_from_text(response)
        
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"[{self.name}] Failed to parse JSON: {e}")
                return None
        
        return None
    
    def extract_code_from_response(self, response: str) -> List[str]:
        """
        Extract code blocks from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            List of code blocks
        """
        return self.parser.extract_code_from_markdown(response)
    
    def get_statistics(self) -> Dict[str, int]:
        """Get agent statistics"""
        return {
            "name": self.name,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens
        }
    
    def log(self, message: str):
        """Log message with agent name"""
        print(f"[{self.name}] {message}")