# agents/modular_agent.py
from typing import Dict, Optional, List, Tuple
from agents.base_agent import BaseAgent

class ModularAgent(BaseAgent):
    """
    Agent responsible for decomposing ideas into modular components.
    Implements the "Decompose" phase from Section 4.2.
    """
    
    def __init__(self):
        super().__init__("ModularAgent")
    
    def get_system_prompt(self) -> str:
        return """You are an expert software architect specializing in modular code design. Your role is to decompose complex ML solutions into clean, testable, independent modules."""
    
    def decompose_idea(
        self,
        problem_description: str,
        idea: str
    ) -> Optional[Dict[str, str]]:
        """
        Decompose idea into logical modules.
        
        Returns a dictionary of {module_name: module_description}
        
        Args:
            problem_description: Task description
            idea: Natural language solution idea
            
        Returns:
            Dict of module names to descriptions
        """
        self.log("Decomposing idea into modules...")
        
        # Get prompt
        prompt = self.prompt_manager.get_prompt(
            "modular_decomposition",
            problem_description=problem_description,
            idea=idea
        )
        
        # Call LLM
        response = self.call_llm(
            user_message=prompt,
            temperature=0.6,
            max_tokens=2000
        )
        
        # Parse JSON response
        modules = self.extract_json_from_response(response["content"])
        
        if modules:
            self.log(f"Decomposed into {len(modules)} modules: {list(modules.keys())}")
            return modules
        else:
            self.log("Failed to parse module decomposition")
            return None
    
    def validate_decomposition(
        self,
        modules: Dict[str, str]
    ) -> tuple[bool, List[str]]:
        """
        Validate module decomposition.
        
        Checks:
        - Has 'main' module
        - Reasonable number of modules (2-15)
        - No obvious circular dependencies
        
        Args:
            modules: Dict of module names to descriptions
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for main
        if "main" not in modules:
            issues.append("Missing 'main' module")
        
        # Check count
        if len(modules) < 2:
            issues.append("Too few modules (minimum 2)")
        elif len(modules) > 15:
            issues.append("Too many modules (maximum 15)")
        
        # Check for empty descriptions
        for name, desc in modules.items():
            if not desc.strip():
                issues.append(f"Module '{name}' has empty description")
        
        is_valid = len(issues) == 0
        return is_valid, issues