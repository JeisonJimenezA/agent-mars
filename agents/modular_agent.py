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
            temperature=0,
            max_tokens=16000
        )
        
        # Parse JSON response
        modules = self.extract_json_from_response(response["content"])

        if modules:
            # Normalize descriptions to strings (LLM sometimes returns dicts)
            modules = self._normalize_module_descriptions(modules)
            self.log(f"Decomposed into {len(modules)} modules: {list(modules.keys())}")
            return modules
        else:
            self.log("Failed to parse module decomposition")
            return None

    def _normalize_module_descriptions(self, modules: Dict) -> Dict[str, str]:
        """
        Normalize module descriptions to ensure they are all strings.
        LLM sometimes returns nested dicts instead of plain strings.
        """
        normalized = {}
        for name, desc in modules.items():
            if isinstance(desc, dict):
                # Extract description from common keys
                desc_str = desc.get('purpose', '') or desc.get('description', '') or ''
                # Include other relevant info if available
                if 'functions' in desc:
                    funcs = desc['functions']
                    if isinstance(funcs, list):
                        desc_str += f" Functions: {', '.join(funcs)}."
                if 'classes' in desc:
                    classes = desc['classes']
                    if isinstance(classes, list):
                        desc_str += f" Classes: {', '.join(classes)}."
                # Fallback to string representation if still empty
                if not desc_str.strip():
                    desc_str = str(desc)
                normalized[name] = desc_str
            elif isinstance(desc, str):
                normalized[name] = desc
            else:
                normalized[name] = str(desc)
        return normalized
    
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
            # Handle both string and dict descriptions from LLM
            if isinstance(desc, dict):
                desc = desc.get('purpose', '') or desc.get('description', '') or str(desc)
            if not isinstance(desc, str) or not desc.strip():
                issues.append(f"Module '{name}' has empty description")
        
        is_valid = len(issues) == 0
        return is_valid, issues