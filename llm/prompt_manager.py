# llm/prompt_manager.py
from pathlib import Path
from typing import Dict, Optional, Any
from core.config import Config
import re

class PromptManager:
    """
    Manages prompt templates from the MARS paper (Appendix F).
    Loads prompts from text files and handles variable substitution with defaults.
    """
    
    def __init__(self, prompt_dir: Optional[Path] = None):
        self.prompt_dir = prompt_dir or Config.PROMPT_DIR
        self._prompts = {}
        self._load_all_prompts()
    
    def _load_all_prompts(self):
        """Load all prompt templates from files"""
        prompt_files = {
            # Task Preparation
            "metric_parsing": "metric_parsing.txt",
            "metadata_generation": "metadata_generation.txt",
            "validation_verification": "validation_verification.txt",
            "metadata_documentation": "metadata_documentation.txt",
            "eda": "eda.txt",
            "model_search": "model_search.txt",
            
            # Idea Generation & Modular Decomposition
            "initial_idea": "initial_idea.txt",
            "idea_improvement": "idea_improvement.txt",
            "modular_decomposition": "modular_decomposition.txt",
            
            # Implementation
            "module_implementation": "module_implementation.txt",
            "module_testing": "module_testing.txt",
            "solution_drafting": "solution_drafting.txt",
            "solution_improvement": "solution_improvement.txt",
            
            # Debugging
            "bug_analysis": "bug_analysis.txt",
            "debugging": "debugging.txt",
            
            # Review & Lessons
            "execution_review": "execution_review.txt",
            "empirical_analysis": "empirical_analysis.txt",  # Stage 1: Empirical Analysis
            "solution_lesson": "solution_lesson.txt",  # Stage 2: Lesson Distillation
            "debug_lesson": "debug_lesson.txt",
            "lesson_deduplication": "lesson_deduplication.txt",
        }
        
        for name, filename in prompt_files.items():
            filepath = self.prompt_dir / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    self._prompts[name] = f.read()
            else:
                print(f"Warning: Prompt file not found: {filepath}")
                self._prompts[name] = ""
    
    def get_prompt(self, name: str, **kwargs) -> str:
        """
        Get prompt template with variable substitution.
        Automatically provides sensible defaults for missing variables.
        
        Args:
            name: Name of the prompt template
            **kwargs: Variables to substitute in the template
            
        Returns:
            Formatted prompt string
        """
        if name not in self._prompts:
            raise ValueError(f"Prompt '{name}' not found")
        
        template = self._prompts[name]
        
        # Extract all variables from template
        required_vars = self._extract_variables(template)
        
        # Provide comprehensive defaults
        defaults = self._get_default_values()
        
        # Merge: defaults < kwargs (kwargs have priority)
        all_vars = {**defaults, **kwargs}
        
        # Check if we still have missing variables
        missing = required_vars - set(all_vars.keys())
        if missing:
            print(f"Warning: Missing variables for prompt '{name}': {missing}")
            # Provide empty string defaults for any still missing
            for var in missing:
                all_vars[var] = ""
        
        try:
            return template.format(**all_vars)
        except KeyError as e:
            # This should rarely happen now
            raise ValueError(f"Missing variable {e} for prompt '{name}'. Available: {list(all_vars.keys())}")
    
    def _extract_variables(self, template: str) -> set:
        """
        Extract variable names from template string.
        Ignores example JSON blocks and malformed variable names.
        
        Args:
            template: Template string with {variable} placeholders
            
        Returns:
            Set of valid variable names
        """
        # Find all {variable_name} patterns
        pattern = r'\{([^}]+)\}'
        matches = re.findall(pattern, template)
        
        # Filter out invalid variable names
        valid_vars = set()
        for match in matches:
            # Clean the match
            match = match.strip()
            
            # Skip if contains newline
            if '\n' in match or '\r' in match:
                continue
            
            # Skip if contains quotes (likely JSON example)
            if '"' in match or "'" in match:
                continue
            
            # Skip if contains colon (likely JSON)
            if ':' in match:
                continue
            
            # Skip if starts with special chars
            if not match or match[0] in (' ', '\t', '{', '}', '[', ']'):
                continue
            
            # Must be valid Python identifier
            # Allow alphanumeric + underscore
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', match):
                valid_vars.add(match)
        
        return valid_vars
    
    def _get_default_values(self) -> Dict[str, Any]:
        """
        Get default values for common prompt variables.
        
        Returns:
            Dictionary of default values
        """
        return {
            # File/Directory names
            "dir_name": "cache",
            "file_name": "module.py",
            "file_description": "Module implementation",
            
            # Execution parameters
            "exec_timeout": "1 hour",
            "timeout": "3600",
            
            # Code/Files
            "library_files": "No existing files.",
            "files": "No files provided.",
            "code": "",
            "previous_solution": "No previous solution.",
            "new_solution": "No new solution.",
            "best_solution": "No best solution yet.",
            
            # Lessons
            "lessons": "No lessons available yet.",
            "debug_lessons": "No debug lessons available yet.",
            "existing_lessons": "No existing lessons.",
            "new_lesson": "",
            
            # Ideas
            "idea": "",
            "previous_ideas": "No previous ideas.",
            
            # Descriptions
            "problem_description": "",
            "task_description": "",
            
            # Analysis/Reports
            "eda_report": "No EDA report available.",
            "model_arch_desc": "No model architectures provided.",
            
            # Execution results
            "exec_result": "",
            "term_out": "",
            "stdout": "",
            "stderr": "",
            "execution_error": "",
            "error_analysis": "",
            "error_message": "",
            
            # Metadata
            "metadata": "",
            "submission_cond": "",
            
            # Diffs
            "diff": "",
            "source_files": "",
            "source_exec_result": "",
            "source_error_analysis": "",
            "final_exec_result": "",

            # Two-stage lesson extraction (Mejora 2)
            "empirical_findings_text": "No empirical findings available.",
            "code_diff": "No code differences available.",
            "execution_output": "",
            "review_findings": "No review findings available.",
        }
    
    def list_prompts(self) -> list:
        """List all available prompt names"""
        return list(self._prompts.keys())
    
    def reload(self):
        """Reload all prompts from files"""
        self._prompts.clear()
        self._load_all_prompts()
    
    def get_required_variables(self, name: str) -> set:
        """
        Get required variables for a prompt.
        
        Args:
            name: Prompt name
            
        Returns:
            Set of required variable names
        """
        if name not in self._prompts:
            raise ValueError(f"Prompt '{name}' not found")
        
        return self._extract_variables(self._prompts[name])


# Global instance
_prompt_manager = None

def get_prompt_manager() -> PromptManager:
    """Get or create global PromptManager"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager