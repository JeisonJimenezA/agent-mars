# agents/coding_agent.py - ESTRUCTURA COMPLETA

from typing import Dict, Optional, List
from agents.base_agent import BaseAgent
from memory.lesson_pool import LessonPool
from memory.lesson_types import LessonType
from core.config import Config

class CodingAgent(BaseAgent):
    """Agent responsible for implementing code modules."""
    
    def __init__(self):
        super().__init__("CodingAgent")
    
    def get_system_prompt(self) -> str:
        return """You are an expert Python programmer specializing in ML engineering."""
    
    def implement_module(
        self,
        problem_description: str,
        idea: str,
        module_name: str,
        module_description: str,
        existing_modules: Dict[str, str],
        lesson_pool: Optional[LessonPool] = None
    ) -> Optional[str]:
        """Implement a single module."""
        
        self.log(f"Implementing module: {module_name}")
        
        # Format existing modules
        library_files = self._format_library_files(existing_modules)
        
        # Get prompt
        try:
            prompt = self.prompt_manager.get_prompt(
                "module_implementation",
                problem_description=problem_description,
                idea=idea,
                file_name=module_name,
                file_description=module_description,
                library_files=library_files,
            )
        except ValueError as e:
            self.log(f"Using fallback prompt")
            prompt = self._create_fallback_prompt(
                module_name, module_description, idea, library_files
            )
        
        # Call LLM - usar max tokens del modelo (8192)
        response = self.call_llm(
            user_message=prompt,
            temperature=0.7,
            max_tokens=Config.MAX_TOKENS
        )
        
        # Extract code
        code = self._extract_code_from_response(response["content"])
        
        if code:
            is_valid, error = self.parser.validate_syntax(code)
            if is_valid:
                self.log(f"Generated {len(code)} chars of valid code")
                return code
            else:
                self.log(f"Syntax error: {error}")
                code_fixed = self._try_fix_syntax(code)
                if code_fixed:
                    is_valid, _ = self.parser.validate_syntax(code_fixed)
                    if is_valid:
                        return code_fixed
        
        self.log("Failed to generate valid code")
        return None
    
    def implement_main_script(
        self,
        problem_description: str,
        idea: str,
        modules: Dict[str, str],
        lesson_pool: Optional[LessonPool] = None
    ) -> Optional[str]:
        """Implement main orchestration script."""
        
        self.log("Implementing main script...")
        
        library_files = self._format_library_files(modules)
        
        try:
            prompt = self.prompt_manager.get_prompt(
                "solution_drafting",
                problem_description=problem_description,
                idea=idea,
                library_files=library_files,
                file_description="Main orchestration script",
            )
        except ValueError:
            prompt = self._create_main_fallback_prompt(
                problem_description, idea, library_files
            )
        
        response = self.call_llm(
            user_message=prompt,
            temperature=0.7,
            max_tokens=Config.MAX_TOKENS
        )

        code = self._extract_code_from_response(response["content"])

        if code:
            is_valid, error = self.parser.validate_syntax(code)
            if is_valid:
                self.log(f"Generated main script ({len(code)} chars)")
                return code

        return None
    
    def generate_module_test(
        self,
        problem_description: str,
        idea: str,
        module_name: str,
        module_code: str,
        existing_modules: Dict[str, str],
    ) -> Optional[str]:
        """
        Generate a lightweight unit-test script for a module.
        Implements DebugModules from Algorithm 2 line 19.

        Returns Python code that imports and exercises the module,
        or None if generation fails.
        """
        self.log(f"Generating test for: {module_name}")

        library_files = self._format_library_files(existing_modules)

        try:
            prompt = self.prompt_manager.get_prompt(
                "module_testing",
                problem_description=problem_description,
                idea=idea,
                library_files=library_files,
            )
        except ValueError:
            prompt = self._create_module_test_fallback(
                module_name, module_code, library_files
            )

        response = self.call_llm(
            user_message=prompt,
            temperature=0.5,
            max_tokens=Config.MAX_TOKENS,
        )

        code = self._extract_code_from_response(response["content"])
        if code:
            is_valid, _ = self.parser.validate_syntax(code)
            if is_valid:
                self.log(f"Generated test ({len(code)} chars)")
                return code

        self.log("Failed to generate valid test code")
        return None

    def _create_module_test_fallback(
        self, module_name: str, module_code: str, library_files: str
    ) -> str:
        """Fallback prompt for module testing."""
        return f"""Write a short Python test script for the module below.
The test must:
- Import and instantiate the key classes/functions from the module
- Use minimal data (small subset, few rows) so it runs fast
- Include basic assertions to verify correctness
- Print "ALL TESTS PASSED" if everything works

Module ({module_name}):
```python
{module_code[:3000]}
```

Other available modules:
{library_files[:2000]}

Provide the test code in a ```python block.
"""
    
    def _format_library_files(self, modules: Dict[str, str]) -> str:
        """Format modules for prompt."""
        if not modules:
            return "No modules yet."
        
        formatted = []
        for name, code in modules.items():
            formatted.append(f"=== {name} ===\n{code}\n")
        
        return "\n".join(formatted)
    
    def _create_fallback_prompt(
        self, module_name: str, description: str, idea: str, library: str
    ) -> str:
        """Create fallback prompt when template is missing."""
        return f"""
Implement Python module: {module_name}

Description: {description}

Overall Idea: {idea}

Existing Modules:
{library}

Requirements:
- Python 3.9+ with type hints
- Use pandas, sklearn, numpy for Titanic
- Import from existing modules
- NO if __name__ == "__main__" block
- Clean, efficient code

Provide code in ```python block.
"""
    
    def _create_main_fallback_prompt(
        self, problem: str, idea: str, library: str
    ) -> str:
        """Create fallback for main script."""
        return f"""
Create main.py orchestration script.

Problem: {problem}

Solution: {idea}

Available Modules:
{library}

Requirements:
- Import and use modules
- Train model and evaluate
- Print "Final Validation Metric: <value>"
- Save predictions to ./submission/submission.csv
- Executable as: python main.py

Provide complete code in ```python block.
"""
    
    def _extract_code_from_response(self, content: str) -> Optional[str]:
        """Extract code from LLM response."""
        import re
        
        # Try markdown blocks
        for pattern in [r'```python\s*\n(.*?)```', r'```\s*\n(.*?)```']:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Try finding Python code
        lines = content.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ', 'def ', 'class ')):
                in_code = True
            if in_code:
                code_lines.append(line)
        
        if code_lines:
            code = '\n'.join(code_lines).strip()
            if any(kw in code for kw in ['import ', 'def ', 'class ']):
                return code
        
        return None
    
    def _try_fix_syntax(self, code: str) -> Optional[str]:
        """Try to fix common syntax errors."""
        code = code.strip('`').strip()
        
        if code.startswith('python\n'):
            code = code[7:]
        
        lines = code.split('\n')
        if not lines:
            return None
        
        min_indent = min(
            (len(line) - len(line.lstrip()) for line in lines if line.strip()),
            default=0
        )
        
        if min_indent > 0:
            code = '\n'.join(
                line[min_indent:] if len(line) >= min_indent else line
                for line in lines
            )
        
        return code