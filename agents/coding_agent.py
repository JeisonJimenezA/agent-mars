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
        return """You are an expert Python programmer specializing in ML engineering.

CRITICAL OUTPUT RULES:
1. Always wrap your code in ```python and ``` markers
2. Output COMPLETE, WORKING code - no placeholders or "..."
3. Use ONLY ASCII characters - NO emojis (✓, ✗, etc.)
4. For dataclasses: ALWAYS use field(default_factory=list) for mutable defaults
5. Import everything you use at the top of the file

Your code must be syntactically valid Python that can be executed directly."""
    
    def implement_module(
        self,
        problem_description: str,
        idea: str,
        module_name: str,
        module_description: str,
        existing_modules: Dict[str, str],
        lesson_pool: Optional[LessonPool] = None,
        eda_report: str = ""
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
                eda_report=eda_report,
                idea=idea,
                file_name=module_name,
                file_description=module_description,
                library_files=library_files,
            )
        except ValueError as e:
            self.log(f"Using fallback prompt: {e}")
            prompt = self._create_fallback_prompt(
                module_name, module_description, idea, library_files, problem_description
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

            # Debug logging
            self.debug_logger.log_code_generation(
                module_name=module_name,
                generated_code=code,
                is_valid=is_valid,
                syntax_error=error,
                agent_name=self.name
            )

            if is_valid:
                self.log(f"Generated {len(code)} chars of valid code")
                return code
            else:
                self.log(f"Syntax error: {error}")

                # Try basic fix first
                code_fixed = self._try_fix_syntax(code)
                if code_fixed:
                    is_valid, _ = self.parser.validate_syntax(code_fixed)
                    if is_valid:
                        self.log(f"Fixed syntax error, {len(code_fixed)} chars")
                        return code_fixed

                # Try asking LLM to fix the syntax error
                self.log("Attempting LLM-based syntax fix...")
                code_fixed = self._ask_llm_to_fix_syntax(code, error, module_name)
                if code_fixed:
                    is_valid, _ = self.parser.validate_syntax(code_fixed)
                    if is_valid:
                        self.log(f"LLM fixed syntax error, {len(code_fixed)} chars")
                        return code_fixed

        self.log("Failed to generate valid code")
        return None

    def _ask_llm_to_fix_syntax(self, code: str, error: str, module_name: str) -> Optional[str]:
        """Ask LLM to fix a syntax error in generated code."""
        fix_prompt = f"""The following Python code has a syntax error. Fix it and return ONLY the corrected code.

SYNTAX ERROR: {error}

CODE WITH ERROR:
```python
{code}
```

REQUIREMENTS:
- Fix ONLY the syntax error
- Return the COMPLETE fixed file
- Use ```python code block
- NO explanations, just the fixed code
"""

        try:
            response = self.call_llm(
                user_message=fix_prompt,
                temperature=0.3,
                max_tokens=Config.MAX_TOKENS
            )

            fixed_code = self._extract_code_from_response(response["content"])
            return fixed_code

        except Exception as e:
            self.log(f"LLM fix failed: {e}")
            return None
    
    def implement_main_script(
        self,
        problem_description: str,
        idea: str,
        modules: Dict[str, str],
        lesson_pool: Optional[LessonPool] = None,
        eda_report: str = ""
    ) -> Optional[str]:
        """Implement main orchestration script."""

        self.log("Implementing main script...")

        library_files = self._format_library_files(modules)

        try:
            prompt = self.prompt_manager.get_prompt(
                "solution_drafting",
                problem_description=problem_description,
                eda_report=eda_report,
                idea=idea,
                library_files=library_files,
                file_description="Main orchestration script that runs the full pipeline",
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
    
    def _format_library_files(self, modules: Dict[str, str], max_total_chars: int = 30000) -> str:
        """
        Format modules for prompt with smart truncation.

        For large codebases, includes only function/class signatures to save tokens.
        This prevents prompt overflow that causes truncated LLM responses.

        Args:
            modules: Dict of {filename: code}
            max_total_chars: Maximum total characters for all modules

        Returns:
            Formatted string with module contents or summaries
        """
        if not modules:
            return "No modules yet."

        formatted = []
        total_chars = 0

        for name, code in modules.items():
            remaining = max_total_chars - total_chars
            if remaining <= 0:
                formatted.append(f"=== {name} ===\n# (omitted - token limit)\n")
                continue

            if len(code) <= remaining:
                # Include full code
                formatted.append(f"=== {name} ===\n{code}\n")
                total_chars += len(code)
            else:
                # Include only signatures (imports + class/function definitions)
                summary = self._extract_signatures(code, max_chars=min(5000, remaining))
                formatted.append(f"=== {name} (signatures only) ===\n{summary}\n")
                total_chars += len(summary)

        return "\n".join(formatted)

    def _extract_signatures(self, code: str, max_chars: int = 5000) -> str:
        """
        Extract function and class signatures from code.

        Returns imports + class/function headers with docstrings,
        but not implementation details.
        """
        lines = code.split('\n')
        result_lines = []
        in_docstring = False
        docstring_char = None

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Always include imports
            if stripped.startswith(('import ', 'from ')):
                result_lines.append(line)
                continue

            # Include class and function definitions
            if stripped.startswith(('class ', 'def ')):
                result_lines.append(line)
                # Check for docstring on next line
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith('"""') or next_line.startswith("'''"):
                        result_lines.append(lines[i + 1])
                        # If docstring is on one line
                        if next_line.count('"""') == 2 or next_line.count("'''") == 2:
                            pass
                        else:
                            # Multi-line docstring - include until close
                            docstring_char = '"""' if '"""' in next_line else "'''"
                            for j in range(i + 2, min(i + 10, len(lines))):
                                result_lines.append(lines[j])
                                if docstring_char in lines[j]:
                                    break
                result_lines.append("        ...")  # Placeholder for implementation
                continue

            # Include decorators
            if stripped.startswith('@'):
                result_lines.append(line)
                continue

            # Include type hints and constants at module level
            if '=' in stripped and not stripped.startswith(' ') and not stripped.startswith('#'):
                if ':' in stripped.split('=')[0]:  # Type-annotated
                    result_lines.append(line)

        result = '\n'.join(result_lines)

        # Truncate if still too long
        if len(result) > max_chars:
            result = result[:max_chars] + "\n# ... (truncated)"

        return result
    
    def _create_fallback_prompt(
        self, module_name: str, description: str, idea: str, library: str, problem_description: str = ""
    ) -> str:
        """Create fallback prompt when template is missing."""
        return f"""
Problem: {problem_description[:500] if problem_description else 'ML competition task'}

Implement Python module: {module_name}

Description: {description}

Overall Idea: {idea}

Existing Modules:
{library}

DATA FILE LOCATIONS:
- Training data: ./train.csv (in the current working directory)
- Test data: ./test.csv (in the current working directory)
- Sample submission: ./sample_submission.csv
- Cache directory: ./working/cache/

Requirements:
- Python 3.9+ with type hints
- Use pandas, sklearn, numpy
- Import from existing modules where appropriate
- NO if __name__ == "__main__" block
- Clean, efficient code
- NO EMOJIS - use only ASCII characters
- NO PARQUET - use CSV for caching
- Use encoding='utf-8' when opening files

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

DATA FILE LOCATIONS:
- Training data: ./train.csv (in the current working directory)
- Test data: ./test.csv (in the current working directory)
- Sample submission: ./sample_submission.csv
- Metadata: ./metadata/

Requirements:
- Import and use modules
- Load data from ./train.csv and ./test.csv
- Train model and evaluate
- Print "Final Validation Metric: <value>"
- Save predictions to ./submission/submission.csv
- Executable as: python main.py

Provide complete code in ```python block.
"""
    
    def _extract_code_from_response(self, content: str) -> Optional[str]:
        """
        Extract code from LLM response.
        Handles multiple formats that different LLMs might produce.

        IMPORTANT: Includes fallback for UNCLOSED code blocks (when LLM
        runs out of tokens and doesn't close with ```).
        """
        import re

        if not content:
            return None

        # Strategy 1: Standard markdown ```python blocks (closed)
        patterns = [
            r'```python\s*\n(.*?)```',           # ```python\n...\n```
            r'```Python\s*\n(.*?)```',           # ```Python\n...\n```
            r'```py\s*\n(.*?)```',               # ```py\n...\n```
            r'```\s*\n(.*?)```',                 # ```\n...\n```
            r'<code>\s*\n?(.*?)</code>',         # <code>...</code>
            r'<python>\s*\n?(.*?)</python>',     # <python>...</python>
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if matches:
                # Return the largest match (most likely the full code)
                code = max(matches, key=len).strip()
                if code and len(code) > 50:
                    return code

        # Strategy 2: UNCLOSED code blocks (LLM ran out of tokens)
        # This is critical for handling truncated responses
        unclosed_patterns = [
            r'```python\s*\n(.+)$',              # ```python\n... (no closing)
            r'```Python\s*\n(.+)$',              # ```Python\n... (no closing)
            r'```py\s*\n(.+)$',                  # ```py\n... (no closing)
            r'```\s*\n(import\s.+)$',            # ``` followed by import (no closing)
        ]

        for pattern in unclosed_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                potential_code = match.group(1).strip()
                # Verify it's substantial and looks like Python
                if len(potential_code) > 200 and self._looks_like_python(potential_code):
                    self.log("WARNING: Extracted code from UNCLOSED block (LLM may have truncated)")
                    # Try to repair common truncation issues
                    repaired = self._repair_truncated_code(potential_code)
                    if repaired:
                        return repaired
                    return potential_code

        # Strategy 3: Find code between markers
        # Some LLMs use markers like "Here's the code:" followed by code
        code_start_markers = [
            r'(?:here\'?s?|the|complete|full|implementation|code)[\s\S]{0,50}?:\s*\n(.*)',
            r'```\s*(.*?)(?:```|\Z)',
        ]

        for pattern in code_start_markers:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                potential_code = match.group(1).strip()
                # Verify it looks like Python
                if self._looks_like_python(potential_code):
                    return potential_code

        # Strategy 4: Find lines that look like Python code
        lines = content.split('\n')
        code_lines = []
        in_code = False

        for line in lines:
            stripped = line.strip()

            # Skip markdown/explanation lines
            if stripped.startswith(('#', '>', '-', '*', '1.', '2.')) and not stripped.startswith('# '):
                if not in_code:
                    continue

            # Detect start of code
            if stripped.startswith(('import ', 'from ', 'def ', 'class ', '@', '"""', "'''")):
                in_code = True

            if in_code:
                # Stop at end markers
                if stripped.startswith('```') or stripped.startswith('---'):
                    break
                code_lines.append(line)

        if code_lines:
            code = '\n'.join(code_lines).strip()
            if self._looks_like_python(code):
                return code

        return None

    def _repair_truncated_code(self, code: str) -> Optional[str]:
        """
        Attempt to repair truncated code by closing open blocks.

        Common truncation issues:
        - Unclosed string literals
        - Unclosed parentheses/brackets
        - Incomplete function definitions

        Returns repaired code or None if repair not possible.
        """
        lines = code.split('\n')

        # Remove obviously incomplete last line
        while lines:
            last_line = lines[-1].strip()
            # Check for incomplete patterns
            if (last_line.endswith(('(', '[', '{', ',', ':', '=', '+', '-', '*', '/')) or
                last_line.count('"') % 2 == 1 or
                last_line.count("'") % 2 == 1):
                lines.pop()
            else:
                break

        if not lines:
            return None

        repaired = '\n'.join(lines)

        # Validate syntax
        is_valid, _ = self.parser.validate_syntax(repaired)
        if is_valid:
            return repaired

        # Try adding common missing closures
        closures_to_try = [
            '\n',                    # Just newline
            '\n    pass\n',          # Close function with pass
            '\n        pass\n',      # Close nested function
            ')\n',                   # Close parenthesis
            ']\n',                   # Close bracket
            '}\n',                   # Close brace
        ]

        for closure in closures_to_try:
            test_code = repaired + closure
            is_valid, _ = self.parser.validate_syntax(test_code)
            if is_valid:
                self.log(f"Repaired truncated code by adding: {repr(closure)}")
                return test_code

        # Return as-is if no repair worked (let caller handle)
        return repaired

    def _looks_like_python(self, code: str) -> bool:
        """Check if text looks like valid Python code."""
        if not code or len(code) < 20:
            return False

        # Must have Python constructs
        python_indicators = [
            'import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ',
            'return ', 'yield ', 'raise ', 'try:', 'with ', '= ', '()'
        ]
        has_python = any(ind in code for ind in python_indicators)

        # Should not be mostly prose
        lines = code.split('\n')
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        if not code_lines:
            return False

        # Check ratio of lines with code constructs
        construct_lines = sum(1 for l in code_lines if any(c in l for c in ['=', '(', ':', 'import', 'def', 'class']))
        ratio = construct_lines / len(code_lines) if code_lines else 0

        return has_python and ratio > 0.3
    
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