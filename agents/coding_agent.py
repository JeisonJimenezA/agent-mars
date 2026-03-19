# agents/coding_agent.py - ESTRUCTURA COMPLETA

import os
from typing import Dict, Optional, List
from agents.base_agent import BaseAgent
from memory.lesson_pool import LessonPool
from memory.lesson_types import LessonType
from core.config import Config

# Stream code generation to stdout by default when using Ollama.
# Set MARS_STREAM=false in .env to disable.
_STREAM = os.getenv("MARS_STREAM", "true").lower() != "false"

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
        eda_report: str = "",
        metric_name: str = "",
        data_schema: str = "",
    ) -> Optional[str]:
        """Implement a single module."""

        self.log(f"Implementing module: {module_name}")

        # Format existing modules
        library_files = self._format_library_files(existing_modules)

        # Extract lessons
        lessons_text = ""
        if lesson_pool:
            lessons_text = lesson_pool.format_for_prompt(
                lesson_type=LessonType.SOLUTION, k=10
            )

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
                lessons=lessons_text,
                metric_name=metric_name or "unknown",
                data_schema=data_schema,
            )
        except ValueError as e:
            self.log(f"Using fallback prompt: {e}")
            prompt = self._create_fallback_prompt(
                module_name, module_description, idea, library_files, problem_description
            )
        
        response = self.call_llm(
            user_message=prompt,
            temperature=0.2,
            max_tokens=Config.MAX_TOKENS,
            stream=_STREAM,
        )

        raw = response["content"]
        code, truncated = self._extract_with_truncation_check(raw)

        # Continuation loop if truncated
        MAX_CONTINUATIONS = 3
        for cont in range(MAX_CONTINUATIONS):
            if not truncated:
                break
            self.log(f"Module {module_name} truncated — continuation {cont + 1}/{MAX_CONTINUATIONS}")
            extra, truncated = self._continue_code(code)
            if extra:
                code = code.rstrip() + "\n" + extra.lstrip()

        if not code:
            self.log("Failed to generate valid code")
            return None

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

        self.log(f"Syntax error: {error}")

        code_fixed = self._try_fix_syntax(code)
        if code_fixed:
            is_valid, _ = self.parser.validate_syntax(code_fixed)
            if is_valid:
                self.log(f"Fixed syntax error, {len(code_fixed)} chars")
                return code_fixed

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
                temperature=0,
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
        eda_report: str = "",
        metric_name: str = "",
        data_schema: str = "",
    ) -> Optional[str]:
        """Implement main orchestration script."""

        self.log("Implementing main script...")

        library_files = self._format_library_files(modules)

        lessons_text = ""
        if lesson_pool:
            lessons_text = lesson_pool.format_for_prompt(
                lesson_type=LessonType.SOLUTION, k=10
            )

        try:
            prompt = self.prompt_manager.get_prompt(
                "solution_drafting",
                problem_description=problem_description,
                eda_report=eda_report,
                idea=idea,
                library_files=library_files,
                file_description="Main orchestration script that runs the full pipeline",
                lessons=lessons_text,
                metric_name=metric_name or "unknown",
                data_schema=data_schema,
            )
        except ValueError:
            prompt = self._create_main_fallback_prompt(
                problem_description, idea, library_files
            )

        response = self.call_llm(
            user_message=prompt,
            temperature=0.2,
            max_tokens=Config.MAX_TOKENS,
            stream=_STREAM,
        )

        code, truncated = self._extract_with_truncation_check(response["content"])

        for cont in range(3):
            if not truncated:
                break
            self.log(f"main.py truncated — continuation {cont + 1}/3")
            extra, truncated = self._continue_code(code)
            if extra:
                code = code.rstrip() + "\n" + extra.lstrip()

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
        data_schema: str = "",
        eda_report: str = "",
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
                data_schema=data_schema,
                eda_report=eda_report,
            )
        except ValueError:
            prompt = self._create_module_test_fallback(
                module_name, module_code, library_files
            )

        response = self.call_llm(
            user_message=prompt,
            temperature=0.2,
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
    
    def _format_library_files(self, modules: Dict[str, str], max_total_chars: int = 80000) -> str:
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

        Returns imports + FULL function signatures (including multi-line
        parameter lists and return type annotations) + docstrings.
        This gives dependent modules enough context to call functions correctly.
        """
        lines = code.split('\n')
        result_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Always include imports
            if stripped.startswith(('import ', 'from ')):
                result_lines.append(line)
                i += 1
                continue

            # Include decorators
            if stripped.startswith('@'):
                result_lines.append(line)
                i += 1
                continue

            # Include class and function definitions — capture FULL signature
            # (handles multi-line signatures like def foo(\n    x: int,\n    y: str\n) -> bool:)
            if stripped.startswith(('class ', 'def ')):
                sig_lines = [line]
                j = i + 1
                # Continue collecting signature lines until we find the `:` that ends it
                # A signature ends when there's a `:` at the end of the stripped line
                # that is NOT inside parentheses
                paren_depth = stripped.count('(') - stripped.count(')')
                while paren_depth > 0 and j < len(lines):
                    sig_lines.append(lines[j])
                    paren_depth += lines[j].count('(') - lines[j].count(')')
                    j += 1

                result_lines.extend(sig_lines)

                # Capture docstring (up to 15 lines)
                if j < len(lines):
                    next_stripped = lines[j].strip()
                    if next_stripped.startswith('"""') or next_stripped.startswith("'''"):
                        quote = '"""' if '"""' in next_stripped else "'''"
                        result_lines.append(lines[j])
                        # Single-line docstring
                        if next_stripped.count(quote) >= 2:
                            j += 1
                        else:
                            # Multi-line: include until closing quotes (max 15 lines)
                            j += 1
                            lines_captured = 0
                            while j < len(lines) and lines_captured < 15:
                                result_lines.append(lines[j])
                                if quote in lines[j]:
                                    j += 1
                                    break
                                j += 1
                                lines_captured += 1

                result_lines.append(lines[i][:len(lines[i]) - len(lines[i].lstrip())] + "    ...")
                i = j
                continue

            # Include module-level type-annotated constants
            if '=' in stripped and not stripped.startswith((' ', '\t', '#')):
                if ':' in stripped.split('=')[0]:
                    result_lines.append(line)

            i += 1

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

DATA FILE LOCATIONS (use environment variables injected by the framework):
import os
DATA_DIR = os.environ.get('DATA_DIR', '.')
METADATA_DIR = os.environ.get('METADATA_DIR', './metadata')
- Full training set:  os.path.join(DATA_DIR, 'train.csv')
- Test set:           os.path.join(DATA_DIR, 'test.csv')
- Train split (80%):  os.path.join(METADATA_DIR, 'train.csv')
- Val split (20%):    os.path.join(METADATA_DIR, 'val.csv')
- Sample submission:  os.path.join(DATA_DIR, 'sample_submission.csv')
- Cache directory:    ./working/cache/

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

DATA FILE LOCATIONS (use environment variables injected by the framework):
import os
DATA_DIR = os.environ.get('DATA_DIR', '.')
METADATA_DIR = os.environ.get('METADATA_DIR', './metadata')
- Full training set:  os.path.join(DATA_DIR, 'train.csv')
- Test set:           os.path.join(DATA_DIR, 'test.csv')
- Train split (80%):  os.path.join(METADATA_DIR, 'train.csv')
- Val split (20%):    os.path.join(METADATA_DIR, 'val.csv')
- Sample submission:  os.path.join(DATA_DIR, 'sample_submission.csv')

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

    def _extract_with_truncation_check(self, raw: str):
        """
        Extract code from response and detect whether the model was truncated.
        Returns (code, is_truncated).
        """
        import re as _re
        if not raw:
            return None, False

        # Clean closed block → not truncated
        closed = _re.search(r'```python\s*\n(.*?)```', raw, _re.DOTALL | _re.IGNORECASE)
        if closed:
            return closed.group(1).strip(), False

        # Unclosed block → truncated
        unclosed = _re.search(r'```python\s*\n(.+)$', raw, _re.DOTALL | _re.IGNORECASE)
        if unclosed:
            code = unclosed.group(1).strip()
            if len(code) > 50:
                return code, True

        # Fallback extractor
        code = self._extract_code_from_response(raw)
        if code:
            return code, not raw.rstrip().endswith("```")

        return None, False

    def _continue_code(self, existing_code: str):
        """
        Send a continuation request for truncated code.
        Passes the FULL existing code so the model has complete context.
        Returns (extra_code, still_truncated).
        """
        prompt = f"""The Python script below was cut off mid-generation due to output length limits.
Continue writing from EXACTLY where it stopped. Do NOT repeat any code already shown.
Write only what comes next, ending with ``` when the script is fully complete.

FULL CODE SO FAR:
```python
{existing_code}
```

Continue the script from the next line:
```python
"""
        response = self.call_llm(
            user_message=prompt,
            temperature=0.1,
            max_tokens=Config.MAX_TOKENS,
            stream=_STREAM,
        )
        return self._extract_with_truncation_check(response["content"])

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