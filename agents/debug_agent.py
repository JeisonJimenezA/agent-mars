# agents/debug_agent.py
from typing import Dict, Optional, Tuple, List
from agents.base_agent import BaseAgent
from memory.lesson_pool import LessonPool
from memory.lesson_types import LessonType
from execution.diff_editor import DiffEdit, DiffEditor
from core.config import Config
import re
import ast

class DebugAgent(BaseAgent):
    """
    Agent responsible for analyzing and fixing errors.
    Implements debugging workflow from Section 4.4.1.
    
    NOW USES: Diff-based editing for atomic fixes.
    """
    
    def __init__(self):
        super().__init__("DebugAgent")
        self.diff_editor = DiffEditor()  # NEW: Diff-based editing
    
    def get_system_prompt(self) -> str:
        return """You are an expert Python debugger specializing in ML code.

CRITICAL OUTPUT RULES:
1. When asked to fix code, wrap your code in ```python and ``` markers
2. Output COMPLETE files when regenerating - no partial code or "..."
3. Use ONLY ASCII characters - NO emojis
4. For dataclasses: ALWAYS use field(default_factory=list) for mutable defaults
5. Make MINIMAL changes to fix the error - don't rewrite unrelated code

You excel at diagnosing errors and proposing targeted fixes."""
    
    def analyze_error(
        self,
        problem_description: str,
        files: Dict[str, str],
        execution_error: str,
        lesson_pool: Optional[LessonPool] = None
    ) -> Optional[str]:
        """
        Analyze execution error and provide diagnosis.
        
        Args:
            problem_description: Task description
            files: Solution files {name: code}
            execution_error: Error traceback
            lesson_pool: Pool of debug lessons
            
        Returns:
            Error analysis text
        """
        self.log("Analyzing execution error...")
        
        # Format files
        files_text = self._format_files(files)
        
        # Get debug lessons
        debug_lessons = ""
        if lesson_pool:
            debug_lessons = lesson_pool.format_for_prompt(
                lesson_type=LessonType.DEBUG,
                k=10
            )
        
        # Get prompt
        prompt = self.prompt_manager.get_prompt(
            "bug_analysis",
            problem_description=problem_description,
            files=files_text,
            exec_result=execution_error,
            lessons=debug_lessons if debug_lessons else "No debug lessons yet."
        )
        
        # Call LLM - usar max tokens del modelo (8192)
        response = self.call_llm(
            user_message=prompt,
            temperature=0,
            max_tokens=Config.MAX_TOKENS
        )
        
        analysis = response["content"].strip()
        
        self.log(f"Error analysis complete ({len(analysis)} chars)")
        
        return analysis
    
    def fix_error(
        self,
        problem_description: str,
        files: Dict[str, str],
        execution_error: str,
        error_analysis: str,
        lesson_pool: Optional[LessonPool] = None
    ) -> Optional[Dict[str, str]]:
        """
        Generate fix for the error using DIFF-BASED EDITING.

        Returns modified files.

        Args:
            problem_description: Task description
            files: Current solution files
            execution_error: Error traceback
            error_analysis: Analysis from analyze_error
            lesson_pool: Pool of debug lessons

        Returns:
            Modified files dict or None
        """
        self.log("=" * 60)
        self.log("Generating error fix (diff-based)...")
        self.log(f"  Files to fix: {list(files.keys())}")
        self.log(f"  Total code size: {sum(len(c) for c in files.values())} chars")

        # Format files
        files_text = self._format_files(files)

        # Get debug lessons
        debug_lessons = ""
        if lesson_pool:
            debug_lessons = lesson_pool.format_for_prompt(
                lesson_type=LessonType.DEBUG,
                k=10
            )

        # Get prompt with DIFF instructions
        prompt = self._create_diff_debugging_prompt(
            problem_description,
            files_text,
            execution_error,
            error_analysis,
            debug_lessons
        )

        self.log(f"  Prompt size: {len(prompt)} chars")

        # Call LLM
        response = self.call_llm(
            user_message=prompt,
            temperature=0.1,
            max_tokens=Config.MAX_TOKENS
        )

        response_content = response.get("content", "")
        self.log(f"  Response size: {len(response_content)} chars")

        # Check if response looks truncated
        if len(response_content) > 0 and not response_content.rstrip().endswith(('```', '</fix>', '</new_code>', '}')):
            self.log("  WARNING: Response may be truncated (no proper ending)")

        # Parse diff-based fixes
        self.log("Parsing diff fixes...")
        success, modified_files, errors = self._parse_diff_fixes(
            response_content,
            files
        )

        if success:
            changes = sum(1 for f in modified_files if modified_files[f] != files.get(f))
            self.log(f"SUCCESS: Applied {changes} fix(es) via diff")
            return modified_files
        else:
            self.log(f"Diff parsing failed: {errors}")
            self.log("Attempting fallback strategies...")

            # Fallback 1: Try pattern-based extraction
            self.log("Fallback 1: Pattern-based extraction")
            modified_files = self._parse_fixed_files_fallback(response_content, files)
            if modified_files:
                changes = sum(1 for f in modified_files if modified_files[f] != files.get(f))
                if changes > 0:
                    self.log(f"Fallback 1 SUCCESS: {changes} file(s) modified")
                    return modified_files

            # Fallback 2: Try file regeneration for the error file
            self.log("Fallback 2: File regeneration")
            target_file = self._extract_error_file(execution_error)
            self.log(f"  Target file from error: {target_file}")

            if target_file and target_file in files:
                regenerated = self.regenerate_file(
                    target_file=target_file,
                    original_code=files[target_file],
                    execution_error=execution_error,
                    error_analysis=error_analysis,
                )
                if regenerated:
                    result = dict(files)
                    result[target_file] = regenerated
                    self.log(f"Fallback 2 SUCCESS: regenerated {target_file}")
                    return result

            self.log("All fallbacks FAILED")
            self.log("-" * 40)
            self.log("LLM Response (first 800 chars):")
            self.log(response_content[:800])
            self.log("-" * 40)
            return None
    
    def _create_diff_debugging_prompt(
        self,
        problem_description: str,
        files_text: str,
        execution_error: str,
        error_analysis: str,
        debug_lessons: str
    ) -> str:
        """
        Create prompt that instructs LLM to use diff-based fixes.

        OPTIMIZED STRUCTURE (based on MARS paper Appendix F):
        1. Instructions and format FIRST (LLMs weight beginning more)
        2. Few-shot examples
        3. Context (error, analysis, code)
        4. Final task reminder
        """
        # Extract only relevant files (files mentioned in error + main.py)
        relevant_files = self._get_relevant_context(files_text, execution_error)

        prompt = f"""# PYTHON DEBUG TASK

## RESPONSE FORMAT (MUST FOLLOW EXACTLY)

You will output 1-3 fixes using this EXACT XML structure:

<fix>
<file>filename.py</file>
<description>What this fixes (1 line)</description>
<old_code>
EXACT code from file (copy-paste, preserve indentation)
</old_code>
<new_code>
Fixed code (same indentation as old_code)
</new_code>
</fix>

## RULES (CRITICAL - READ FIRST)

1. <old_code> must match EXACTLY (same spaces, same indentation)
2. Fixes can be any size needed - from 1 line to entire functions/classes
3. <old_code> must appear exactly ONCE in the file
4. NO explanation text outside the <fix> tags
5. If a function needs major changes, replace the ENTIRE function

## EXAMPLES

Example 1 - Fix import error:
<fix>
<file>model.py</file>
<description>Add missing numpy import</description>
<old_code>
import pandas as pd
from sklearn.model_selection import train_test_split
</old_code>
<new_code>
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
</new_code>
</fix>

Example 2 - Fix attribute error:
<fix>
<file>trainer.py</file>
<description>Fix undefined variable n_classes</description>
<old_code>
        output_dim = n_classes
</old_code>
<new_code>
        output_dim = self.config.n_classes
</new_code>
</fix>

Example 3 - Fix dataclass mutable default:
<fix>
<file>config.py</file>
<description>Use field() for mutable default</description>
<old_code>
@dataclass
class ModelConfig:
    layers: List[int] = [64, 32]
</old_code>
<new_code>
@dataclass
class ModelConfig:
    layers: List[int] = field(default_factory=lambda: [64, 32])
</new_code>
</fix>

---

## ERROR TO FIX

```
{execution_error[:3000]}
```

## ERROR ANALYSIS

{error_analysis[:1500]}

## RELEVANT CODE (files related to error)

{relevant_files}

## PROBLEM CONTEXT

{problem_description[:1000]}

## DEBUG LESSONS FROM PREVIOUS ATTEMPTS

{debug_lessons if debug_lessons else "No previous lessons."}

---

Now output your fix(es) using the EXACT XML format shown above. Start with <fix>:
"""

        return prompt

    def _get_relevant_context(
        self,
        files_text: str,
        execution_error: str,
        max_chars: int = 30000
    ) -> str:
        """
        Extract only files relevant to the error, reducing context size.

        Priority:
        1. Files mentioned in traceback (highest priority)
        2. main.py (usually the entry point)
        3. config.py (common dependency)
        4. Other files (truncated if needed)

        Args:
            files_text: Full formatted files text
            execution_error: Error traceback
            max_chars: Maximum characters to return (default 30KB)

        Returns:
            Reduced context with only relevant files
        """
        # Extract filenames from error traceback
        error_files = set(re.findall(r'File\s+"[^"]*[\\/](\w+\.py)"', execution_error))
        # Also check for module-style references like "in module_name"
        error_files.update(re.findall(r'in\s+(\w+)', execution_error))

        # Parse files from files_text (format: === filename.py ===\n<code>\n)
        file_pattern = r'===\s*(\S+\.py)\s*===\n(.*?)(?====\s*\S+\.py\s*===|\Z)'
        files_dict = {}
        for match in re.finditer(file_pattern, files_text, re.DOTALL):
            filename = match.group(1)
            code = match.group(2).strip()
            files_dict[filename] = code

        if not files_dict:
            # Fallback: return truncated original
            return files_text[:max_chars]

        # Categorize files by priority
        priority_files = []  # Files in error traceback
        secondary_files = []  # main.py, config.py
        other_files = []  # Everything else

        for filename, code in files_dict.items():
            basename = filename.replace('.py', '')
            if filename in error_files or basename in error_files:
                priority_files.append((filename, code))
            elif filename in ('main.py', 'config.py'):
                secondary_files.append((filename, code))
            else:
                other_files.append((filename, code))

        # Build result respecting max_chars
        result_parts = []
        current_size = 0

        # Add priority files first (always include, but may truncate)
        for filename, code in priority_files:
            file_text = f"=== {filename} (ERROR HERE) ===\n{code}\n"
            if current_size + len(file_text) <= max_chars:
                result_parts.append(file_text)
                current_size += len(file_text)
            else:
                # Truncate this file but still include it
                remaining = max_chars - current_size - 200  # Reserve space for header
                if remaining > 500:
                    truncated = code[:remaining] + "\n# ... (truncated for context limit)"
                    result_parts.append(f"=== {filename} (ERROR HERE, truncated) ===\n{truncated}\n")
                    current_size = max_chars
                break

        # Add secondary files if space permits
        for filename, code in secondary_files:
            file_text = f"=== {filename} ===\n{code}\n"
            if current_size + len(file_text) <= max_chars:
                result_parts.append(file_text)
                current_size += len(file_text)

        # Add other files (signatures only if space is tight)
        for filename, code in other_files:
            remaining = max_chars - current_size
            if remaining < 500:
                break

            if len(code) <= remaining - 100:
                result_parts.append(f"=== {filename} ===\n{code}\n")
                current_size += len(code) + 50
            else:
                # Include only function/class signatures
                signatures = self._extract_signatures_quick(code)
                if len(signatures) < remaining - 100:
                    result_parts.append(f"=== {filename} (signatures only) ===\n{signatures}\n")
                    current_size += len(signatures) + 50

        if not result_parts:
            return files_text[:max_chars]

        return "\n".join(result_parts)

    def _extract_signatures_quick(self, code: str, max_lines: int = 30) -> str:
        """
        Quickly extract function/class signatures from code.
        Used to include context without full implementation.
        """
        lines = code.split('\n')
        result = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            # Include imports
            if stripped.startswith(('import ', 'from ')):
                result.append(line)
            # Include class/function definitions
            elif stripped.startswith(('class ', 'def ', '@')):
                result.append(line)
                # Include docstring if present
                if i + 1 < len(lines) and '"""' in lines[i + 1]:
                    result.append(lines[i + 1])

            if len(result) >= max_lines:
                result.append("# ... more definitions")
                break

        return '\n'.join(result)
    
    def _parse_diff_fixes(
        self,
        response: str,
        original_files: Dict[str, str]
    ) -> Tuple[bool, Dict[str, str], List[str]]:
        """
        Parse diff-based fixes from LLM response.
        Supports both XML-structured format and legacy formats.

        Returns:
            Tuple of (success, modified_files, errors)
        """
        diff_edits = []
        errors = []

        # ═══════════════════════════════════════════════════════════════════
        # Try XML-structured format first (most reliable)
        # ═══════════════════════════════════════════════════════════════════
        xml_pattern = r'<fix>\s*(.*?)\s*</fix>'
        xml_fixes = re.findall(xml_pattern, response, re.DOTALL | re.IGNORECASE)

        if xml_fixes:
            self.log(f"Parsing {len(xml_fixes)} XML-structured fix(es)")
            for i, fix_block in enumerate(xml_fixes):
                # Extract file
                file_match = re.search(r'<file>\s*(.*?)\s*</file>', fix_block, re.DOTALL)
                if not file_match:
                    self.log(f"  Fix {i+1}: no <file> tag found")
                    continue

                target_file = file_match.group(1).strip()
                if target_file not in original_files:
                    self.log(f"  Fix {i+1}: file '{target_file}' not in solution")
                    continue

                # Extract description (optional)
                desc_match = re.search(r'<description>\s*(.*?)\s*</description>', fix_block, re.DOTALL)
                description = desc_match.group(1).strip() if desc_match else "Debug fix"

                # Extract old_code and new_code
                old_match = re.search(r'<old_code>\s*\n?(.*?)\n?\s*</old_code>', fix_block, re.DOTALL)
                new_match = re.search(r'<new_code>\s*\n?(.*?)\n?\s*</new_code>', fix_block, re.DOTALL)

                if not old_match or not new_match:
                    self.log(f"  Fix {i+1}: missing <old_code> or <new_code>")
                    continue

                old_code = old_match.group(1)
                new_code = new_match.group(1)

                # Minimal cleaning - preserve indentation
                old_clean = self._clean_code_snippet_minimal(old_code)
                new_clean = self._clean_code_snippet_minimal(new_code)

                if old_clean and new_clean is not None:  # new_clean can be empty for deletions
                    self.log(f"  Fix {i+1}: {target_file} - {description[:50]}")
                    self.log(f"    OLD ({len(old_clean)} chars): {old_clean[:80]}...")
                    self.log(f"    NEW ({len(new_clean)} chars): {new_clean[:80]}...")

                    diff_edit = DiffEdit(
                        file_path=target_file,
                        old_str=old_clean,
                        new_str=new_clean,
                        description=description
                    )
                    diff_edits.append(diff_edit)

        # ═══════════════════════════════════════════════════════════════════
        # Fallback: Legacy format parsing
        # ═══════════════════════════════════════════════════════════════════
        if not diff_edits:
            self.log("No XML fixes found, trying legacy format")
            fixes = self._extract_fix_blocks(response)

            if not fixes:
                return False, {}, ["No fixes found in response"]

            for fix in fixes:
                target_file = self._extract_target_file(fix)
                if not target_file or target_file not in original_files:
                    continue

                old_new_pairs = self._extract_old_new_pairs(fix)

                for old_code, new_code in old_new_pairs:
                    old_clean = self._clean_code_snippet(old_code)
                    new_clean = self._clean_code_snippet(new_code)

                    if old_clean and new_clean:
                        diff_edit = DiffEdit(
                            file_path=target_file,
                            old_str=old_clean,
                            new_str=new_clean,
                            description="Debug fix from error analysis"
                        )
                        diff_edits.append(diff_edit)

        if not diff_edits:
            return False, {}, ["Failed to parse any valid diffs"]

        self.log(f"Applying {len(diff_edits)} diff edit(s)")

        # Apply diffs
        success, modified_files, apply_errors = self.diff_editor.apply_multiple_edits(
            original_files,
            diff_edits
        )
        errors.extend(apply_errors)

        # Validate that resulting files are still valid Python
        if success:
            for filename, code in modified_files.items():
                if filename.endswith('.py') and not self._is_valid_python_code(code):
                    self.log(f"  Diff resulted in invalid Python: {filename}")
                    # Revert to original
                    modified_files[filename] = original_files.get(filename, code)
                    errors.append(f"Diff for {filename} produced invalid Python")

            # Check if any files were actually modified successfully
            actually_modified = sum(1 for f in modified_files
                                   if modified_files[f] != original_files.get(f))
            if actually_modified == 0:
                return False, {}, errors + ["No valid modifications applied"]

        return success, modified_files, errors

    def _clean_code_snippet_minimal(self, code: str) -> str:
        """
        Minimal cleaning for XML-extracted code - just strip leading/trailing whitespace.
        Preserves internal indentation exactly as provided.
        """
        if not code:
            return ""

        # Remove only the common leading whitespace (dedent)
        lines = code.split('\n')

        # Find minimum indentation (ignoring empty lines)
        non_empty_lines = [l for l in lines if l.strip()]
        if not non_empty_lines:
            return ""

        min_indent = min(len(l) - len(l.lstrip()) for l in non_empty_lines)

        # Remove the common indent
        dedented = []
        for line in lines:
            if line.strip():
                dedented.append(line[min_indent:] if len(line) >= min_indent else line)
            else:
                dedented.append('')

        # Strip leading/trailing empty lines
        while dedented and not dedented[0].strip():
            dedented.pop(0)
        while dedented and not dedented[-1].strip():
            dedented.pop()

        return '\n'.join(dedented)
    
    def _extract_fix_blocks(self, response: str) -> List[str]:
        """Extract fix blocks from response"""
        # Split by fix markers
        pattern = r'\*\*Fix \d+:'
        parts = re.split(pattern, response)
        
        # Skip first part (preamble)
        fixes = [part.strip() for part in parts[1:] if part.strip()]
        
        return fixes
    
    def _extract_target_file(self, text: str) -> Optional[str]:
        """Extract target filename from fix block"""
        patterns = [
            r'Target File:\s*([^\n]+)',
            r'In\s+([^\s,]+\.py)',
            r'File:\s*([^\n]+)',
            r'# In\s+([^\s,]+\.py)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                filename = match.group(1).strip()
                filename = filename.replace('[', '').replace(']', '').strip()
                return filename
        
        return None
    
    def _extract_old_new_pairs(self, text: str) -> List[Tuple[str, str]]:
        """Extract OLD/NEW code pairs - handles multiple LLM response formats"""
        pairs = []

        # ═══════════════════════════════════════════════════════════════════
        # PATTERN 0 (PRIMARY): XML-like structured format (most reliable)
        # <old_code>...</old_code> <new_code>...</new_code>
        # ═══════════════════════════════════════════════════════════════════
        pattern_xml = r'<old_code>\s*\n?(.*?)\n?\s*</old_code>\s*\n?\s*<new_code>\s*\n?(.*?)\n?\s*</new_code>'
        matches_xml = re.findall(pattern_xml, text, re.DOTALL | re.IGNORECASE)
        if matches_xml:
            self.log(f"  XML format: found {len(matches_xml)} fix(es)")
            pairs.extend(matches_xml)
            # If we found XML matches, prioritize them and return early
            if pairs:
                return pairs

        # ═══════════════════════════════════════════════════════════════════
        # LEGACY PATTERNS (fallbacks for older responses)
        # ═══════════════════════════════════════════════════════════════════

        # Pattern 1: # OLD: ... # NEW: ... (with flexible spacing)
        pattern1 = r'#\s*OLD:?\s*\n(.*?)\n\n?#\s*NEW:?\s*\n(.*?)(?:\n```|```|\n\n|\Z)'
        matches1 = re.findall(pattern1, text, re.DOTALL | re.IGNORECASE)
        if matches1:
            self.log(f"  Legacy pattern 1 (# OLD/NEW): found {len(matches1)} match(es)")
            pairs.extend(matches1)

        # Pattern 2: OLD CODE / NEW CODE in markdown blocks
        pattern2 = r'```python\s*\n#\s*OLD:?\s*\n(.*?)\n\s*#\s*NEW:?\s*\n(.*?)\n```'
        matches2 = re.findall(pattern2, text, re.DOTALL | re.IGNORECASE)
        if matches2:
            self.log(f"  Legacy pattern 2 (code block): found {len(matches2)} match(es)")
            pairs.extend(matches2)

        # Pattern 3: Before/After format
        pattern3 = r'#?\s*(?:BEFORE|Original):?\s*\n(.*?)\n\s*#?\s*(?:AFTER|Fixed|Replacement):?\s*\n(.*?)(?:\n```|\n#|\Z)'
        matches3 = re.findall(pattern3, text, re.DOTALL | re.IGNORECASE)
        if matches3:
            self.log(f"  Legacy pattern 3 (BEFORE/AFTER): found {len(matches3)} match(es)")
            pairs.extend(matches3)

        # Pattern 4: **OLD:** / **NEW:** markdown bold headers
        pattern4 = r'\*\*OLD:?\*\*\s*\n?```python\n(.*?)\n```\s*\*\*NEW:?\*\*\s*\n?```python\n(.*?)\n```'
        matches4 = re.findall(pattern4, text, re.DOTALL | re.IGNORECASE)
        if matches4:
            self.log(f"  Legacy pattern 4 (bold headers): found {len(matches4)} match(es)")
            pairs.extend(matches4)

        # Pattern 5: Simple arrow format (old → new or old -> new)
        pattern5 = r'`([^`]+)`\s*(?:→|->|=>)\s*`([^`]+)`'
        matches5 = re.findall(pattern5, text)
        if matches5:
            self.log(f"  Legacy pattern 5 (arrow): found {len(matches5)} match(es)")
            pairs.extend(matches5)

        # Pattern 6: Inline code replacement
        pattern6 = r'OLD:\s*```(?:python)?\n?(.*?)\n?```\s*NEW:\s*```(?:python)?\n?(.*?)\n?```'
        matches6 = re.findall(pattern6, text, re.DOTALL | re.IGNORECASE)
        if matches6:
            self.log(f"  Legacy pattern 6 (inline): found {len(matches6)} match(es)")
            pairs.extend(matches6)

        if not pairs:
            self.log("  No OLD/NEW patterns found in response")

        return pairs
    
    def _clean_code_snippet(self, code: str) -> str:
        """Clean code snippet for diff matching"""
        # Remove markdown code blocks
        code = re.sub(r'```python|```', '', code).strip()

        # Remove leading comment markers but preserve indentation
        lines = code.split('\n')
        cleaned_lines = []

        # Markers to skip entirely (these are instruction comments, not code)
        skip_patterns = [
            r'^#\s*In\s+\w+\.py',  # "# In filename.py"
            r'^#\s*OLD:?',         # "# OLD:"
            r'^#\s*NEW:?',         # "# NEW:"
            r'^#\s*around line',   # "# around line X:"
        ]

        for line in lines:
            stripped = line.strip()

            # Skip instruction-only comment lines
            should_skip = False
            for pattern in skip_patterns:
                if re.match(pattern, stripped, re.IGNORECASE):
                    should_skip = True
                    break

            if should_skip:
                continue

            # Skip pure comment-only lines (no code content)
            if stripped.startswith('#') and ':' not in stripped and '=' not in stripped:
                continue

            cleaned_lines.append(line)

        result = '\n'.join(cleaned_lines).strip()

        # Remove empty lines at start/end
        while result.startswith('\n'):
            result = result[1:]
        while result.endswith('\n\n'):
            result = result[:-1]

        return result
    
    def _parse_fixed_files_fallback(
        self,
        response: str,
        original_files: Dict[str, str]
    ) -> Optional[Dict[str, str]]:
        """
        Fallback: extract complete file modifications if diff parsing fails.
        Tries multiple patterns to handle various LLM response formats.

        IMPORTANT: Validates all extracted code before accepting it.
        Returns original files unchanged if no valid fixes found.
        """
        self.log("  Using fallback: complete file extraction")

        modified_files = dict(original_files)
        found_any = False

        def try_update_file(filename: str, code: str, source: str) -> bool:
            """Helper to validate and update a file. Returns True if successful."""
            # Clean the code first to remove instruction comments
            code = self._clean_code_snippet(code)

            # NO SIZE LIMITS - Accept all fixes regardless of size
            # Only validate Python syntax
            if self._is_valid_python_code(code):
                modified_files[filename] = code
                self.log(f"  Accepted: {filename} ({source}, {len(code)} chars)")
                return True
            else:
                self.log(f"  Invalid Python syntax for: {filename} ({source})")
                return False

        # Pattern 1: === filename.py ===
        pattern1 = r'===\s*([^\s]+\.py)\s*===\s*```python\n(.*?)```'
        for filename, code in re.findall(pattern1, response, re.DOTALL):
            if filename in modified_files:
                if try_update_file(filename, code, "=== pattern"):
                    found_any = True

        # Pattern 2: # filename.py
        pattern2 = r'#\s*([^\s]+\.py)\s*\n```python\n(.*?)```'
        for filename, code in re.findall(pattern2, response, re.DOTALL):
            if filename in modified_files and not found_any:
                if try_update_file(filename, code, "# pattern"):
                    found_any = True

        # Pattern 3: **filename.py** (markdown bold)
        pattern3 = r'\*\*([^\s]+\.py)\*\*[:\s]*\n```python\n(.*?)```'
        for filename, code in re.findall(pattern3, response, re.DOTALL):
            if filename in modified_files:
                if try_update_file(filename, code, "**bold** pattern"):
                    found_any = True

        # Pattern 4: `filename.py`: (backtick format)
        pattern4 = r'`([^\s]+\.py)`[:\s]*\n```python\n(.*?)```'
        for filename, code in re.findall(pattern4, response, re.DOTALL):
            if filename in modified_files:
                if try_update_file(filename, code, "backtick pattern"):
                    found_any = True

        # Pattern 5: File: filename.py
        pattern5 = r'File:\s*([^\s\n]+\.py)[:\s]*\n```python\n(.*?)```'
        for filename, code in re.findall(pattern5, response, re.DOTALL | re.IGNORECASE):
            if filename in modified_files:
                if try_update_file(filename, code, "File: pattern"):
                    found_any = True

        # Pattern 6: Target File: filename.py (from our prompt format)
        pattern6 = r'Target File:\s*([^\s\n]+\.py)'
        target_matches = re.findall(pattern6, response, re.IGNORECASE)
        if target_matches and not found_any:
            for target_file in target_matches:
                if target_file in modified_files:
                    idx = response.find(f"Target File: {target_file}")
                    if idx == -1:
                        idx = response.lower().find(f"target file: {target_file.lower()}")
                    if idx != -1:
                        rest = response[idx:]
                        code_match = re.search(r'```python\n(.*?)```', rest, re.DOTALL)
                        if code_match:
                            code = code_match.group(1).strip()
                            # Skip if it's clearly a diff instruction
                            if '# OLD:' not in code and '# NEW:' not in code:
                                if try_update_file(target_file, code, "Target File pattern"):
                                    found_any = True

        if found_any:
            return modified_files

        # Last resort: try to infer target file from error context
        code_blocks = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
        if code_blocks:
            # If only one file, assume the fix is for that file
            if len(original_files) == 1:
                only_file = list(original_files.keys())[0]
                code = self._clean_code_snippet(code_blocks[0])

                # NO SIZE LIMITS - Accept all valid Python code
                if self._is_valid_python_code(code):
                    modified_files[only_file] = code
                    self.log(f"  ✓ Inferred single file: {only_file} ({len(code)} chars)")
                    return modified_files
                else:
                    self.log(f"  ✗ Inferred code invalid syntax: {only_file}")
                    return None

            # Try to apply diff-style fixes when Target File is explicitly specified
            # This is safer than inferring from filename mentions
            target_file_match = re.search(r'Target File:\s*([^\s\n]+\.py)', response, re.IGNORECASE)
            if target_file_match:
                target_file = target_file_match.group(1).strip()
                if target_file in original_files:
                    # Try to extract and apply OLD/NEW diff
                    old_new = self._extract_old_new_from_response(response)
                    if old_new:
                        old_code, new_code = old_new
                        original_content = original_files[target_file]
                        if old_code in original_content:
                            # Apply the diff
                            fixed_content = original_content.replace(old_code, new_code, 1)
                            if self._is_valid_python_code(fixed_content):
                                modified_files[target_file] = fixed_content
                                self.log(f"  ✓ Applied diff to: {target_file}")
                                return modified_files
                            else:
                                self.log(f"  ✗ Diff result is invalid Python: {target_file}")
                        else:
                            self.log(f"  ✗ OLD code not found in {target_file}")

            self.log("  ✗ Multiple files present, cannot safely infer target")
            return None

        # No valid fixes found - return None to signal failure
        # This prevents corrupting files with invalid code
        self.log("  No valid Python code found in response")
        return None
    
    def _extract_old_new_from_response(self, response: str) -> Optional[Tuple[str, str]]:
        """
        Extract OLD and NEW code snippets from an LLM response.
        Returns (old_code, new_code) tuple or None if not found.
        """
        # Pattern for: # OLD:\n<code>\n\n# NEW:\n<code>
        patterns = [
            # Pattern 1: Inside code block with # In file header
            r'```python\s*\n#\s*In\s+[^\n]+\n#\s*OLD:?\s*\n(.*?)\n\n#\s*NEW:?\s*\n(.*?)\n```',
            # Pattern 2: Simple # OLD / # NEW
            r'#\s*OLD:?\s*\n(.*?)\n\n#\s*NEW:?\s*\n(.*?)(?:\n```|```|\Z)',
            # Pattern 3: Without blank line
            r'#\s*OLD:?\s*\n(.*?)\n#\s*NEW:?\s*\n(.*?)(?:\n```|```|\Z)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                old_code = match.group(1).strip()
                new_code = match.group(2).strip()
                if old_code and new_code:
                    return (old_code, new_code)

        return None

    def _looks_like_complete_file(self, code: str) -> bool:
        """
        Check if code looks like a complete Python file, not a snippet.

        This check is VERY lenient to avoid rejecting valid fixes.
        Large rewrites are common and should be allowed.
        """
        lines = code.strip().split('\n')

        # Only reject very short code (less than 2 lines)
        if len(lines) < 2:
            return False

        # Accept if it has ANY Python structural element
        has_import = any('import ' in line for line in lines[:20])
        has_class = any(line.strip().startswith('class ') for line in lines)
        has_def = any(line.strip().startswith('def ') for line in lines)
        has_main_block = any('__name__' in line for line in lines)
        has_decorator = any(line.strip().startswith('@') for line in lines)

        if has_import or has_class or has_def or has_main_block or has_decorator:
            return True

        # Check for variable assignments (config files, constants)
        has_assignment = any('=' in line and not line.strip().startswith('#') for line in lines)
        if has_assignment:
            return True

        # Fallback: if it has reasonable length, accept it
        if len(lines) >= 5:
            return True

        # Accept any code with common Python keywords
        python_keywords = ['def ', 'class ', 'import ', 'return ', 'if ', 'for ', 'while ', 'try:', 'with ']
        if any(kw in code for kw in python_keywords):
            return True

        return False

    def _format_files(self, files: Dict[str, str]) -> str:
        """Format files for prompt"""
        formatted = []
        for name, code in files.items():
            formatted.append(f"=== {name} ===\n{code}\n")
        return "\n".join(formatted)

    def _is_valid_python_code(self, code: str, strict: bool = True) -> bool:
        """
        Validate that code is syntactically valid Python.

        Args:
            code: The code to validate
            strict: If True, apply additional checks beyond syntax

        Returns:
            True if code is valid Python
        """
        code_stripped = code.strip()

        # Basic length check
        if len(code_stripped) < 10:
            return False

        # Try to parse as Python (the definitive test)
        try:
            ast.parse(code)
        except SyntaxError as e:
            self.log(f"  Syntax error: {e.msg} at line {e.lineno}")
            return False
        except Exception as e:
            self.log(f"  Parse error: {e}")
            return False

        # If not strict mode, syntax validity is enough
        if not strict:
            return True

        # Reject if it looks like diff instructions instead of code
        # Only check first 3 lines to avoid false positives
        first_lines = '\n'.join(code_stripped.split('\n')[:3])
        diff_indicators = ['# OLD:', '# NEW:', '<old_code>', '<new_code>', '# BEFORE:', '# AFTER:']
        if any(indicator in first_lines for indicator in diff_indicators):
            self.log(f"  Rejected: looks like diff instructions")
            return False

        return True

    def _extract_error_file(self, error: str) -> Optional[str]:
        """
        Extract the filename that caused the error from traceback.

        Looks for patterns like:
        - File "C:/path/to/file.py", line X
        - File "/path/to/file.py", line X

        Returns the basename (e.g., "config.py") or None.
        """
        # Match the last file mentioned in traceback (usually the actual error location)
        matches = re.findall(r'File\s+"[^"]*[\\/](\w+\.py)"', error)
        if matches:
            # Return the last match (deepest in call stack = actual error)
            return matches[-1]
        return None

    def regenerate_file(
        self,
        target_file: str,
        original_code: str,
        execution_error: str,
        error_analysis: str,
    ) -> Optional[str]:
        """
        Regenerate an entire file with fixes applied.

        This is a fallback when diff-based editing fails. Instead of trying
        to apply patches, we ask the LLM to output the complete fixed file.

        For LARGE files (>15000 chars), uses surgical line-based fixing instead
        of full regeneration to avoid truncation.

        Args:
            target_file: Name of the file to fix (e.g., "config.py")
            original_code: Current content of the file
            execution_error: The error traceback
            error_analysis: Analysis of what went wrong

        Returns:
            Fixed file content or None if regeneration fails
        """
        self.log(f"  Regenerating {target_file} with fixes applied...")

        # For large files, use surgical line-based fixing
        if len(original_code) > 15000:
            return self._surgical_fix_large_file(
                target_file, original_code, execution_error, error_analysis
            )

        prompt = f"""Fix this Python file that has an error. Output the COMPLETE fixed file.

FILE: {target_file}
SIZE: {len(original_code)} characters

CURRENT CODE:
```python
{original_code}
```

ERROR:
{execution_error[:3000]}

ANALYSIS:
{error_analysis[:1500]}

REQUIREMENTS:
1. Output the COMPLETE file in a single ```python block
2. Keep all working code - only fix the error
3. The file MUST be syntactically valid Python
4. NO emojis - ASCII only
5. Use encoding='utf-8' for file operations
6. For dataclasses: use field(default_factory=list) for mutable defaults

Output format:
```python
# Fixed {target_file}
[complete file content here]
```
"""

        response = self.call_llm(
            user_message=prompt,
            temperature=0.1,
            max_tokens=Config.MAX_TOKENS
        )

        # Extract code from response
        code = self._extract_code_from_regeneration(response["content"])

        # NO SIZE LIMITS - Accept all valid Python code
        if code and self._is_valid_python_code(code):
            self.log(f"  ✓ Regenerated {target_file} ({len(code)} chars)")
            return code
        else:
            self.log(f"  ✗ Regenerated code is invalid or empty")
            return None

    def _extract_code_from_regeneration(self, response: str) -> Optional[str]:
        """Extract code from a regeneration response."""
        # Look for python code block
        match = re.search(r'```python\s*\n(.*?)```', response, re.DOTALL)
        if match:
            code = match.group(1).strip()
            # Remove any leading comment like "# Complete fixed config.py"
            lines = code.split('\n')
            if lines and lines[0].strip().startswith('# Complete fixed'):
                lines = lines[1:]
            return '\n'.join(lines).strip()
        return None

    def _surgical_fix_large_file(
        self,
        target_file: str,
        original_code: str,
        execution_error: str,
        error_analysis: str,
    ) -> Optional[str]:
        """
        Fix a large file by surgically editing only the problematic section.

        Instead of regenerating the entire file (which may exceed token limits),
        this method:
        1. Extracts the error line number from the traceback
        2. Gets a context window around that line
        3. Asks the LLM to fix just that section
        4. Replaces the section in the original file

        Args:
            target_file: Name of the file
            original_code: Full file content
            execution_error: Error traceback
            error_analysis: Analysis of the error

        Returns:
            Fixed file content or None
        """
        self.log(f"  Using surgical fix for large file ({len(original_code)} chars)")

        # Extract error line number
        line_num = self._extract_error_line_number(execution_error, target_file)
        if not line_num:
            self.log("  Could not extract line number, falling back to full regen")
            return self._fallback_full_regeneration(
                target_file, original_code, execution_error, error_analysis
            )

        # Get context window (50 lines before and after)
        lines = original_code.split('\n')
        context_start = max(0, line_num - 50)
        context_end = min(len(lines), line_num + 50)

        context_lines = lines[context_start:context_end]
        context_code = '\n'.join(context_lines)

        # Create focused prompt
        prompt = f"""Fix the error in this Python code section.

FILE: {target_file}
ERROR LINE: {line_num} (shown as line {line_num - context_start} in the context below)

CODE SECTION (lines {context_start + 1} to {context_end}):
```python
{context_code}
```

ERROR:
{execution_error[:1500]}

ANALYSIS:
{error_analysis[:800]}

CRITICAL REQUIREMENTS:
1. Output ONLY the fixed code section (lines {context_start + 1} to {context_end})
2. Keep the SAME number of lines if possible
3. Preserve indentation exactly
4. NO EMOJIS - ASCII only
5. Fix ONLY the error - don't change working code
6. For dataclass fields, use field(default_factory=list) not []
7. Import 'field' from dataclasses if using default_factory

Output the fixed section in a ```python block:
"""

        response = self.call_llm(
            user_message=prompt,
            temperature=0,
            max_tokens=Config.MAX_TOKENS  # Use full token limit for surgical fixes
        )

        # Extract fixed section
        fixed_section = self._extract_code_from_regeneration(response["content"])
        if not fixed_section:
            self.log("  Could not extract fixed section")
            return None

        # Validate the fixed section
        # Try to parse just the section (may fail if it's mid-class, but worth trying)
        fixed_lines = fixed_section.split('\n')

        # Reconstruct the file
        new_lines = lines[:context_start] + fixed_lines + lines[context_end:]
        new_code = '\n'.join(new_lines)

        # Validate the complete file
        if self._is_valid_python_code(new_code):
            self.log(f"  ✓ Surgical fix successful ({len(new_code)} chars)")
            return new_code
        else:
            self.log("  ✗ Surgical fix produced invalid Python")
            return None

    def _extract_error_line_number(self, error: str, target_file: str) -> Optional[int]:
        """Extract the line number from error traceback for the target file."""
        # Pattern: File "...target_file", line X
        pattern = rf'File\s+"[^"]*{re.escape(target_file)}"\s*,\s*line\s+(\d+)'
        matches = re.findall(pattern, error)
        if matches:
            return int(matches[-1])  # Last match = deepest in stack
        return None

    def _fallback_full_regeneration(
        self,
        target_file: str,
        original_code: str,
        execution_error: str,
        error_analysis: str,
    ) -> Optional[str]:
        """Fallback to full file regeneration for smaller files."""
        # Use a truncated version of the original code if too long
        max_code_len = 12000
        if len(original_code) > max_code_len:
            # Truncate but keep structure
            truncated = original_code[:max_code_len] + "\n# ... (truncated) ..."
            self.log(f"  Truncating code from {len(original_code)} to {max_code_len} chars")
        else:
            truncated = original_code

        prompt = f"""Fix this Python file. Output the COMPLETE fixed file.

FILE: {target_file}

CURRENT CODE:
```python
{truncated}
```

ERROR:
{execution_error[:1500]}

ANALYSIS:
{error_analysis[:800]}

REQUIREMENTS:
1. Output COMPLETE file, not just changes
2. NO EMOJIS - ASCII characters only
3. For dataclass: use field(default_factory=list), never []
4. Import field from dataclasses if needed

Output in a ```python block:
"""

        response = self.call_llm(
            user_message=prompt,
            temperature=0.1,
            max_tokens=Config.MAX_TOKENS
        )

        code = self._extract_code_from_regeneration(response["content"])

        # NO SIZE LIMITS - Accept all valid Python code
        if code and self._is_valid_python_code(code):
            self.log(f"  ✓ Regenerated {target_file} ({len(code)} chars)")
            return code

        self.log(f"  ✗ Regeneration failed")
        return None