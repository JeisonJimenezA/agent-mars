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
        return """You are an expert Python debugger specializing in ML code. You excel at diagnosing errors and proposing minimal, targeted fixes using diff-based editing."""
    
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
            temperature=0.6,
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
        self.log("Generating error fix (diff-based)...")
        
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
        
        # Call LLM - usar max tokens del modelo (8192)
        response = self.call_llm(
            user_message=prompt,
            temperature=0.7,
            max_tokens=Config.MAX_TOKENS
        )
        
        # Parse diff-based fixes
        success, modified_files, errors = self._parse_diff_fixes(
            response["content"],
            files
        )
        
        if success:
            self.log(f"Generated fixes for {len(modified_files)} files (diff-based)")
            return modified_files
        else:
            self.log(f"Diff parsing failed, attempting fallback...")
            # Fallback to old method
            modified_files = self._parse_fixed_files_fallback(response["content"], files)
            if modified_files:
                self.log(f"Fallback succeeded: {len(modified_files)} files")
                return modified_files
            else:
                self.log("Failed to parse fixed files")
                # Log first 500 chars of response for debugging
                response_preview = response["content"][:500] if len(response["content"]) > 500 else response["content"]
                self.log(f"LLM response preview:\n{response_preview}")
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
        """
        prompt = f"""You are debugging a Python ML solution. Propose MINIMAL, TARGETED fixes.

PROBLEM:
{problem_description[:500]}

CURRENT FILES:
{files_text[:6000]}

EXECUTION ERROR:
{execution_error[:2000]}

ERROR ANALYSIS:
{error_analysis}

DEBUG LESSONS:
{debug_lessons}

YOUR TASK:
Propose 1-3 MINIMAL fixes to resolve this error. Each fix should:
1. Target the EXACT cause of the error
2. Change only what's necessary
3. Preserve all working code
4. Be implementable as a code diff

RESPONSE FORMAT:
For each fix:

**Fix [N]: [Brief Description]**
Reasoning: [Why this fixes the error - cite lessons if applicable]
Target File: [filename.py]
Risk Level: [low|medium|high]

Code Modification:
```python
# In [filename.py], around line [X]:
# OLD:
[exact old code - must be UNIQUE in file]

# NEW:
[replacement code]
```

CRITICAL RULES:
- OLD code must appear EXACTLY ONCE in the target file
- Keep changes minimal (1-5 lines preferred)
- Preserve indentation exactly
- Do NOT rewrite entire functions unless necessary
- Focus on the ERROR ROOT CAUSE only

EXAMPLES OF GOOD FIXES:
✅ Change: max_iter=10 → max_iter=100 (fix convergence)
✅ Add: import numpy as np (fix import error)
✅ Change: train.csv → metadata/train.csv (fix path)

EXAMPLES OF BAD FIXES:
❌ Rewriting entire file
❌ Changing working code
❌ Multiple unrelated changes
"""
        
        return prompt
    
    def _parse_diff_fixes(
        self,
        response: str,
        original_files: Dict[str, str]
    ) -> Tuple[bool, Dict[str, str], List[str]]:
        """
        Parse diff-based fixes from LLM response.
        
        Returns:
            Tuple of (success, modified_files, errors)
        """
        # Extract fix blocks
        fixes = self._extract_fix_blocks(response)
        
        if not fixes:
            return False, {}, ["No fixes found in response"]
        
        # Parse each fix as a DiffEdit
        diff_edits = []
        for fix in fixes:
            target_file = self._extract_target_file(fix)
            if not target_file or target_file not in original_files:
                continue
            
            old_new_pairs = self._extract_old_new_pairs(fix)
            
            for old_code, new_code in old_new_pairs:
                # Clean code snippets
                old_clean = self._clean_code_snippet(old_code)
                new_clean = self._clean_code_snippet(new_code)
                
                if old_clean and new_clean:
                    diff_edit = DiffEdit(
                        file_path=target_file,
                        old_str=old_clean,
                        new_str=new_clean,
                        description=f"Debug fix from error analysis"
                    )
                    diff_edits.append(diff_edit)
        
        if not diff_edits:
            return False, {}, ["Failed to parse any valid diffs"]
        
        # Apply diffs
        success, modified_files, errors = self.diff_editor.apply_multiple_edits(
            original_files,
            diff_edits
        )

        # Validate that resulting files are still valid Python
        if success:
            for filename, code in modified_files.items():
                if filename.endswith('.py') and not self._is_valid_python_code(code):
                    self.log(f"  ✗ Diff resulted in invalid Python: {filename}")
                    # Revert to original
                    modified_files[filename] = original_files.get(filename, code)
                    errors.append(f"Diff for {filename} produced invalid Python")

            # Check if any files were actually modified successfully
            actually_modified = sum(1 for f in modified_files
                                   if modified_files[f] != original_files.get(f))
            if actually_modified == 0:
                return False, {}, errors + ["No valid modifications applied"]

        return success, modified_files, errors
    
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

        # Pattern 1: # OLD: ... # NEW: ... (with flexible spacing)
        pattern1 = r'#\s*OLD:?\s*\n(.*?)\n\s*#\s*NEW:?\s*\n(.*?)(?:\n```|\n#|\Z)'
        matches1 = re.findall(pattern1, text, re.DOTALL | re.IGNORECASE)
        pairs.extend(matches1)

        # Pattern 2: OLD CODE / NEW CODE in markdown blocks
        pattern2 = r'```python\s*\n#\s*OLD:?\s*\n(.*?)\n\s*#\s*NEW:?\s*\n(.*?)\n```'
        matches2 = re.findall(pattern2, text, re.DOTALL | re.IGNORECASE)
        pairs.extend(matches2)

        # Pattern 3: Before/After format
        pattern3 = r'#?\s*(?:BEFORE|Original):?\s*\n(.*?)\n\s*#?\s*(?:AFTER|Fixed|Replacement):?\s*\n(.*?)(?:\n```|\n#|\Z)'
        matches3 = re.findall(pattern3, text, re.DOTALL | re.IGNORECASE)
        pairs.extend(matches3)

        # Pattern 4: Change/Replace format
        pattern4 = r'(?:Change|Replace):?\s*\n```python\n(.*?)\n```\s*(?:To|With):?\s*\n```python\n(.*?)\n```'
        matches4 = re.findall(pattern4, text, re.DOTALL | re.IGNORECASE)
        pairs.extend(matches4)

        # Pattern 5: Simple arrow format (old → new or old -> new)
        pattern5 = r'`([^`]+)`\s*(?:→|->|=>)\s*`([^`]+)`'
        matches5 = re.findall(pattern5, text)
        pairs.extend(matches5)

        # Pattern 6: From/To format
        pattern6 = r'From:?\s*\n?```python\n(.*?)\n```\s*To:?\s*\n?```python\n(.*?)\n```'
        matches6 = re.findall(pattern6, text, re.DOTALL | re.IGNORECASE)
        pairs.extend(matches6)

        # Pattern 7: **OLD:** / **NEW:** markdown bold headers
        pattern7 = r'\*\*OLD:?\*\*\s*\n?```python\n(.*?)\n```\s*\*\*NEW:?\*\*\s*\n?```python\n(.*?)\n```'
        matches7 = re.findall(pattern7, text, re.DOTALL | re.IGNORECASE)
        pairs.extend(matches7)

        # Pattern 8: Numbered change format (common LLM pattern)
        # "1. Change X to Y" with code blocks
        pattern8 = r'```python\n(.*?)\n```\s*(?:should be|becomes|change to|→|->)\s*```python\n(.*?)\n```'
        matches8 = re.findall(pattern8, text, re.DOTALL | re.IGNORECASE)
        pairs.extend(matches8)

        # Pattern 9: Inline code replacement (less strict)
        pattern9 = r'OLD:\s*```(?:python)?\n?(.*?)\n?```\s*NEW:\s*```(?:python)?\n?(.*?)\n?```'
        matches9 = re.findall(pattern9, text, re.DOTALL | re.IGNORECASE)
        pairs.extend(matches9)

        # Pattern 10: Loose # In [file] format followed by code
        pattern10 = r'#\s*In\s+[^\n]+\n#\s*OLD:?\n(.*?)\n#\s*NEW:?\n(.*?)(?:\n```|\Z)'
        matches10 = re.findall(pattern10, text, re.DOTALL | re.IGNORECASE)
        pairs.extend(matches10)

        return pairs
    
    def _clean_code_snippet(self, code: str) -> str:
        """Clean code snippet for diff matching"""
        # Remove markdown code blocks
        code = re.sub(r'```python|```', '', code).strip()
        
        # Remove leading comment markers but preserve indentation
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip pure comment-only lines
            stripped = line.strip()
            if stripped.startswith('#') and ':' not in stripped:
                continue
            # Remove leading "# " from code lines
            if line.strip().startswith('# ') and not line.strip().startswith('# In'):
                line = line.replace('# ', '', 1)
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
            code = code.strip()
            if self._is_valid_python_code(code):
                modified_files[filename] = code
                self.log(f"  ✓ Extracted valid code: {filename} ({source})")
                return True
            else:
                self.log(f"  ✗ Invalid code rejected for: {filename} ({source})")
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
                code = code_blocks[0].strip()
                if self._is_valid_python_code(code):
                    modified_files[only_file] = code
                    self.log(f"  ✓ Inferred single file: {only_file}")
                    return modified_files
                else:
                    self.log(f"  ✗ Inferred code invalid, keeping original: {only_file}")
                    return None

            # Try to infer from filenames mentioned in the response
            for filename in original_files.keys():
                if filename in response or filename.replace('.py', '') in response:
                    for code_block in code_blocks:
                        code = code_block.strip()
                        if self._is_valid_python_code(code):
                            modified_files[filename] = code
                            self.log(f"  ✓ Inferred from context: {filename}")
                            return modified_files

        # No valid fixes found - return None to signal failure
        # This prevents corrupting files with invalid code
        self.log("  No valid Python code found in response")
        return None
    
    def _format_files(self, files: Dict[str, str]) -> str:
        """Format files for prompt"""
        formatted = []
        for name, code in files.items():
            formatted.append(f"=== {name} ===\n{code}\n")
        return "\n".join(formatted)

    def _is_valid_python_code(self, code: str) -> bool:
        """
        Validate that code is syntactically valid Python.
        Returns False if code contains diff instructions or is invalid.
        """
        # Reject if it looks like diff instructions instead of code
        diff_patterns = [
            r'^#\s*In\s+\w+\.py',      # "# In filename.py"
            r'^#\s*OLD:',               # "# OLD:"
            r'^#\s*NEW:',               # "# NEW:"
            r'^\s*#\s*(?:BEFORE|AFTER):', # "# BEFORE:" or "# AFTER:"
            r'^\(\s*file\s+(?:likely\s+)?missing', # "(file likely missing"
        ]

        first_lines = '\n'.join(code.strip().split('\n')[:5])
        for pattern in diff_patterns:
            if re.search(pattern, first_lines, re.IGNORECASE | re.MULTILINE):
                self.log(f"  Rejected: code looks like diff instructions")
                return False

        # Check minimum code characteristics
        code_stripped = code.strip()
        if len(code_stripped) < 20:
            self.log(f"  Rejected: code too short ({len(code_stripped)} chars)")
            return False

        # Must contain at least one Python construct
        has_python_construct = any(kw in code for kw in [
            'import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ',
            'return ', 'yield ', 'raise ', 'try:', 'with ', '= ', '()'
        ])
        if not has_python_construct:
            self.log(f"  Rejected: no Python constructs found")
            return False

        # Try to parse as Python
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            self.log(f"  Rejected: syntax error - {e.msg} at line {e.lineno}")
            return False
        except Exception as e:
            self.log(f"  Rejected: parse error - {e}")
            return False