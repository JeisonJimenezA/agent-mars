# execution/diff_editor.py
from typing import Dict, List, Optional
from dataclasses import dataclass
import difflib

@dataclass
class DiffEdit:
    """
    Represents a diff-based edit to a file.
    Implements the diff-based editing mechanism from Section 4.2.
    """
    file_path: str
    old_str: str
    new_str: str
    description: str = ""
    
    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "old_str": self.old_str,
            "new_str": self.new_str,
            "description": self.description
        }

class DiffEditor:
    """
    Applies diff-based edits to code files.
    Enables atomic, multi-file updates in a single step.
    """
    
    def apply_edit(
        self,
        original_code: str,
        edit: DiffEdit
    ) -> tuple[bool, str, str]:
        """
        Apply a single edit to code with multiple matching strategies.

        Args:
            original_code: Original file content
            edit: Edit to apply

        Returns:
            Tuple of (success, modified_code, error_message)
        """
        old_str = edit.old_str
        new_str = edit.new_str

        # Strategy 1: Exact match
        if old_str in original_code:
            count = original_code.count(old_str)
            if count == 1:
                modified_code = original_code.replace(old_str, new_str, 1)
                return True, modified_code, ""
            elif count > 1:
                # Try to find unique context
                pass  # Continue to other strategies

        # Strategy 2: Match with normalized line endings
        old_normalized = old_str.replace('\r\n', '\n').replace('\r', '\n')
        code_normalized = original_code.replace('\r\n', '\n').replace('\r', '\n')
        if old_normalized in code_normalized:
            count = code_normalized.count(old_normalized)
            if count == 1:
                modified_code = code_normalized.replace(old_normalized, new_str.replace('\r\n', '\n'), 1)
                return True, modified_code, ""

        # Strategy 3: Match stripped version
        old_stripped = old_str.strip()
        if old_stripped and old_stripped in original_code:
            count = original_code.count(old_stripped)
            if count == 1:
                # Preserve surrounding whitespace structure
                idx = original_code.find(old_stripped)
                # Get the indentation of the original
                line_start = original_code.rfind('\n', 0, idx) + 1
                indent = original_code[line_start:idx]

                # Apply indent to new code
                new_lines = new_str.strip().split('\n')
                if len(new_lines) > 1:
                    indented_new = new_lines[0]
                    for line in new_lines[1:]:
                        if line.strip():
                            indented_new += '\n' + indent + line.lstrip()
                        else:
                            indented_new += '\n'
                    new_str_indented = indented_new
                else:
                    new_str_indented = new_str.strip()

                modified_code = original_code[:idx] + new_str_indented + original_code[idx + len(old_stripped):]
                return True, modified_code, ""

        # Strategy 4: Line-by-line matching for multi-line old_str
        old_lines = [l.strip() for l in old_str.strip().split('\n') if l.strip()]
        if len(old_lines) >= 1:
            code_lines = original_code.split('\n')

            for i in range(len(code_lines) - len(old_lines) + 1):
                match = True
                for j, old_line in enumerate(old_lines):
                    if code_lines[i + j].strip() != old_line:
                        match = False
                        break

                if match:
                    # Found match - replace the lines
                    # Preserve indentation from first matched line
                    first_line = code_lines[i]
                    indent = len(first_line) - len(first_line.lstrip())
                    indent_str = first_line[:indent]

                    new_lines = new_str.strip().split('\n')
                    indented_new_lines = []
                    for k, nl in enumerate(new_lines):
                        if k == 0:
                            indented_new_lines.append(indent_str + nl.lstrip())
                        elif nl.strip():
                            indented_new_lines.append(indent_str + nl.lstrip())
                        else:
                            indented_new_lines.append('')

                    result_lines = code_lines[:i] + indented_new_lines + code_lines[i + len(old_lines):]
                    modified_code = '\n'.join(result_lines)
                    return True, modified_code, ""

        # Strategy 5: Single line fuzzy match
        if '\n' not in old_str.strip():
            old_stripped = old_str.strip()
            for i, line in enumerate(original_code.split('\n')):
                if line.strip() == old_stripped:
                    # Preserve indentation
                    indent = len(line) - len(line.lstrip())
                    lines = original_code.split('\n')
                    lines[i] = line[:indent] + new_str.strip()
                    return True, '\n'.join(lines), ""

        return False, original_code, f"Old string not found (tried 5 strategies)"

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace for fuzzy matching"""
        import re
        # Replace multiple spaces/tabs with single space
        return re.sub(r'\s+', ' ', text.strip())
    
    def apply_multiple_edits(
        self,
        files: Dict[str, str],
        edits: List[DiffEdit]
    ) -> tuple[bool, Dict[str, str], List[str]]:
        """
        Apply multiple edits to multiple files.

        Args:
            files: Dictionary of {filename: content}
            edits: List of edits to apply

        Returns:
            Tuple of (all_success, modified_files, list_of_errors)
        """
        modified_files = files.copy()
        errors = []
        applied_count = 0

        for i, edit in enumerate(edits):
            print(f"  Edit {i+1}/{len(edits)}: {edit.file_path}")

            if edit.file_path not in modified_files:
                errors.append(f"File not found: {edit.file_path}")
                print(f"    SKIP: file not in solution")
                continue

            # Show what we're trying to match
            old_preview = edit.old_str[:60].replace('\n', '\\n')
            new_preview = edit.new_str[:60].replace('\n', '\\n')
            print(f"    OLD: {old_preview}...")
            print(f"    NEW: {new_preview}...")

            success, new_code, error = self.apply_edit(
                modified_files[edit.file_path],
                edit
            )

            if success:
                modified_files[edit.file_path] = new_code
                applied_count += 1
                print(f"    OK: applied successfully")
            else:
                errors.append(f"{edit.file_path}: {error}")
                print(f"    FAIL: {error}")

        print(f"  Summary: {applied_count}/{len(edits)} edits applied")

        # Consider partial success as success if at least one edit worked
        all_success = applied_count > 0 and len(errors) == 0
        return all_success, modified_files, errors
    
    def generate_unified_diff(
        self,
        original: str,
        modified: str,
        filename: str = "file"
    ) -> str:
        """
        Generate unified diff between two versions.
        
        Args:
            original: Original content
            modified: Modified content
            filename: File name for diff header
            
        Returns:
            Unified diff string
        """
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm=''
        )
        
        return ''.join(diff)
    
    def parse_diff_from_llm_response(self, response: str) -> List[DiffEdit]:
        """
        Parse diff edits from LLM response.
        
        Expected format:
```diff
        file_path: path/to/file.py
        old_str: |
          old code here
        new_str: |
          new code here
```
        
        Args:
            response: LLM response text
            
        Returns:
            List of DiffEdit objects
        """
        # This is a simplified parser
        # In production, would use more robust parsing
        
        edits = []
        
        # TODO: Implement proper parsing of structured diff format
        # For now, this is a placeholder
        
        return edits