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
        Apply a single edit to code.

        Args:
            original_code: Original file content
            edit: Edit to apply

        Returns:
            Tuple of (success, modified_code, error_message)
        """
        old_str = edit.old_str

        # First try exact match
        if old_str in original_code:
            count = original_code.count(old_str)
            if count > 1:
                return False, original_code, f"Old string appears {count} times (must be unique)"
            modified_code = original_code.replace(old_str, edit.new_str, 1)
            return True, modified_code, ""

        # Try fuzzy matching: normalize whitespace
        normalized_old = self._normalize_whitespace(old_str)

        # Try to find a match with normalized whitespace
        lines = original_code.split('\n')
        for i, line in enumerate(lines):
            normalized_line = self._normalize_whitespace(line)
            if normalized_old in normalized_line or normalized_line in normalized_old:
                # Found potential match - try replacing the whole line
                if old_str.strip() == line.strip():
                    lines[i] = line.replace(line.strip(), edit.new_str.strip())
                    modified_code = '\n'.join(lines)
                    return True, modified_code, ""

        # Try matching by stripping leading/trailing whitespace from both
        old_stripped = old_str.strip()
        if old_stripped in original_code:
            count = original_code.count(old_stripped)
            if count == 1:
                modified_code = original_code.replace(old_stripped, edit.new_str.strip(), 1)
                return True, modified_code, ""

        return False, original_code, f"Old string not found in file (tried exact and fuzzy match)"

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
        
        for edit in edits:
            if edit.file_path not in modified_files:
                errors.append(f"File not found: {edit.file_path}")
                continue
            
            success, new_code, error = self.apply_edit(
                modified_files[edit.file_path],
                edit
            )
            
            if success:
                modified_files[edit.file_path] = new_code
                print(f"  ✓ Applied edit to {edit.file_path}")
            else:
                errors.append(f"{edit.file_path}: {error}")
        
        all_success = len(errors) == 0
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