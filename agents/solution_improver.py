# agents/solution_improver.py
from typing import Dict, List, Optional, Tuple
from agents.base_agent import BaseAgent
from memory.lesson_pool import LessonPool
from memory.lesson_types import LessonType
from execution.diff_editor import DiffEdit, DiffEditor
from core.config import Config
import re

class SolutionImprover(BaseAgent):
    """
    Agent responsible for improving existing solutions.
    
    Implements the IMPROVE action from Section 4.4.1 of the paper.
    Uses diff-based editing for atomic modifications.
    """
    
    def __init__(self):
        super().__init__("SolutionImprover")
        self.diff_editor = DiffEditor()
    
    def get_system_prompt(self) -> str:
        return """You are an expert ML engineer specializing in iterative solution improvement. You excel at proposing targeted, high-impact modifications based on empirical evidence and best practices."""
    
    def propose_improvements(
        self,
        problem_description: str,
        current_solution: Dict[str, str],  # {filename: code}
        current_metric: Optional[float],
        lesson_pool: LessonPool
    ) -> Tuple[bool, Dict[str, str], str]:
        """
        Propose improvements to current solution.
        
        Returns:
            Tuple of (success, improved_files, reasoning)
        """
        self.log("Proposing solution improvements...")
        
        # Get relevant lessons
        solution_lessons = lesson_pool.format_for_prompt(
            lesson_type=LessonType.SOLUTION,
            k=10
        )
        
        # Format current solution
        solution_text = self._format_solution(current_solution)
        
        # Build prompt
        prompt = self._create_improvement_prompt(
            problem_description,
            solution_text,
            current_metric,
            solution_lessons
        )
        
        # Call LLM - usar max tokens del modelo (8192)
        response = self.call_llm(
            user_message=prompt,
            temperature=0.7,
            max_tokens=Config.MAX_TOKENS
        )
        
        # Parse improvements
        success, improvements, reasoning = self._parse_improvement_response(
            response["content"],
            current_solution
        )
        
        if success:
            self.log(f"  Proposed {len(improvements)} file modifications")
            return True, improvements, reasoning
        else:
            self.log("  Failed to parse improvements")
            return False, {}, ""
    
    def apply_diff_improvements(
        self,
        current_solution: Dict[str, str],
        diff_edits: List[DiffEdit]
    ) -> Tuple[bool, Dict[str, str], List[str]]:
        """
        Apply diff-based edits to solution.
        
        Returns:
            Tuple of (success, modified_files, errors)
        """
        self.log(f"Applying {len(diff_edits)} diff edits...")
        
        success, modified_files, errors = self.diff_editor.apply_multiple_edits(
            current_solution,
            diff_edits
        )
        
        if success:
            self.log("  All edits applied successfully")
        else:
            self.log(f"  {len(errors)} errors during edit application")
            for error in errors:
                self.log(f"    - {error}")
        
        return success, modified_files, errors
    
    def _format_solution(self, files: Dict[str, str]) -> str:
        """Format solution files for prompt"""
        formatted = []
        for filename, code in files.items():
            formatted.append(f"=== {filename} ===")
            formatted.append(code)
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _create_improvement_prompt(
        self,
        problem_description: str,
        solution_text: str,
        current_metric: Optional[float],
        lessons: str
    ) -> str:
        """Create prompt for improvement proposal"""
        
        metric_context = ""
        if current_metric is not None:
            metric_context = f"Current validation metric: {current_metric:.6f}"
        
        prompt = f"""You are improving an ML solution. Propose targeted modifications to boost performance.

PROBLEM:
{problem_description[:600]}

CURRENT SOLUTION:
{solution_text[:3000]}

{metric_context}

LEARNED LESSONS:
{lessons}

YOUR TASK:
Propose 1-3 HIGH-IMPACT modifications to improve this solution. Each modification should:
1. Be specific and targeted (change one aspect)
2. Apply insights from the lessons (cite them!)
3. Be implementable as a code diff
4. Balance performance gain vs. execution cost

MODIFICATION TYPES:
- Hyperparameter tuning (learning rate, batch size, epochs)
- Feature engineering (new features, transformations)
- Model architecture tweaks (layers, regularization)
- Training strategy (optimizer, scheduler, loss function)
- Data augmentation or preprocessing

RESPONSE FORMAT:
For each modification, provide:

**Modification 1: [Brief Title]**
Reasoning: [Why this will help, cite lessons if applicable]
Target File: [filename]
Change Type: [hyperparameter|architecture|feature|training|data]

Specific Changes:
- [Describe exact change 1]
- [Describe exact change 2]

Code Modification:
```python
# In [filename], change:
# OLD:
[old code snippet - must be EXACT and UNIQUE in file]

# NEW:
[new code snippet]
```

IMPORTANT:
- OLD code must appear EXACTLY ONCE in the target file
- Be precise with indentation and syntax
- Keep changes minimal and focused
- Cite lessons when applicable: "Cite {{lesson_id}}"
"""
        
        return prompt
    
    def _parse_improvement_response(
        self,
        response: str,
        current_solution: Dict[str, str]
    ) -> Tuple[bool, Dict[str, str], str]:
        """
        Parse LLM response for improvement modifications.
        
        Extracts code diffs and applies them.
        """
        # Extract modifications
        modifications = self._extract_modifications(response)
        
        if not modifications:
            return False, {}, ""
        
        # Extract reasoning
        reasoning = self._extract_reasoning(response)
        
        # Parse diff edits
        diff_edits = []
        for mod in modifications:
            edits = self._parse_code_diffs(mod, current_solution)
            diff_edits.extend(edits)
        
        if not diff_edits:
            # Fallback: try to extract modified files directly
            return self._fallback_parse_files(response, current_solution, reasoning)
        
        # Apply diffs
        success, improved_files, errors = self.apply_diff_improvements(
            current_solution,
            diff_edits
        )
        
        return success, improved_files, reasoning
    
    def _extract_modifications(self, response: str) -> List[str]:
        """Extract modification sections from response"""
        # Split by modification markers
        pattern = r'\*\*Modification \d+:'
        parts = re.split(pattern, response)
        
        # Skip first part (preamble)
        modifications = [part.strip() for part in parts[1:] if part.strip()]
        
        return modifications
    
    def _extract_reasoning(self, response: str) -> str:
        """Extract overall reasoning from response"""
        # Look for reasoning sections
        lines = []
        for line in response.split('\n')[:10]:  # First 10 lines
            if 'reasoning' in line.lower() or 'because' in line.lower():
                lines.append(line)
        
        return ' '.join(lines) if lines else "Incremental improvements"
    
    def _parse_code_diffs(
        self,
        modification_text: str,
        current_solution: Dict[str, str]
    ) -> List[DiffEdit]:
        """
        Parse code diffs from modification text.
        
        Looks for patterns like:
        # OLD:
        [old code]
        # NEW:
        [new code]
        """
        diff_edits = []
        
        # Extract target file
        target_file = self._extract_target_file(modification_text)
        if not target_file or target_file not in current_solution:
            return []
        
        # Extract OLD/NEW pairs
        old_new_pairs = self._extract_old_new_pairs(modification_text)
        
        for old_code, new_code in old_new_pairs:
            # Clean code snippets
            old_clean = self._clean_code_snippet(old_code)
            new_clean = self._clean_code_snippet(new_code)
            
            if old_clean and new_clean:
                diff_edit = DiffEdit(
                    file_path=target_file,
                    old_str=old_clean,
                    new_str=new_clean,
                    description=f"Improvement from LLM suggestion"
                )
                diff_edits.append(diff_edit)
        
        return diff_edits
    
    def _extract_target_file(self, text: str) -> Optional[str]:
        """Extract target filename from modification text"""
        # Look for "Target File:" or "In [filename]"
        patterns = [
            r'Target File:\s*([^\n]+)',
            r'In\s+([^\s,]+\.py)',
            r'File:\s*([^\n]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                filename = match.group(1).strip()
                # Clean up
                filename = filename.replace('[', '').replace(']', '').strip()
                return filename
        
        return None
    
    def _extract_old_new_pairs(self, text: str) -> List[Tuple[str, str]]:
        """Extract OLD/NEW code pairs"""
        pairs = []
        
        # Pattern: # OLD: ... # NEW: ...
        pattern = r'#\s*OLD:\s*\n(.*?)\n\s*#\s*NEW:\s*\n(.*?)(?:\n```|\n#|\Z)'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        for old_code, new_code in matches:
            pairs.append((old_code.strip(), new_code.strip()))
        
        # Alternative pattern: OLD CODE: ... NEW CODE: ...
        if not pairs:
            pattern2 = r'OLD CODE:\s*\n```python\n(.*?)\n```\s*NEW CODE:\s*\n```python\n(.*?)\n```'
            matches2 = re.findall(pattern2, text, re.DOTALL | re.IGNORECASE)
            pairs.extend(matches2)
        
        return pairs
    
    def _clean_code_snippet(self, code: str) -> str:
        """Clean code snippet (remove markdown, fix indentation)"""
        # Remove markdown code blocks
        code = re.sub(r'```python|```', '', code).strip()
        
        # Remove comment markers
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            # Skip pure comment lines
            if line.strip().startswith('#') and not any(c.isalnum() for c in line.replace('#', '')):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _fallback_parse_files(
        self,
        response: str,
        current_solution: Dict[str, str],
        reasoning: str
    ) -> Tuple[bool, Dict[str, str], str]:
        """
        Fallback: extract complete file modifications if diff parsing fails.
        """
        self.log("  Fallback: attempting to extract complete file modifications")
        
        # Look for file markers
        pattern = r'===\s*([^\s]+\.py)\s*===\s*```python\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            modified_files = dict(current_solution)  # Copy
            for filename, code in matches:
                if filename in modified_files:
                    modified_files[filename] = code.strip()
                    self.log(f"  Extracted modified file: {filename}")
            
            return True, modified_files, reasoning
        
        return False, {}, reasoning