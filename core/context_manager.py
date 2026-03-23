# core/context_manager.py
"""
Context Budget Manager for MARS.

Implements intelligent context allocation to prevent arbitrary truncation.
Ensures critical information (errors, relevant code) gets priority over
less important content (full code, old lessons).

Based on MARS paper recommendations for context management.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import tiktoken
import re


@dataclass
class ContentBlock:
    """A block of content with metadata for prioritization."""
    name: str
    content: str
    priority: float  # 0.0 to 1.0, higher = more important
    min_chars: int = 0  # Minimum characters to keep (0 = can be dropped entirely)
    category: str = "general"  # For grouping related content

    @property
    def char_count(self) -> int:
        return len(self.content)


@dataclass
class ContextBudget:
    """
    Manages context window allocation across different content types.

    Ensures that high-priority content (error traces, relevant code) is never
    truncated while lower-priority content is reduced as needed.

    Priority levels (higher = more important):
    - 1.0: Error tracebacks, current error analysis (NEVER truncate)
    - 0.9: Code directly referenced in error (HIGH priority)
    - 0.8: Recent lessons (last 5)
    - 0.7: Main script code
    - 0.6: Module code
    - 0.5: Older lessons
    - 0.4: EDA report, metadata
    - 0.3: Previous ideas
    - 0.2: Full file contents (can be reduced to signatures)
    """

    # Default priorities by content type
    DEFAULT_PRIORITIES = {
        "error_traceback": 1.0,
        "error_analysis": 1.0,
        "error_file_code": 0.95,
        "relevant_code": 0.9,
        "recent_lessons": 0.85,
        "debug_lessons": 0.85,
        "main_script": 0.75,
        "module_code": 0.65,
        "older_lessons": 0.55,
        "eda_report": 0.45,
        "problem_description": 0.5,
        "previous_ideas": 0.35,
        "full_code": 0.25,
        "metadata": 0.2,
    }

    def __init__(
        self,
        max_tokens: int = 100000,
        reserve_for_response: int = 8000,
        model: str = "gpt-4"
    ):
        """
        Initialize context budget.

        Args:
            max_tokens: Maximum context window size
            reserve_for_response: Tokens to reserve for model response
            model: Model name for token counting (uses tiktoken)
        """
        self.max_tokens = max_tokens
        self.reserve_for_response = reserve_for_response
        self.available_tokens = max_tokens - reserve_for_response

        # Try to load tokenizer, fallback to char-based estimation
        try:
            self.encoding = tiktoken.encoding_for_model(model)
            self._use_tiktoken = True
        except Exception:
            self._use_tiktoken = False
            # Rough estimate: 1 token ~= 4 chars for code
            self._chars_per_token = 4

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        if self._use_tiktoken:
            return len(self.encoding.encode(text))
        return len(text) // self._chars_per_token

    def allocate(
        self,
        content_blocks: List[ContentBlock],
        target_tokens: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Allocate context budget across content blocks.

        High-priority content is preserved, low-priority content is truncated
        or dropped as needed to fit within budget.

        Args:
            content_blocks: List of ContentBlock objects to allocate
            target_tokens: Target token count (defaults to available_tokens)

        Returns:
            Dict mapping block names to (possibly truncated) content
        """
        target = target_tokens or self.available_tokens

        # Sort by priority (highest first)
        sorted_blocks = sorted(content_blocks, key=lambda b: b.priority, reverse=True)

        # Phase 1: Calculate total tokens needed
        total_needed = sum(self.count_tokens(b.content) for b in sorted_blocks)

        # If we fit, return everything
        if total_needed <= target:
            return {b.name: b.content for b in sorted_blocks}

        # Phase 2: Progressive truncation
        result = {}
        remaining_tokens = target

        for block in sorted_blocks:
            block_tokens = self.count_tokens(block.content)

            if block_tokens <= remaining_tokens:
                # Block fits entirely
                result[block.name] = block.content
                remaining_tokens -= block_tokens
            elif block.priority >= 0.9:
                # High priority: truncate but keep substantial portion
                allocated = max(remaining_tokens, block.min_chars // self._chars_per_token)
                result[block.name] = self._smart_truncate(
                    block.content,
                    allocated * self._chars_per_token,
                    preserve_structure=True
                )
                remaining_tokens = max(0, remaining_tokens - allocated)
            elif block.priority >= 0.6:
                # Medium priority: truncate more aggressively
                allocated = remaining_tokens // 2
                if allocated > block.min_chars // self._chars_per_token:
                    result[block.name] = self._smart_truncate(
                        block.content,
                        allocated * self._chars_per_token,
                        preserve_structure=True
                    )
                    remaining_tokens -= allocated
                elif block.min_chars > 0:
                    # Keep minimum
                    result[block.name] = self._smart_truncate(
                        block.content, block.min_chars, preserve_structure=False
                    )
            else:
                # Low priority: only include if we have space
                if remaining_tokens > 500:
                    allocated = min(remaining_tokens // 3, block_tokens)
                    result[block.name] = self._smart_truncate(
                        block.content,
                        allocated * self._chars_per_token,
                        preserve_structure=False
                    )
                    remaining_tokens -= allocated
                else:
                    # Drop entirely
                    result[block.name] = ""

        return result

    def _smart_truncate(
        self,
        content: str,
        max_chars: int,
        preserve_structure: bool = True
    ) -> str:
        """
        Truncate content intelligently.

        For code, preserves:
        - Imports at the top
        - Class/function definitions
        - The section around errors (if detectable)

        Args:
            content: Content to truncate
            max_chars: Maximum characters to keep
            preserve_structure: If True, try to keep structural elements
        """
        if len(content) <= max_chars:
            return content

        if not preserve_structure:
            # Simple truncation with indicator
            return content[:max_chars - 50] + "\n\n... [truncated for context limit]"

        # For code, try to preserve structure
        lines = content.split('\n')

        # Categorize lines
        imports = []
        definitions = []  # class, def, @decorator lines
        other = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                imports.append((i, line))
            elif stripped.startswith(('class ', 'def ', '@')):
                # Include next line too (docstring or first line of body)
                definitions.append((i, line))
                if i + 1 < len(lines):
                    definitions.append((i + 1, lines[i + 1]))
            else:
                other.append((i, line))

        # Build truncated version
        result_parts = []
        current_chars = 0

        # Always include imports (up to 20% of budget)
        import_budget = max_chars // 5
        for _, line in imports:
            if current_chars + len(line) < import_budget:
                result_parts.append(line)
                current_chars += len(line) + 1

        if result_parts:
            result_parts.append("")  # Blank line after imports

        # Include definitions (up to 40% of budget)
        def_budget = max_chars * 2 // 5
        for _, line in definitions:
            if current_chars + len(line) < def_budget + import_budget:
                result_parts.append(line)
                current_chars += len(line) + 1

        # Fill remaining with other content
        remaining = max_chars - current_chars - 50  # Reserve for truncation notice
        if remaining > 100:
            other_content = '\n'.join(line for _, line in other[:50])
            if len(other_content) > remaining:
                other_content = other_content[:remaining]
            result_parts.append(other_content)

        result_parts.append("\n# ... [truncated for context limit]")

        return '\n'.join(result_parts)

    def extract_error_relevant_code(
        self,
        all_files: Dict[str, str],
        error_traceback: str,
        context_lines: int = 30
    ) -> Dict[str, str]:
        """
        Extract only the code sections relevant to an error.

        Parses the traceback to find file:line references and extracts
        surrounding context from those files.

        Args:
            all_files: Dict of filename -> code
            error_traceback: Error traceback text
            context_lines: Lines of context around error location

        Returns:
            Dict of filename -> relevant code section
        """
        # Parse error locations from traceback
        # Pattern: File "...filename.py", line N
        locations = re.findall(
            r'File\s+"[^"]*[\\/](\w+\.py)"\s*,\s*line\s+(\d+)',
            error_traceback
        )

        if not locations:
            # No specific locations found, return main.py truncated
            if "main.py" in all_files:
                return {"main.py": all_files["main.py"][:5000]}
            return {}

        relevant = {}

        for filename, line_str in locations:
            if filename not in all_files:
                continue

            line_num = int(line_str)
            code = all_files[filename]
            lines = code.split('\n')

            # Extract context around error line
            start = max(0, line_num - context_lines)
            end = min(len(lines), line_num + context_lines)

            # Include imports from the top
            import_lines = []
            for i, line in enumerate(lines[:30]):
                if line.strip().startswith(('import ', 'from ')):
                    import_lines.append(line)

            # Build relevant section
            section_lines = []
            if import_lines:
                section_lines.extend(import_lines)
                section_lines.append("")
                section_lines.append(f"# ... [lines {len(import_lines)+1}-{start} omitted]")
                section_lines.append("")

            # Add context around error
            for i in range(start, end):
                prefix = ">>> " if i == line_num - 1 else "    "
                section_lines.append(f"{prefix}{lines[i]}")

            if end < len(lines):
                section_lines.append("")
                section_lines.append(f"# ... [{len(lines) - end} more lines]")

            relevant[filename] = '\n'.join(section_lines)

        return relevant

    def build_debug_context(
        self,
        error_traceback: str,
        error_analysis: str,
        all_files: Dict[str, str],
        debug_lessons: str,
        problem_description: str,
    ) -> Dict[str, str]:
        """
        Build optimized context for debugging.

        Prioritizes error-relevant information while keeping context
        within budget.

        Args:
            error_traceback: Full error traceback
            error_analysis: Analysis of the error
            all_files: All solution files
            debug_lessons: Formatted debug lessons
            problem_description: Problem description

        Returns:
            Dict with allocated content for each section
        """
        # Extract error-relevant code
        relevant_code = self.extract_error_relevant_code(all_files, error_traceback)
        relevant_code_str = "\n\n".join(
            f"=== {fname} (ERROR CONTEXT) ===\n{code}"
            for fname, code in relevant_code.items()
        )

        # Format remaining files (signatures only for large files)
        other_files = []
        for fname, code in all_files.items():
            if fname not in relevant_code:
                if len(code) > 2000:
                    # Extract signatures only
                    sigs = self._extract_signatures(code)
                    other_files.append(f"=== {fname} (signatures) ===\n{sigs}")
                else:
                    other_files.append(f"=== {fname} ===\n{code}")
        other_files_str = "\n\n".join(other_files)

        # Create content blocks
        blocks = [
            ContentBlock("error_traceback", error_traceback, 1.0, min_chars=500),
            ContentBlock("error_analysis", error_analysis, 0.95, min_chars=200),
            ContentBlock("relevant_code", relevant_code_str, 0.9, min_chars=500),
            ContentBlock("debug_lessons", debug_lessons, 0.8, min_chars=0),
            ContentBlock("other_files", other_files_str, 0.5, min_chars=0),
            ContentBlock("problem_description", problem_description, 0.4, min_chars=100),
        ]

        return self.allocate(blocks)

    def build_lesson_context(
        self,
        new_solution: str,
        best_solution: str,
        code_diff: str,
        empirical_findings: str,
        execution_output: str,
    ) -> Dict[str, str]:
        """
        Build optimized context for lesson extraction.

        Args:
            new_solution: New solution code/idea
            best_solution: Best solution for comparison
            code_diff: Diff between solutions
            empirical_findings: Stage 1 findings
            execution_output: Execution logs

        Returns:
            Dict with allocated content for each section
        """
        blocks = [
            ContentBlock("empirical_findings", empirical_findings, 1.0, min_chars=200),
            ContentBlock("code_diff", code_diff, 0.95, min_chars=300),
            ContentBlock("new_solution", new_solution, 0.85, min_chars=500),
            ContentBlock("best_solution", best_solution, 0.7, min_chars=300),
            ContentBlock("execution_output", execution_output, 0.5, min_chars=0),
        ]

        return self.allocate(blocks)

    def _extract_signatures(self, code: str, max_lines: int = 40) -> str:
        """Extract function/class signatures from code."""
        lines = code.split('\n')
        result = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                result.append(line)
            elif stripped.startswith(('class ', 'def ', '@')):
                result.append(line)
                # Include docstring if present
                if i + 1 < len(lines) and '"""' in lines[i + 1]:
                    result.append(lines[i + 1])

            if len(result) >= max_lines:
                result.append("# ... (more definitions)")
                break

        return '\n'.join(result)


# Global instance
_context_manager: Optional[ContextBudget] = None


def get_context_manager(max_tokens: int = 100000) -> ContextBudget:
    """Get or create global context manager."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextBudget(max_tokens=max_tokens)
    return _context_manager
