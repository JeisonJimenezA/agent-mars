# memory/lesson_extractor.py
from typing import Optional, Dict
import json
import re
import difflib

from memory.lesson_types import SolutionLesson, DebugLesson, LessonType
from memory.lesson_pool import LessonPool
from core.tree_node import TreeNode
from llm.deepseek_client import get_client
from llm.prompt_manager import get_prompt_manager

class LessonExtractor:
    """
    Extracts lessons from execution results.
    Implements comparative analysis from Section 4.3.
    """
    
    def __init__(self, lesson_pool: LessonPool):
        self.lesson_pool = lesson_pool
        self.client = get_client()
        self.prompt_manager = get_prompt_manager()
    
    def extract_solution_lesson(
        self,
        new_node: TreeNode,
        best_node: Optional[TreeNode] = None,
        review_findings: str = "",
    ) -> Optional[SolutionLesson]:
        """
        Extract lesson by comparing new solution with best solution.

        Implements the TWO-STAGE process from Section 4.3:
        1. Empirical Analysis: Analyze execution output + review findings
        2. Lesson Distillation: Compare with best solution using Stage 1 findings

        Args:
            new_node: The newly executed node
            best_node: The current best node (None if this is first)
            review_findings: Review findings from ReviewAgent (stored in ExecutionResult)

        Returns:
            SolutionLesson or None if extraction failed
        """
        print(f"\n[Lesson Extraction] Two-stage analysis for node {new_node.id}")

        # Prepare context
        new_solution_str = self._format_solution(new_node)
        new_results_str = self._format_execution_results(new_node)

        if best_node is not None:
            best_solution_str = self._format_solution(best_node)
            code_diff = self._compute_code_diff(best_node, new_node)
        else:
            best_solution_str = "No previous solution"
            code_diff = "No previous solution to compare."

        # Use review_findings from argument or from node's execution result
        if not review_findings and new_node.execution_result:
            review_findings = new_node.execution_result.review_findings or ""

        try:
            # ══════════════════════════════════════════════════════════════
            # STAGE 1: Empirical Analysis
            # ══════════════════════════════════════════════════════════════
            print("  [Stage 1] Empirical Analysis...")
            empirical_findings = self._empirical_analysis(new_node, review_findings)

            # ══════════════════════════════════════════════════════════════
            # STAGE 2: Lesson Distillation
            # ══════════════════════════════════════════════════════════════
            print("  [Stage 2] Lesson Distillation...")
            prompt = self.prompt_manager.get_prompt(
                "solution_lesson",
                best_solution=best_solution_str,
                new_solution=new_solution_str,
                code_diff=code_diff,
                empirical_findings_text=empirical_findings,
            )

            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are an expert ML engineer distilling lessons from solution comparisons."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500,
            )

            # Parse response
            lesson_data = self._parse_solution_lesson_response(response["content"])

            if lesson_data:
                # Create lesson
                lesson = SolutionLesson(
                    id=self.lesson_pool.generate_lesson_id(LessonType.SOLUTION),
                    type=LessonType.SOLUTION,
                    title=lesson_data["title"],
                    summary=lesson_data["summary"],
                    empirical_findings=lesson_data["empirical_findings"] or empirical_findings,
                    key_lesson=lesson_data["key_lesson"],
                    source_node_id=new_node.id,
                    old_metric=best_node.metric_value if best_node else None,
                    new_metric=new_node.metric_value,
                    metric_delta=self._compute_metric_delta(new_node, best_node),
                    old_time=best_node.execution_time if best_node else None,
                    new_time=new_node.execution_time,
                    time_delta=self._compute_time_delta(new_node, best_node),
                )

                # Add to pool (with deduplication)
                if self.lesson_pool.add_lesson(lesson, check_duplicate=True):
                    return lesson

            return None

        except Exception as e:
            print(f"  ✗ Failed to extract solution lesson: {e}")
            return None

    def _empirical_analysis(self, node: TreeNode, review_findings: str = "") -> str:
        """
        Stage 1: Empirical Analysis of execution output.

        Produces structured findings about performance, training behavior,
        strengths, weaknesses, and quantitative metrics.

        Args:
            node: The executed node
            review_findings: Findings from ReviewAgent

        Returns:
            String with structured empirical findings
        """
        # Get execution output
        execution_output = ""
        if node.execution_result:
            execution_output = node.execution_result.stdout[-2000:]  # Last 2000 chars
            if node.execution_result.stderr:
                execution_output += f"\n[stderr]: {node.execution_result.stderr[-500:]}"

        # Get idea
        idea = node.solution.idea if node.solution else ""

        # Get prompt for Stage 1
        prompt = self.prompt_manager.get_prompt(
            "empirical_analysis",
            execution_output=execution_output,
            review_findings=review_findings or "No review findings available.",
            idea=idea[:500],
        )

        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are an expert ML engineer analyzing execution results."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=800,
            )
            return response["content"].strip()
        except Exception as e:
            print(f"    ✗ Stage 1 failed: {e}")
            # Fallback: return basic info
            return f"Metric: {node.metric_value}, Time: {node.execution_time}s"
    
    def extract_debug_lesson(
        self,
        buggy_node: TreeNode,
        fixed_node: TreeNode,
        error_analysis: str
    ) -> Optional[DebugLesson]:
        """
        Extract lesson from debugging attempt.
        
        Args:
            buggy_node: The node with the error
            fixed_node: The node with the fix
            error_analysis: Analysis of the error
            
        Returns:
            DebugLesson or None if extraction failed
        """
        print(f"\n[Debug Lesson] Analyzing fix from node {fixed_node.id}")
        
        # Prepare context
        source_files = self._format_solution(buggy_node)
        source_error = buggy_node.execution_result.stderr if buggy_node.execution_result else ""
        
        # Get diff between buggy and fixed
        diff = self._compute_code_diff(buggy_node, fixed_node)
        
        final_result = "Success" if fixed_node.execution_result.success else "Still failed"
        
        # Get prompt
        prompt = self.prompt_manager.get_prompt(
            "debug_lesson",
            source_files=source_files,
            source_exec_result=source_error,
            source_error_analysis=error_analysis,
            diff=diff,
            final_exec_result=final_result,
        )
        
        # Call LLM
        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are an expert Python debugger."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1200,
            )
            
            # Parse response
            lesson_data = self._parse_debug_lesson_response(response["content"])
            
            if lesson_data:
                lesson = DebugLesson(
                    id=self.lesson_pool.generate_lesson_id(LessonType.DEBUG),
                    type=LessonType.DEBUG,
                    title=lesson_data["title"],
                    explanation=lesson_data["explanation"],
                    detection=lesson_data["detection"],
                    fix_description=lesson_data.get("fix", ""),
                    error_type=self._extract_error_type(source_error),
                    source_node_id=fixed_node.id,
                )
                
                # Add to pool
                if self.lesson_pool.add_lesson(lesson, check_duplicate=True):
                    return lesson
            
            return None
            
        except Exception as e:
            print(f"  ✗ Failed to extract debug lesson: {e}")
            return None
    
    def _format_solution(self, node: TreeNode) -> str:
        """Format solution as string"""
        solution = node.solution
        
        text = f"Idea: {solution.idea}\n\n"
        text += "Files:\n"
        
        for filename, code in solution.get_all_files().items():
            text += f"\n=== {filename} ===\n"
            text += code
            text += "\n"
        
        return text
    
    def _format_execution_results(self, node: TreeNode) -> str:
        """Format execution results as string"""
        result = node.execution_result
        
        if result is None:
            return "Not executed"
        
        text = f"Success: {result.success}\n"
        text += f"Metric: {node.metric_value}\n"
        text += f"Execution Time: {node.execution_time}s\n"
        text += f"\nStdout:\n{result.stdout[-1000:]}\n"  # Last 1000 chars
        
        if result.stderr:
            text += f"\nStderr:\n{result.stderr[-1000:]}\n"
        
        return text
    
    def _compute_metric_delta(self, new_node: TreeNode, old_node: Optional[TreeNode]) -> Optional[float]:
        """Compute metric improvement"""
        if old_node is None or new_node.metric_value is None:
            return None
        
        if old_node.metric_value is None:
            return None
        
        return new_node.metric_value - old_node.metric_value
    
    def _compute_time_delta(self, new_node: TreeNode, old_node: Optional[TreeNode]) -> Optional[float]:
        """Compute time difference"""
        if old_node is None:
            return None
        
        return new_node.execution_time - old_node.execution_time
    
    def _compute_code_diff(self, node1: TreeNode, node2: TreeNode) -> str:
        """
        Compute unified diff between two solutions using difflib.
        This gives the Lesson Distillation Agent precise information
        about what code changes caused performance shifts.
        """
        files1 = node1.solution.get_all_files()
        files2 = node2.solution.get_all_files()

        diff_parts = []

        all_files = set(list(files1.keys()) + list(files2.keys()))
        for filename in sorted(all_files):
            old = files1.get(filename, "")
            new = files2.get(filename, "")

            if old == new:
                continue

            if not old:
                diff_parts.append(f"\n+++ Added file: {filename} ({len(new.splitlines())} lines)")
                continue

            if not new:
                diff_parts.append(f"\n--- Removed file: {filename}")
                continue

            # Unified diff (limit to 500 lines to avoid prompt bloat)
            udiff = list(difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{filename}",
                tofile=f"b/{filename}",
                n=3,
            ))

            if udiff:
                diff_text = "".join(udiff[:500])
                if len(udiff) > 500:
                    diff_text += f"\n... ({len(udiff) - 500} more diff lines truncated)\n"
                diff_parts.append(diff_text)

        if not diff_parts:
            return "No code differences detected."

        return "Code Changes:\n" + "\n".join(diff_parts)
    
    def _parse_solution_lesson_response(self, response: str) -> Optional[Dict]:
        """Parse LLM response for solution lesson"""
        try:
            # Expected format from prompt:
            # Title: ...
            # Summary: ...
            # Empirical Findings: ...
            # Key Lesson: ...
            
            data = {}
            
            # Extract sections using regex
            title_match = re.search(r"Title:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
            summary_match = re.search(r"Summary:\s*(.+?)(?:\n(?:Empirical|Key|$))", response, re.IGNORECASE | re.DOTALL)
            findings_match = re.search(r"Empirical Findings?:\s*(.+?)(?:\n(?:Key|$))", response, re.IGNORECASE | re.DOTALL)
            lesson_match = re.search(r"Key Lesson:\s*(.+?)$", response, re.IGNORECASE | re.DOTALL)
            
            if title_match:
                data["title"] = title_match.group(1).strip()
            else:
                data["title"] = "Untitled Lesson"
            
            data["summary"] = summary_match.group(1).strip() if summary_match else ""
            data["empirical_findings"] = findings_match.group(1).strip() if findings_match else ""
            data["key_lesson"] = lesson_match.group(1).strip() if lesson_match else ""
            
            return data
            
        except Exception as e:
            print(f"  ✗ Failed to parse solution lesson: {e}")
            return None
    
    def _parse_debug_lesson_response(self, response: str) -> Optional[Dict]:
        """Parse LLM response for debug lesson"""
        try:
            data = {}
            
            title_match = re.search(r"Title:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
            explanation_match = re.search(r"Explanation:\s*(.+?)(?:\nDetection|$)", response, re.IGNORECASE | re.DOTALL)
            detection_match = re.search(r"Detection:\s*(.+?)(?:\nFix|$)", response, re.IGNORECASE | re.DOTALL)
            fix_match = re.search(r"Fix:\s*(.+?)$", response, re.IGNORECASE | re.DOTALL)
            
            data["title"] = title_match.group(1).strip() if title_match else "Debug Lesson"
            data["explanation"] = explanation_match.group(1).strip() if explanation_match else ""
            data["detection"] = detection_match.group(1).strip() if detection_match else ""
            data["fix"] = fix_match.group(1).strip() if fix_match else ""
            
            return data
            
        except Exception as e:
            print(f"  ✗ Failed to parse debug lesson: {e}")
            return None
    
    def _extract_error_type(self, stderr: str) -> str:
        """Extract error type from stderr"""
        if not stderr:
            return "Unknown"
        
        # Common Python errors
        error_types = [
            "AttributeError", "TypeError", "ValueError", "KeyError",
            "IndexError", "ImportError", "ModuleNotFoundError",
            "FileNotFoundError", "RuntimeError", "MemoryError",
            "SyntaxError", "IndentationError", "NameError"
        ]
        
        for error_type in error_types:
            if error_type in stderr:
                return error_type
        
        return "Unknown"