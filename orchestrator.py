# orchestrator.py
from pathlib import Path
from typing import Optional, Dict, List
import time
import json
import re

from core.config import Config
from utils.debug_logger import get_debug_logger
from core.mcts import MCTSEngine
from core.tree_node import TreeNode, Solution, ActionType, NodeStatus, ExecutionResult
from memory.lesson_pool import LessonPool
from memory.lesson_extractor import LessonExtractor
from agents.idea_agent import IdeaAgent
from agents.modular_agent import ModularAgent
from agents.coding_agent import CodingAgent
from agents.debug_agent import DebugAgent
from agents.review_agent import ReviewAgent
from agents.solution_improver import SolutionImprover
from agents.validation_agent import ValidationAgent
from utils.file_manager import FileManager
from execution.executor import Executor, ExecutionConfig
from execution.validator import SolutionValidator

class MARSOrchestrator:
    """
    Main orchestrator that coordinates all MARS components.
    Implements the complete workflow from Algorithm 2 in the paper.
    """

    def __init__(
        self,
        problem_description: str,
        eda_report: str,
        metadata_dir: Path,
        data_dir: Optional[Path] = None,
        time_budget: int = 3600,
        working_dir: Optional[Path] = None,
        lower_is_better: bool = False,
    ):
        self.problem_description = problem_description
        self.eda_report = eda_report
        self.metadata_dir = Path(metadata_dir)
        self.data_dir = Path(data_dir) if data_dir else None
        self.time_budget = time_budget
        self.working_dir = working_dir or Config.WORKING_DIR
        self.lower_is_better = lower_is_better

        # Initialize components
        self.mcts = MCTSEngine(time_budget=time_budget, lower_is_better=lower_is_better)
        self.lesson_pool = LessonPool()

        # Load lessons from previous runs if they exist
        lesson_path = self.working_dir / "lessons.json"
        if lesson_path.exists():
            try:
                self.lesson_pool.load_from_file(lesson_path)
                print(f"[Orchestrator] Loaded lessons from {lesson_path}")
            except Exception as e:
                print(f"[Orchestrator] Warning: Could not load lessons: {e}")

        self.lesson_extractor = LessonExtractor(self.lesson_pool)

        # Initialize agents
        self.idea_agent = IdeaAgent()
        self.modular_agent = ModularAgent()
        self.coding_agent = CodingAgent()
        self.debug_agent = DebugAgent()
        self.review_agent = ReviewAgent()
        self.solution_improver = SolutionImprover()
        self.validation_agent = ValidationAgent()  # Mejora 4: Validation verification

        # Initialize utilities
        self.file_manager = FileManager(self.working_dir)
        self.executor = Executor(ExecutionConfig(timeout=1800))
        self.validator = SolutionValidator()
        self.debug_logger = get_debug_logger()

        # Statistics
        self.start_time = 0
        self.iterations = 0

        # Tracking for curriculum exploration (Mejora 3)
        self.valid_solutions_count = 0
        self.stagnation_count = 0
        self.last_best_metric: Optional[float] = None

        # Validation verification limit (Mejora 4)
        self._validation_verified_count = 0
        self._MAX_VALIDATION_CHECKS = 3  # Only verify first 3 valid solutions

    def run(self) -> TreeNode:
        """
        Execute the complete MARS workflow (Algorithm 2).

        The key difference from the previous implementation:
        - MCTS only produces DRAFT and IMPROVE actions.
        - After generating + executing a node, if it is BUGGY,
          an **internal debugging loop** retries up to Nd times
          on the SAME node before moving to the next MCTS iteration.

        Returns:
            Best solution node found
        """
        print("=" * 70)
        print("MARS: Modular Agent with Reflective Search")
        print("=" * 70)
        print(f"Time budget: {self.time_budget}s ({self.time_budget / 3600:.1f}h)")
        print(f"Working directory: {self.working_dir}")
        print(f"Metric direction: {'minimize' if self.lower_is_better else 'maximize'}")
        print("=" * 70 + "\n")

        self.start_time = time.time()

        # Main MCTS loop
        for new_node in self.mcts.search():
            self.iterations += 1
            elapsed = time.time() - self.start_time

            print(f"\n{'=' * 70}")
            print(f"ITERATION {self.iterations} | Elapsed: {elapsed:.1f}s / {self.time_budget}s")
            print(f"{'=' * 70}")

            # ----------------------------------------------------------
            # Step 1: Generate solution (DRAFT or IMPROVE only)
            # ----------------------------------------------------------
            success = self._generate_solution(new_node)

            if not success:
                self.log("Failed to generate solution, marking as buggy")
                new_node.status = NodeStatus.BUGGY
                self.mcts.update_after_execution(
                    new_node,
                    ExecutionResult(success=False, error_message="Code generation failed"),
                )
                self._save_progress()
                continue

            # ----------------------------------------------------------
            # Step 2: Execute solution
            # ----------------------------------------------------------
            self._execute_solution(new_node)

            # ----------------------------------------------------------
            # Step 3: Internal Debugging Loop (Algorithm 2, lines 27-30)
            #   while IsBuggy(v_new) and k < Nd:
            #       v_new <- DebugNode(v_new, L_debug)
            #       k <- k + 1
            # ----------------------------------------------------------
            debug_attempt = 0
            max_debug = Config.ND

            while new_node.status == NodeStatus.BUGGY and debug_attempt < max_debug:
                debug_attempt += 1
                print(f"\n  [Debug Loop] Attempt {debug_attempt}/{max_debug}")

                fixed = self._debug_solution_inline(new_node)
                if not fixed:
                    print(f"  [Debug Loop] Could not generate fix, stopping debug loop")
                    break

                # Re-execute the fixed solution
                self._execute_solution(new_node)

                # Extract debug lesson from this attempt
                self._extract_debug_lesson_inline(new_node, debug_attempt)

            # ----------------------------------------------------------
            # Step 4: Validation verification (Mejora 4)
            # ----------------------------------------------------------
            if new_node.status == NodeStatus.VALID:
                self._verify_validation(new_node)

            # ----------------------------------------------------------
            # Step 5: Review execution result (ExecuteAndReview)
            # ----------------------------------------------------------
            if new_node.status == NodeStatus.VALID:
                self._review_execution(new_node)

            # ----------------------------------------------------------
            # Step 6: Update MCTS tree (backpropagation)
            # ----------------------------------------------------------
            self.mcts.update_after_execution(new_node, new_node.execution_result)

            # ----------------------------------------------------------
            # Step 7: Track stagnation and valid solutions (Mejora 3)
            # ----------------------------------------------------------
            if new_node.status == NodeStatus.VALID:
                self.valid_solutions_count += 1
                # Check for stagnation (no improvement in best metric)
                current_best = self.mcts.best_node.metric_value if self.mcts.best_node else None
                if self.last_best_metric is not None and current_best == self.last_best_metric:
                    self.stagnation_count += 1
                else:
                    self.stagnation_count = 0
                self.last_best_metric = current_best

            # ----------------------------------------------------------
            # Step 8: Extract lessons from valid solutions
            # ----------------------------------------------------------
            if new_node.status == NodeStatus.VALID:
                self._extract_lessons(new_node)

            # Save progress
            self._save_progress()

        # Search complete
        print("\n" + "=" * 70)
        print("SEARCH COMPLETE")
        print("=" * 70)
        self._print_final_statistics()

        return self.mcts.best_node

    # ==================================================================
    # Solution Generation (DRAFT / IMPROVE only — no DEBUG action)
    # ==================================================================

    def _generate_solution(self, node: TreeNode) -> bool:
        """
        Generate solution code for a node based on its action type.
        Only DRAFT and IMPROVE are valid here; DEBUG is handled inline.
        """
        action = node.action_type
        self.log(f"Action: {action.value if action else 'root'}")

        try:
            if action == ActionType.DRAFT:
                return self._draft_solution(node)
            elif action == ActionType.IMPROVE:
                return self._improve_solution(node)
            else:
                self.log(f"Unexpected action type: {action}")
                return False
        except Exception as e:
            self.log(f"Exception during solution generation: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _draft_solution(self, node: TreeNode) -> bool:
        """Draft a new solution from scratch (Algorithm 2, lines 16-22)."""

        # Curriculum-based idea generation (Mejora 3)
        if len(self.idea_agent.generated_ideas) == 0:
            idea = self.idea_agent.generate_initial_idea(
                self.problem_description,
                self.eda_report,
                [],
            )
        else:
            idea = self.idea_agent.improve_idea(
                self.problem_description,
                self.eda_report,
                self.lesson_pool,
                previous_ideas=self.idea_agent.generated_ideas,
                valid_solutions_count=self.valid_solutions_count,
                stagnation_count=self.stagnation_count,
            )

        if not idea:
            return False

        # Decompose into modules
        modules_desc = self.modular_agent.decompose_idea(
            self.problem_description, idea
        )
        if not modules_desc:
            return False

        is_valid, issues = self.modular_agent.validate_decomposition(modules_desc)
        if not is_valid:
            self.log(f"Invalid decomposition: {issues}")
            return False

        # Implement modules
        implemented_modules = {}
        for module_name, module_desc in modules_desc.items():
            if module_name == "main":
                continue

            code = self.coding_agent.implement_module(
                self.problem_description,
                idea,
                f"{module_name}.py",
                module_desc,
                implemented_modules,
                self.lesson_pool,
                eda_report=self.eda_report,
            )

            if code:
                implemented_modules[f"{module_name}.py"] = code
            else:
                self.log(f"Failed to implement {module_name}")

        # Unit-test modules (Algorithm 2, line 19-20)
        implemented_modules = self._test_and_debug_modules(
            idea, implemented_modules, modules_desc
        )

        # Implement main script
        main_code = self.coding_agent.implement_main_script(
            self.problem_description,
            idea,
            implemented_modules,
            self.lesson_pool,
            eda_report=self.eda_report,
        )
        if not main_code:
            return False

        node.solution = Solution(
            idea=idea,
            modules=implemented_modules,
            main_script=main_code,
            module_descriptions=modules_desc,
        )

        self.log(f"Solution drafted: {len(implemented_modules)} modules + main")
        return True

    def _improve_solution(self, node: TreeNode) -> bool:
        """Improve existing solution (Section 4.4.1)."""

        parent = node.parent
        if not parent or not parent.solution:
            self.log("No parent solution to improve")
            return False

        self.log(f"Improving solution from parent node {parent.id}")
        self.log(f"  Parent metric: {parent.metric_value}")

        success, improved_files, reasoning = self.solution_improver.propose_improvements(
            problem_description=self.problem_description,
            current_solution=parent.solution.get_all_files(),
            current_metric=parent.metric_value,
            lesson_pool=self.lesson_pool,
        )

        if not success:
            self.log("Failed to generate improvements, using parent solution")
            node.solution = Solution(
                idea=parent.solution.idea + "\n[Attempted improvement - fallback to parent]",
                modules=parent.solution.modules.copy(),
                main_script=parent.solution.main_script,
                module_descriptions=parent.solution.module_descriptions.copy(),
            )
            return True

        # Validate syntax
        from utils.code_parser import CodeParser
        parser = CodeParser()

        syntax_errors = []
        for filename, code in improved_files.items():
            if filename.endswith(".py"):
                is_valid, error = parser.validate_syntax(code)
                if not is_valid:
                    syntax_errors.append(f"{filename}: {error}")

        if syntax_errors:
            self.log(f"Syntax errors in improved solution: {syntax_errors}")
            self.log("Falling back to parent solution")
            node.solution = Solution(
                idea=parent.solution.idea,
                modules=parent.solution.modules.copy(),
                main_script=parent.solution.main_script,
                module_descriptions=parent.solution.module_descriptions.copy(),
            )
            return True

        # Build improved solution
        new_modules = {}
        new_main = parent.solution.main_script
        for filename, code in improved_files.items():
            if filename == "main.py":
                new_main = code
            elif filename.endswith(".py"):
                new_modules[filename] = code

        node.solution = Solution(
            idea=parent.solution.idea + f"\n\n[Improvement]: {reasoning}",
            modules=new_modules,
            main_script=new_main,
            module_descriptions=parent.solution.module_descriptions.copy(),
        )

        self.log(f"Solution improved: modified {len(improved_files)} files")
        return True

    # ==================================================================
    # Internal Debugging (inline, not a separate MCTS action)
    # ==================================================================

    def _debug_solution_inline(self, node: TreeNode) -> bool:
        """
        Debug a buggy node **in-place** (Algorithm 2, lines 27-30).

        Uses multiple strategies to fix errors:
        1. Diff-based fix via DebugAgent (with fallbacks)
        2. Direct file regeneration for the error file
        3. Environment-specific fixes (emojis, parquet, etc.)

        Modifies node.solution directly so the next execution uses the fix.
        Returns True if a fix was generated, False otherwise.
        """
        if not node.solution or not node.execution_result:
            return False

        error_output = node.execution_result.stderr or node.execution_result.error_message
        all_files = node.solution.get_all_files()

        # ══════════════════════════════════════════════════════════════
        # Strategy 1: Standard debug flow (diff-based with fallbacks)
        # ══════════════════════════════════════════════════════════════
        error_analysis = self.debug_agent.analyze_error(
            self.problem_description,
            all_files,
            error_output,
            self.lesson_pool,
        )

        if error_analysis:
            fixed_files = self.debug_agent.fix_error(
                self.problem_description,
                all_files,
                error_output,
                error_analysis,
                self.lesson_pool,
            )
            if fixed_files:
                self._apply_fixes(node, fixed_files)
                return True

        # ══════════════════════════════════════════════════════════════
        # Strategy 2: Direct regeneration of error file
        # ══════════════════════════════════════════════════════════════
        target_file = self._extract_error_file(error_output)
        if target_file and target_file in all_files:
            self.log(f"Strategy 2: Regenerating {target_file} directly")
            regenerated = self.debug_agent.regenerate_file(
                target_file=target_file,
                original_code=all_files[target_file],
                execution_error=error_output,
                error_analysis=error_analysis or "Error analysis not available",
            )
            if regenerated:
                self._apply_fixes(node, {target_file: regenerated})
                return True

        # ══════════════════════════════════════════════════════════════
        # Strategy 3: Quick environment fixes (no LLM needed)
        # ══════════════════════════════════════════════════════════════
        if self._apply_environment_fixes(node, error_output):
            self.log("Strategy 3: Applied environment fixes")
            return True

        self.log("All debug strategies failed")
        return False

    def _apply_fixes(self, node: TreeNode, fixed_files: Dict[str, str]):
        """Apply fixes to a node's solution in-place."""
        for filename, code in fixed_files.items():
            if filename == "main.py":
                node.solution.main_script = code
            else:
                node.solution.modules[filename] = code
        self.log(f"Applied fix to {len(fixed_files)} files")

    def _extract_error_file(self, error: str) -> Optional[str]:
        """Extract the filename that caused the error from traceback."""
        import re
        matches = re.findall(r'File\s+"[^"]*[\\/](\w+\.py)"', error)
        if matches:
            return matches[-1]  # Last match = deepest in call stack
        return None

    def _apply_environment_fixes(self, node: TreeNode, error: str) -> bool:
        """
        Apply quick environment-specific fixes without LLM.

        Handles common issues:
        - UnicodeEncodeError from emojis
        - ImportError from pyarrow/parquet
        """
        import re
        fixed_any = False
        all_files = node.solution.get_all_files()

        # Fix 1: Remove emojis causing UnicodeEncodeError
        if "UnicodeEncodeError" in error and "charmap" in error:
            emoji_pattern = re.compile(
                "["
                "\U0001F300-\U0001F9FF"
                "\U00002600-\U000026FF"
                "\U00002700-\U000027BF"
                "\U0001F600-\U0001F64F"
                "\U0001F680-\U0001F6FF"
                "]+",
                flags=re.UNICODE
            )
            for filename, code in all_files.items():
                if emoji_pattern.search(code):
                    cleaned = emoji_pattern.sub("", code)
                    if filename == "main.py":
                        node.solution.main_script = cleaned
                    else:
                        node.solution.modules[filename] = cleaned
                    self.log(f"  Removed emojis from {filename}")
                    fixed_any = True

        # Fix 2: Replace parquet with CSV
        if "pyarrow" in error.lower() or "parquet" in error.lower():
            for filename, code in all_files.items():
                if ".to_parquet(" in code or ".read_parquet(" in code:
                    fixed_code = code.replace(".to_parquet(", ".to_csv(")
                    fixed_code = fixed_code.replace(".read_parquet(", ".read_csv(")
                    fixed_code = fixed_code.replace(".parquet", ".csv")
                    if filename == "main.py":
                        node.solution.main_script = fixed_code
                    else:
                        node.solution.modules[filename] = fixed_code
                    self.log(f"  Replaced parquet with CSV in {filename}")
                    fixed_any = True

        return fixed_any

    def _extract_debug_lesson_inline(self, node: TreeNode, attempt: int):
        """Extract a debug lesson after an inline debugging attempt."""
        if not node.execution_result:
            return

        error_output = node.execution_result.stderr or node.execution_result.error_message
        status = "Success" if node.execution_result.success else "Still failing"

        try:
            self.lesson_extractor.extract_debug_lesson(
                node, node, f"Debug attempt {attempt}: {status}"
            )
        except Exception as e:
            self.log(f"Failed to extract debug lesson: {e}")

    # ==================================================================
    # Module Unit Testing (Algorithm 2, line 19)
    # ==================================================================

    def _test_and_debug_modules(
        self,
        idea: str,
        implemented_modules: Dict[str, str],
        modules_desc: Dict[str, str],
    ) -> Dict[str, str]:
        """
        Unit-test each module and attempt to fix failures.
        Implements DebugModules from Algorithm 2, line 19.
        """
        tested_modules = {}

        for filename, code in implemented_modules.items():
            module_name = filename.replace(".py", "")
            self.log(f"Testing module: {module_name}")

            # Generate test code
            test_code = self.coding_agent.generate_module_test(
                self.problem_description,
                idea,
                filename,
                code,
                implemented_modules,
            )

            if not test_code:
                self.log(f"  Could not generate test for {module_name}, keeping as-is")
                tested_modules[filename] = code
                continue

            # Write module + test to temp dir and execute
            test_dir = self.file_manager.create_solution_directory(f"test_{module_name}")
            test_files = {filename: code, f"test_{filename}": test_code}

            # Include dependencies
            for dep_name, dep_code in implemented_modules.items():
                if dep_name != filename:
                    test_files[dep_name] = dep_code

            self.file_manager.write_files(test_dir, test_files)
            if self.metadata_dir.exists():
                self.file_manager.copy_metadata(test_dir, self.metadata_dir)
            if self.data_dir and self.data_dir.exists():
                self.file_manager.copy_input_data(test_dir, self.data_dir)

            # Use auto-install for module tests too
            test_result = self.executor.execute_with_auto_install(
                test_dir / f"test_{filename}",
                working_dir=test_dir,
                timeout=120,
                max_install_attempts=2,
            )

            if test_result.success:
                self.log(f"  Module {module_name} passed tests")
                tested_modules[filename] = code
            else:
                self.log(f"  Module {module_name} failed tests, attempting fix")
                # One debug attempt for module
                fixed_code = self._debug_module(
                    idea, filename, code, test_result.stderr, modules_desc.get(module_name, "")
                )
                tested_modules[filename] = fixed_code if fixed_code else code

        return tested_modules

    def _debug_module(
        self, idea: str, filename: str, code: str, error: str, description: str
    ) -> Optional[str]:
        """Attempt to fix a single module that failed unit testing."""
        error_analysis = self.debug_agent.analyze_error(
            self.problem_description,
            {filename: code},
            error,
            self.lesson_pool,
        )
        if not error_analysis:
            return None

        fixed_files = self.debug_agent.fix_error(
            self.problem_description,
            {filename: code},
            error,
            error_analysis,
            self.lesson_pool,
        )
        if fixed_files and filename in fixed_files:
            return fixed_files[filename]
        return None

    # ==================================================================
    # Execution + Review
    # ==================================================================

    def _execute_solution(self, node: TreeNode):
        """Execute solution and collect results."""

        self.log("Running solution...")

        # Create solution directory
        solution_dir = self.file_manager.create_solution_directory(node.id)

        # Write files
        files = node.solution.get_all_files()
        self.file_manager.write_files(solution_dir, files)

        # Copy metadata
        if self.metadata_dir.exists():
            self.file_manager.copy_metadata(solution_dir, self.metadata_dir)

        # Copy input data
        if self.data_dir and self.data_dir.exists():
            self.file_manager.copy_input_data(solution_dir, self.data_dir)

        # Create working subdirectories
        self.file_manager.create_working_subdirs(solution_dir)

        # Execute main script with auto-install for missing dependencies
        main_path = solution_dir / "main.py"
        result = self.executor.execute_with_auto_install(
            main_path,
            working_dir=solution_dir,
            max_install_attempts=3
        )

        # Set execution result on node
        node.set_execution_result(result)

        if result.success:
            self.log(f"Execution successful | Metric: {result.metric_value} | Time: {result.execution_time:.1f}s")
        else:
            self.log(f"Execution failed: {result.error_message}")
            # Show actual error for debugging
            if result.stderr:
                # Show last 500 chars of stderr for debugging
                stderr_preview = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
                self.log(f"Error details:\n{stderr_preview}")

            # Debug logging - save full error context
            self.debug_logger.log_execution_error(
                node_id=node.id,
                error_type=self._categorize_error(result.stderr or result.error_message),
                error_message=result.error_message,
                stderr=result.stderr or "",
                code_files=node.solution.get_all_files() if node.solution else None
            )

    def _review_execution(self, node: TreeNode):
        """
        Review execution result using ReviewAgent (ExecuteAndReview step).
        Validates metric correctness, checks for data leakage, etc.

        Now stores review findings in node.execution_result.review_findings
        for two-stage lesson extraction (Mejora 2).
        """
        if not node.execution_result or not node.solution:
            return

        self.log("Reviewing execution result...")

        try:
            review = self.review_agent.review_execution(
                problem_description=self.problem_description,
                code=node.solution.main_script,
                execution_output=node.execution_result.stdout,
                execution_error=node.execution_result.stderr,
            )

            if review:
                # Store review findings for two-stage lesson extraction (Mejora 2)
                review_summary = review.get("summary", "")
                node.execution_result.review_findings = review_summary

                # If the review says metric is invalid, downgrade node
                if not review.get("valid_metric", True):
                    self.log(f"ReviewAgent flagged invalid metric: {review_summary}")
                    node.execution_result.validation_metric_valid = False
                    node.status = NodeStatus.BUGGY
                    node.metric_value = None

                # If the review provides a corrected metric value
                reviewed_metric = review.get("metric")
                if reviewed_metric is not None and review.get("valid_metric", False):
                    node.metric_value = float(reviewed_metric)
                    node.execution_result.metric_value = float(reviewed_metric)

                self.log(f"Review summary: {review_summary[:120]}")
        except Exception as e:
            self.log(f"Review failed (non-blocking): {e}")

    # ==================================================================
    # Validation Verification (Mejora 4)
    # ==================================================================

    def _verify_validation(self, node: TreeNode):
        """
        Verify validation split and check for data leakage.

        Only runs for the first _MAX_VALIDATION_CHECKS valid solutions
        to avoid excessive LLM costs.
        """
        if self._validation_verified_count >= self._MAX_VALIDATION_CHECKS:
            return

        if not node.execution_result or not node.solution:
            return

        self.log("Verifying validation methodology...")
        self._validation_verified_count += 1

        try:
            # Get all code for static analysis
            all_code = "\n\n".join(node.solution.get_all_files().values())

            # Run full verification (static + LLM)
            result = self.validation_agent.full_verification(
                code=all_code,
                execution_output=node.execution_result.stdout,
                execution_error=node.execution_result.stderr or "",
            )

            if not result.get("success", True):
                issues = result.get("issues", [])
                self.log(f"Validation issues detected: {issues}")

                # If critical issues, downgrade the node
                if not result.get("leakage_check", True):
                    self.log("Data leakage detected, marking node as BUGGY")
                    node.status = NodeStatus.BUGGY
                    node.execution_result.validation_metric_valid = False
            else:
                self.log("Validation verification passed")

        except Exception as e:
            self.log(f"Validation verification failed (non-blocking): {e}")

    # ==================================================================
    # Lesson Extraction
    # ==================================================================

    def _extract_lessons(self, node: TreeNode):
        """
        Extract lessons from successful execution.

        Now uses two-stage extraction with review_findings (Mejora 2).
        """
        self.log("Extracting lessons...")

        # Always extract a solution lesson (comparative or empirical)
        best = self.mcts.best_node
        compare_node = best if (best and best != node) else None

        # Get review findings for two-stage extraction (Mejora 2)
        review_findings = ""
        if node.execution_result:
            review_findings = node.execution_result.review_findings or ""

        try:
            self.lesson_extractor.extract_solution_lesson(
                node,
                compare_node,
                review_findings=review_findings,
            )
        except Exception as e:
            self.log(f"Solution lesson extraction failed: {e}")

    # ==================================================================
    # Persistence
    # ==================================================================

    def _save_progress(self):
        """Save current progress including lessons, MCTS log, and agent caches."""
        lesson_path = self.working_dir / "lessons.json"
        self.lesson_pool.save_to_file(lesson_path)

        mcts_log_path = self.working_dir / "mcts_log.json"
        self.mcts.save_search_log(mcts_log_path)

        # Save IdeaAgent cache (discovered models, ideas, curriculum state)
        self.idea_agent.save_cache()

    # ==================================================================
    # Statistics
    # ==================================================================

    def _print_final_statistics(self):
        """Print final statistics."""
        elapsed = time.time() - self.start_time

        print(f"\nTime elapsed: {elapsed:.1f}s ({elapsed / 3600:.2f}h)")
        print(f"Iterations: {self.iterations}")

        mcts_stats = self.mcts.get_statistics()
        print(f"\nMCTS Statistics:")
        print(f"  Nodes explored: {mcts_stats['nodes_explored']}")
        print(f"  Valid nodes: {mcts_stats['valid_nodes']}")
        print(f"  Buggy nodes: {mcts_stats['buggy_nodes']}")

        lesson_stats = self.lesson_pool.get_statistics()
        print(f"\nLesson Statistics:")
        print(f"  Solution lessons: {lesson_stats['solution_lessons']}")
        print(f"  Debug lessons: {lesson_stats['debug_lessons']}")

        if self.mcts.best_node:
            print(f"\nBest Solution:")
            print(f"  Node ID: {self.mcts.best_node.id}")
            print(f"  Metric: {self.mcts.best_node.metric_value}")
            print(f"  Execution time: {self.mcts.best_node.execution_time:.1f}s")

    def log(self, message: str):
        """Log message with orchestrator prefix."""
        print(f"[Orchestrator] {message}")

    def _categorize_error(self, error_text: str) -> str:
        """
        Categorize an error for logging and analysis.

        Returns a category string like:
        - "ModuleNotFoundError"
        - "SyntaxError"
        - "RuntimeError"
        - "DataError"
        - "Unknown"
        """
        if not error_text:
            return "Unknown"

        error_lower = error_text.lower()

        # Common error categories
        if "modulenotfounderror" in error_lower or "no module named" in error_lower:
            return "ModuleNotFoundError"
        elif "syntaxerror" in error_lower:
            return "SyntaxError"
        elif "nameerror" in error_lower:
            return "NameError"
        elif "typeerror" in error_lower:
            return "TypeError"
        elif "valueerror" in error_lower:
            return "ValueError"
        elif "keyerror" in error_lower:
            return "KeyError"
        elif "indexerror" in error_lower:
            return "IndexError"
        elif "attributeerror" in error_lower:
            return "AttributeError"
        elif "filenotfounderror" in error_lower or "no such file" in error_lower:
            return "FileNotFoundError"
        elif "memoryerror" in error_lower or "out of memory" in error_lower:
            return "MemoryError"
        elif "cuda" in error_lower or "gpu" in error_lower:
            return "CUDAError"
        elif "timeout" in error_lower:
            return "TimeoutError"
        elif "unicodeencodeerror" in error_lower or "charmap" in error_lower:
            return "EncodingError"
        elif "importerror" in error_lower:
            return "ImportError"
        elif "connectionerror" in error_lower or "network" in error_lower:
            return "NetworkError"
        else:
            return "RuntimeError"
