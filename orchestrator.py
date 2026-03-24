# orchestrator.py
from pathlib import Path
from typing import Optional, Dict, List
import ast
import time
import json
import re
import os

from core.config import Config
from utils.debug_logger import get_debug_logger
from core.mcts import MCTSEngine
from core.tree_node import TreeNode, Solution, ActionType, NodeStatus, ExecutionResult
from core.llm_action_parser import LLMActionParser, LLMAction, ActionResult
from core.llm_action_executor import LLMActionExecutor
from memory.lesson_pool import LessonPool
from memory.lesson_extractor import LessonExtractor
from memory.lesson_types import LessonType
from agents.idea_agent import IdeaAgent
from agents.modular_agent import ModularAgent
from agents.coding_agent import CodingAgent
from agents.debug_agent import DebugAgent
from agents.review_agent import ReviewAgent
from agents.solution_improver import SolutionImprover
from agents.validation_agent import ValidationAgent
from mle.eda_agent import EDAAgent
from utils.file_manager import FileManager
from utils.tree_visualizer import TreeVisualizer
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
        time_budget: int = 18000,
        working_dir: Optional[Path] = None,
        lower_is_better: bool = False,
        metric_name: str = "metric",
        data_schema: str = "",
    ):
        self.problem_description = problem_description
        self.eda_report = eda_report
        self.metadata_dir = Path(metadata_dir)
        self.data_dir = Path(data_dir) if data_dir else None
        self.time_budget = time_budget
        self.working_dir = working_dir or Config.WORKING_DIR
        self.lower_is_better = lower_is_better
        self.metric_name = metric_name
        self.data_schema = data_schema
        # Extract challenge name from working_dir name as identifier for lesson scoping
        self.challenge_name = Path(working_dir).name if working_dir else ""

        # Initialize components
        self.mcts = MCTSEngine(time_budget=time_budget, lower_is_better=lower_is_better)
        self.lesson_pool = LessonPool(challenge_name=self.challenge_name)

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
        self.executor = Executor(ExecutionConfig(timeout=7200))
        self.validator = SolutionValidator()
        self.debug_logger = get_debug_logger()

        # ═══════════════════════════════════════════════════════════════
        # LLM Autonomous Actions System (NEW)
        # ═══════════════════════════════════════════════════════════════
        self.action_parser = LLMActionParser()
        self.action_executor = LLMActionExecutor(working_dir=str(self.working_dir))
        self.action_executor.set_orchestrator(self)

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

        # EDA augmentation: enrich EDA report with lesson insights every N iterations
        self._eda_refresh_interval: int = int(os.getenv("EDA_REFRESH_INTERVAL", "10"))
        self._last_eda_refresh: int = 0

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
        # Emergency budget threshold: 5 minutes before deadline
        _emergency_threshold = max(300, self.time_budget * 0.05)

        # Main MCTS loop
        for new_node in self.mcts.search():
            self.iterations += 1
            elapsed = time.time() - self.start_time

            # ── Emergency mode: time almost up ────────────────────────
            if self.time_budget - elapsed <= _emergency_threshold:
                self.log(
                    f"[EMERGENCY] Only {self.time_budget - elapsed:.0f}s left "
                    f"— submitting best solution found so far"
                )
                break

            print(f"\n{'=' * 70}")
            print(f"ITERATION {self.iterations} | Elapsed: {elapsed:.1f}s / {self.time_budget}s")
            print(f"{'=' * 70}")

            # ----------------------------------------------------------
            # Step 1: Generate solution (DRAFT or IMPROVE only)
            # ----------------------------------------------------------
            success = self._generate_solution(new_node)

            # ══════════════════════════════════════════════════════════
            # Step 1.5: Register discovered models with MCTS (first time)
            # ══════════════════════════════════════════════════════════
            if self.iterations == 1 and self.idea_agent.discovered_models:
                if not self.mcts.discovered_models:
                    self.mcts.set_discovered_models(self.idea_agent.discovered_models)

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

                fixed = self._debug_solution_inline(new_node, attempt=debug_attempt, max_attempts=max_debug)
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
            # Step 6: Extract lessons (Algorithm 2, line 32: ExtractLesson
            #   before Backpropagate — paper order: ExecuteAndReview →
            #   ExtractLesson → Backpropagate)
            # ----------------------------------------------------------
            if new_node.status == NodeStatus.VALID:
                self._extract_lessons(new_node)

            # ----------------------------------------------------------
            # Step 7: Update MCTS tree (backpropagation)
            #   Algorithm 2, line 33: Backpropagate(v_new, R)
            # ----------------------------------------------------------
            self.mcts.update_after_execution(new_node, new_node.execution_result)

            # ----------------------------------------------------------
            # Step 8: Track stagnation and valid solutions (Mejora 3)
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
            # Step 9: Model rotation check (NEW)
            #   If current model has stagnated for MODEL_STAGNATION_THRESHOLD
            #   valid nodes, switch to the next unexplored model.
            # ----------------------------------------------------------
            if self.mcts.should_switch_model():
                switched = self.mcts.switch_to_next_model()
                if switched:
                    self.log(f"Switched to model: {self.mcts.get_current_model_name()}")
                    # Reset global stagnation counter for new model exploration
                    self.stagnation_count = 0
                elif self.mcts.models_exhausted:
                    self.log("All models explored. Continuing with best-performing model.")

            # ----------------------------------------------------------
            # Step 10: Exploration Phase Update (NEW)
            #   Track progress and trigger phase transitions for radical
            #   strategy shifts after many iterations.
            # ----------------------------------------------------------
            current_best = self.mcts.best_metric if self.mcts.best_node else None
            old_phase = self.idea_agent.current_phase
            new_phase = self.idea_agent.update_exploration_state(
                iteration=self.iterations,
                current_best_metric=current_best,
                model_exploration_stats=self.mcts.model_exploration_stats,
                lower_is_better=self.lower_is_better,
            )

            if new_phase != old_phase:
                self.log(f"EXPLORATION PHASE CHANGED: {old_phase.value} → {new_phase.value}")

                # Handle RADICAL_PIVOT: trigger new SOTA search
                from agents.idea_agent import ExplorationPhase
                if new_phase == ExplorationPhase.RADICAL_PIVOT:
                    self.log("Triggering radical SOTA re-search...")
                    new_models = self.idea_agent.trigger_radical_search(self.problem_description)
                    if new_models:
                        # Register new models with MCTS
                        self.mcts.set_discovered_models(self.idea_agent.discovered_models)
                        self.log(f"Registered {len(new_models)} new models from radical search")

            # Save progress
            self._save_progress()

            # Augment EDA report with accumulated lesson insights
            self._augment_eda_if_needed()

        # Search complete
        print("\n" + "=" * 70)
        print("SEARCH COMPLETE")
        print("=" * 70)
        self._print_final_statistics()
        self._generate_tree_visualization()

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
        """
        Draft a new solution from scratch (Algorithm 2, lines 16-22).

        Uses a MONOLITHIC approach: generates a single self-contained main.py
        with all logic inline. This avoids cross-module import mismatches that
        cause most runtime errors with GLM-5. Modular refactoring happens in
        IMPROVE iterations once a working baseline exists.

        Enhanced with model-aware exploration: uses the model assigned by MCTS.
        """
        # Get the model to use from MCTS
        current_model = self.mcts.get_current_model()
        current_model_index = self.mcts.current_model_index

        idea = self._generate_idea(
            force_model=current_model,
            force_model_index=current_model_index,
        )
        if not idea:
            return False

        self.log("Drafting monolithic main.py...")
        main_code = self._generate_monolithic_script(idea)
        if not main_code:
            return False

        node.solution = Solution(
            idea=idea,
            modules={},          # No separate modules in DRAFT
            main_script=main_code,
            module_descriptions={},
        )

        # ═══════════════════════════════════════════════════════════════
        # Assign model info to node for tracking (NEW)
        # ═══════════════════════════════════════════════════════════════
        if current_model:
            node.model_name = current_model.get("name", "Unknown")
            node.model_index = current_model_index
        else:
            # Fallback: get from IdeaAgent
            model_info, model_idx = self.idea_agent.get_last_used_model_info()
            if model_info:
                node.model_name = model_info.get("name", "Unknown")
                node.model_index = model_idx

        self.log(f"Monolithic draft complete (model: {node.model_name})")
        return True

    def _generate_idea(
        self,
        force_model: Optional[Dict] = None,
        force_model_index: int = -1,
    ) -> Optional[str]:
        """
        Generate or improve an idea using the IdeaAgent.

        Now includes LLM action processing: parses the response for action
        requests and executes them as suggestions.

        Args:
            force_model: If provided, force this ML model for the idea
            force_model_index: Index of the forced model
        """
        # Check if MCTS is forcing a new model (switched due to stagnation)
        is_new_model = force_model is not None and force_model_index >= 0

        # If new model OR first idea, generate initial idea with that model
        if len(self.idea_agent.generated_ideas) == 0 or is_new_model:
            self.log(f"Generating {'new model' if is_new_model else 'initial'} idea...")
            idea = self.idea_agent.generate_initial_idea(
                self.problem_description,
                self.eda_report,
                model_architectures=self.idea_agent.discovered_models,
                force_model=force_model,
                force_model_index=force_model_index,
            )
        else:
            best_solution_code = None
            if self.mcts.best_node and self.mcts.best_node.solution:
                files = self.mcts.best_node.solution.get_all_files()
                best_solution_code = "\n\n".join(
                    f"=== {fname} ===\n{code}" for fname, code in files.items()
                )

            idea = self.idea_agent.improve_idea(
                self.problem_description,
                self.eda_report,
                self.lesson_pool,
                previous_ideas=self.idea_agent.generated_ideas,
                valid_solutions_count=self.valid_solutions_count,
                stagnation_count=self.stagnation_count,
                best_solution_code=best_solution_code,
            )

        # ═══════════════════════════════════════════════════════════════
        # Process LLM Action Requests (NEW)
        # ═══════════════════════════════════════════════════════════════
        if idea:
            idea, action_results = self._process_llm_actions(idea)
            if action_results:
                self.log(f"Processed {len(action_results)} LLM action(s)")

        return idea

    def _process_llm_actions(
        self,
        response: str,
        node_id: Optional[str] = None
    ) -> tuple:
        """
        Parse LLM response for action requests and execute them.

        All actions are treated as SUGGESTIONS that the system evaluates
        before deciding whether to accept.

        Args:
            response: Full LLM response text
            node_id: Optional node ID for logging

        Returns:
            Tuple of (clean_response, list of ActionResults)
        """
        # Parse actions from response
        actions = self.action_parser.parse(response)

        if not actions:
            return response, []

        results = []
        for action in actions:
            # Validate action parameters
            if not self.action_parser.validate(action):
                self.log(f"  [Action] Rejected (invalid): {action.action_type.value} - {action.rejection_reason}")
                continue

            # Execute action
            result = self.action_executor.execute(
                action,
                iteration=self.iterations,
                node_id=node_id,
            )
            results.append(result)

            # Log result
            status = "OK" if result.success else "REJECTED"
            self.log(f"  [Action] {action.action_type.value}: {status} - {result.message[:80]}")

        # Extract clean content (without action markers)
        clean_response = self.action_parser.extract_content(response)

        return clean_response, results

    def _generate_monolithic_script(self, idea: str) -> Optional[str]:
        """
        Generate a single self-contained main.py with all pipeline logic inline.
        No imports from sibling files — eliminates cross-module API mismatches.

        Handles token-limit truncation via continuation: if the model stops
        mid-script, sends up to MAX_CONTINUATIONS follow-up requests that
        pick up from the last generated line.
        """
        from utils.hardware_info import get_hardware_context
        hardware_context = get_hardware_context()

        lessons_text = self.lesson_pool.format_for_prompt(
            lesson_type=None, k=10
        ) if self.lesson_pool else ""

        try:
            prompt = self.coding_agent.prompt_manager.get_prompt(
                "monolithic_draft",
                problem_description=self.problem_description,
                idea=idea,
                metric_name=self.metric_name or "unknown",
                data_schema=self.data_schema or "Not available",
                eda_report=self.eda_report or "Not available",
                lessons=lessons_text or "No lessons yet.",
                hardware_context=hardware_context,
            )
        except Exception as e:
            self.log(f"Prompt load failed ({e}), using fallback")
            prompt = self._monolithic_fallback_prompt(idea, lessons_text, hardware_context)

        from agents.coding_agent import _STREAM
        response = self.coding_agent.call_llm(
            user_message=prompt,
            temperature=0.2,
            max_tokens=Config.MAX_TOKENS,
            stream=_STREAM,
        )

        raw_content = response["content"]
        code, truncated = self._extract_and_detect_truncation(raw_content)

        if not code:
            self.log("Failed to extract code from monolithic draft response")
            return None

        # ── Continuation loop for truncated responses ──────────────────
        MAX_CONTINUATIONS = 3
        continuation = 0
        while truncated and continuation < MAX_CONTINUATIONS:
            continuation += 1
            self.log(
                f"Response truncated (attempt {continuation}/{MAX_CONTINUATIONS}) "
                f"— requesting continuation ({len(code)} chars so far)"
            )
            extra_code, still_truncated = self._continue_truncated_code(code)
            if not extra_code:
                self.log("Continuation returned no code — stopping")
                break
            code = code.rstrip() + "\n" + extra_code.lstrip()
            truncated = still_truncated

        if truncated:
            self.log("Script still truncated after continuations — attempting repair")
            code = self._repair_truncated_script(code)

        if not code:
            return None

        is_valid, error = self.coding_agent.parser.validate_syntax(code)
        if not is_valid:
            self.log(f"Monolithic draft has syntax error: {error} — attempting fix")
            fixed = self.coding_agent._ask_llm_to_fix_syntax(code, error, "main.py")
            if fixed:
                is_valid, _ = self.coding_agent.parser.validate_syntax(fixed)
                if is_valid:
                    self.log(f"Syntax fixed, final size: {len(fixed)} chars")
                    return fixed
            return None

        self.log(f"Monolithic main.py generated ({len(code)} chars, {code.count(chr(10))} lines)")
        return code

    def _extract_and_detect_truncation(self, raw: str) -> tuple:
        """
        Extract Python code from LLM response and detect if it was truncated.

        Returns (code, is_truncated).
        Truncation signals:
        - No closing ``` after the opening ```python
        - Code ends mid-statement (open parenthesis, trailing comma, etc.)
        - AST parse fails with EOF error
        """
        if not raw:
            return None, False

        # Try clean closed block first
        import re as _re
        closed = _re.search(r'```python\s*\n(.*?)```', raw, _re.DOTALL | _re.IGNORECASE)
        if closed:
            return closed.group(1).strip(), False  # Properly closed — not truncated

        # Check for unclosed block (truncation)
        unclosed = _re.search(r'```python\s*\n(.+)$', raw, _re.DOTALL | _re.IGNORECASE)
        if unclosed:
            code = unclosed.group(1).strip()
            if len(code) > 100:
                return code, True  # Truncated

        # Fallback: use existing extractor, mark as possibly truncated
        code = self.coding_agent._extract_code_from_response(raw)
        if code:
            # Truncated if raw response doesn't end with closing ```
            is_truncated = not raw.rstrip().endswith("```")
            return code, is_truncated

        return None, False

    def _continue_truncated_code(self, existing_code: str) -> tuple:
        """
        Ask the model to continue a truncated script.
        Passes the FULL existing code so the model has complete context.
        Returns (continuation_code, still_truncated).
        """
        from agents.coding_agent import _STREAM
        continuation_prompt = f"""The Python script below was cut off mid-generation due to output length limits.
Continue writing from EXACTLY where it stopped. Do NOT repeat any code already shown.
Write only what comes next, ending with ``` when the script is fully complete.

FULL SCRIPT SO FAR:
```python
{existing_code}
```

Continue the script from the next line:
```python
"""
        response = self.coding_agent.call_llm(
            user_message=continuation_prompt,
            temperature=0.1,
            max_tokens=Config.MAX_TOKENS,
            stream=_STREAM,
        )

        raw = response["content"]
        code, still_truncated = self._extract_and_detect_truncation(raw)

        # If continuation itself is a full closed block, take it
        # If it's also unclosed, we got more code but may need another round
        if not code:
            # Last resort: take everything after the prompt marker
            import re as _re
            m = _re.search(r'```python\s*\n(.+)', raw, _re.DOTALL | _re.IGNORECASE)
            if m:
                code = m.group(1).strip()
                still_truncated = not raw.rstrip().endswith("```")

        return code, still_truncated

    def _repair_truncated_script(self, code: str) -> Optional[str]:
        """
        Attempt to repair a script that's still truncated after continuations.
        Tries to close open blocks so it at least parses as valid Python.
        """
        self.log("Attempting truncation repair...")
        lines = code.split("\n")

        # Remove incomplete last line (mid-token)
        while lines:
            last = lines[-1].strip()
            if last and not last.endswith(
                (":", ",", "(", "[", "{", "\\", "=", "+", "-", "*", "/", "and", "or", "not")
            ) and last.count('"') % 2 == 0 and last.count("'") % 2 == 0:
                break
            lines.pop()

        if not lines:
            return None

        repaired = "\n".join(lines)

        # Try progressively adding closure tokens
        for closure in ["", "\n", "\n    pass", "\n        pass", "\n)\n", "\n]\n"]:
            candidate = repaired + closure
            is_valid, _ = self.coding_agent.parser.validate_syntax(candidate)
            if is_valid:
                self.log(f"Repair succeeded ({len(candidate)} chars)")
                return candidate

        self.log("Repair failed — returning None")
        return None

    def _monolithic_fallback_prompt(
        self, idea: str, lessons: str, hardware_context: str
    ) -> str:
        """Fallback prompt when monolithic_draft.txt template fails to load."""
        return f"""Write a single self-contained Python script (main.py) that solves this ML problem.

PROBLEM: {self.problem_description[:800]}

APPROACH: {idea}

METRIC TO OPTIMIZE: {self.metric_name or "unknown"}

DATA SCHEMA: {self.data_schema or "Not available"}

DATA LOCATIONS:
import os
DATA_DIR     = os.environ.get('DATA_DIR', '.')
METADATA_DIR = os.environ.get('METADATA_DIR', './metadata')
- Train (fit): os.path.join(METADATA_DIR, 'train.csv')
- Val (eval):  os.path.join(METADATA_DIR, 'val.csv')
- Test (pred): os.path.join(DATA_DIR, 'test.csv')
- Output:      ./submission/submission.csv

LESSONS FROM PREVIOUS RUNS:
{lessons or "None yet."}

HARDWARE: {hardware_context}

REQUIREMENTS:
1. Self-contained — all code in one file, no local imports
2. Print exactly: Final Validation Metric: <value>
3. Save predictions to ./submission/submission.csv
4. ASCII only, no emojis
5. Max 800 lines

Provide code in a ```python block.
"""

    def _draft_solution_modular(self, node: TreeNode, idea: str) -> bool:
        """
        Modular draft: decomposes idea into separate .py files.
        Called from IMPROVE iterations when a working monolithic baseline exists.

        Validates cross-module imports before assembling to catch API mismatches
        early.
        """
        modules_desc = self.modular_agent.decompose_idea(
            self.problem_description,
            idea,
            eda_report=self.eda_report,
            lesson_pool=self.lesson_pool,
            metric_name=self.metric_name,
            data_schema=self.data_schema,
        )
        if not modules_desc:
            return False

        is_valid, issues = self.modular_agent.validate_decomposition(modules_desc)
        if not is_valid:
            self.log(f"Invalid decomposition: {issues}")
            return False

        module_order = self.modular_agent.get_module_order(modules_desc)
        self.log(f"  Module generation order: {module_order}")

        implemented_modules = {}
        for module_name in module_order:
            if module_name == "main" or module_name not in modules_desc:
                continue
            module_desc = modules_desc[module_name]

            code = self.coding_agent.implement_module(
                self.problem_description,
                idea,
                f"{module_name}.py",
                module_desc,
                implemented_modules,
                self.lesson_pool,
                eda_report=self.eda_report,
                metric_name=self.metric_name,
                data_schema=self.data_schema,
            )

            if code:
                filename = f"{module_name}.py"
                import_issues = self._validate_cross_module_imports(
                    filename, code, implemented_modules
                )
                if import_issues:
                    self.log(f"  Cross-module import issues in {filename}: {import_issues}")
                    code = self._fix_cross_module_imports(
                        filename, code, import_issues, implemented_modules,
                        idea, module_desc
                    ) or code
                implemented_modules[filename] = code
            else:
                self.log(f"Failed to implement {module_name}")

        implemented_modules = self._test_and_debug_modules(
            idea, implemented_modules, modules_desc
        )

        main_code = self.coding_agent.implement_main_script(
            self.problem_description,
            idea,
            implemented_modules,
            self.lesson_pool,
            eda_report=self.eda_report,
            metric_name=self.metric_name,
            data_schema=self.data_schema,
        )
        if not main_code:
            return False

        node.solution = Solution(
            idea=idea,
            modules=implemented_modules,
            main_script=main_code,
            module_descriptions=modules_desc,
        )

        self.log(f"Modular solution drafted: {len(implemented_modules)} modules + main")
        return True

    def _improve_solution(self, node: TreeNode) -> bool:
        """Improve existing solution (Section 4.4.1)."""

        parent = node.parent
        if not parent or not parent.solution:
            self.log("No parent solution to improve")
            return False

        self.log(f"Improving solution from parent node {parent.id}")
        self.log(f"  Parent metric: {parent.metric_value}")

        # ═══════════════════════════════════════════════════════════════
        # Inherit model info from parent (NEW)
        # ═══════════════════════════════════════════════════════════════
        if parent.model_name:
            node.model_name = parent.model_name
            node.model_index = parent.model_index
        else:
            # Fallback for legacy nodes without model info
            current_model = self.mcts.get_current_model()
            if current_model:
                node.model_name = current_model.get("name", "Unknown")
                node.model_index = self.mcts.current_model_index

        # Collect modification history from entire parent lineage
        lineage_mods: list = []
        ancestor = parent
        while ancestor is not None:
            if ancestor.applied_modifications:
                lineage_mods = ancestor.applied_modifications + lineage_mods
            ancestor = ancestor.parent

        success, improved_files, reasoning = self.solution_improver.propose_improvements(
            problem_description=self.problem_description,
            current_solution=parent.solution.get_all_files(),
            current_metric=parent.metric_value,
            lesson_pool=self.lesson_pool,
            metric_name=self.metric_name,
            lower_is_better=self.lower_is_better,
            data_schema=self.data_schema,
            applied_modifications=lineage_mods,
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

        # Record this improvement in the node's modification history
        if reasoning:
            node.applied_modifications = lineage_mods + [reasoning[:200]]

        self.log(f"Solution improved: modified {len(improved_files)} files")
        return True

    # ==================================================================
    # Internal Debugging (inline, not a separate MCTS action)
    # ==================================================================

    def _debug_solution_inline(
        self, node: TreeNode, attempt: int = 1, max_attempts: int = 10
    ) -> bool:
        """
        Debug a buggy node **in-place** (Algorithm 2, lines 27-30).

        Uses PROGRESSIVE strategies based on attempt number:
        - Attempts 1-3  : Targeted diff-based fix (minimal change)
        - Attempts 4-6  : Direct file regeneration (structural rewrite)
        - Attempts 7+   : Full solution regeneration from scratch

        Modifies node.solution directly so the next execution uses the fix.
        Returns True if a fix was generated, False otherwise.
        """
        if not node.solution or not node.execution_result:
            return False

        error_output = node.execution_result.stderr or node.execution_result.error_message
        all_files = node.solution.get_all_files()

        # Build context from previous debug attempts so the model avoids
        # repeating the same failing approaches (multi-turn context)
        prev_context = ""
        if node.debug_history:
            prev_context = "\n".join(
                f"Attempt {i + 1}: {entry}"
                for i, entry in enumerate(node.debug_history)
            )

        # ══════════════════════════════════════════════════════════════
        # Strategy 0 (always first): Quick environment fixes — no LLM
        # ══════════════════════════════════════════════════════════════
        if self._apply_environment_fixes(node, error_output):
            self.log("Strategy 0: Applied environment fixes (no LLM)")
            return True

        error_analysis = None

        # ══════════════════════════════════════════════════════════════
        # Strategy 1 (attempts 1-4): Direct file regeneration (PRIMARY)
        # More reliable than diffs for models that struggle with exact
        # old_code matching. Identify the error file and regenerate it.
        # ══════════════════════════════════════════════════════════════
        if attempt <= 4:
            target_file = self._extract_error_file(error_output)
            if not target_file:
                target_file = "main.py"
            if target_file and target_file in all_files:
                self.log(f"Strategy 1 (attempt {attempt}): Regenerating {target_file}")
                error_analysis = self.debug_agent.analyze_error(
                    self.problem_description, all_files, error_output,
                    self.lesson_pool, previous_attempts=prev_context,
                    eda_report=self.eda_report, data_schema=self.data_schema,
                    metric_name=self.metric_name,
                )
                regenerated = self.debug_agent.regenerate_file(
                    target_file=target_file,
                    original_code=all_files[target_file],
                    execution_error=error_output,
                    error_analysis=error_analysis or "No analysis available",
                )
                if regenerated:
                    self._apply_fixes(node, {target_file: regenerated})
                    node.debug_history.append(
                        f"Attempt {attempt}: regenerated {target_file} OK | Error: {error_output[:100].strip()}"
                    )
                    return True
                node.debug_history.append(
                    f"Attempt {attempt}: regenerate {target_file} FAILED | Error: {error_output[:100].strip()}"
                )

        # ══════════════════════════════════════════════════════════════
        # Strategy 2 (attempts 5-7): Targeted diff-based fix (FALLBACK)
        # Used when file regeneration fails — attempts minimal targeted
        # edits using XML diff format.
        # ══════════════════════════════════════════════════════════════
        if 5 <= attempt <= 7:
            self.log(f"Strategy 2 (attempt {attempt}): Diff-based fix")
            if error_analysis is None:
                error_analysis = self.debug_agent.analyze_error(
                    self.problem_description, all_files, error_output,
                    self.lesson_pool, previous_attempts=prev_context,
                    eda_report=self.eda_report, data_schema=self.data_schema,
                    metric_name=self.metric_name,
                )
            if error_analysis:
                fixed_files = self.debug_agent.fix_error(
                    self.problem_description,
                    all_files,
                    error_output,
                    error_analysis,
                    self.lesson_pool,
                    previous_attempts=prev_context,
                    eda_report=self.eda_report,
                    data_schema=self.data_schema,
                    metric_name=self.metric_name,
                )
                if fixed_files:
                    self._apply_fixes(node, fixed_files)
                    node.debug_history.append(
                        f"Attempt {attempt}: diff-fix OK | Error: {error_output[:100].strip()}"
                    )
                    return True
                node.debug_history.append(
                    f"Attempt {attempt}: diff-fix FAILED | Error: {error_output[:100].strip()}"
                )

        # ══════════════════════════════════════════════════════════════
        # Strategy 3 (attempts 8+): Full solution redraft from scratch
        # ══════════════════════════════════════════════════════════════
        if attempt >= 8 and node.solution:
            self.log(f"Strategy 3 (attempt {attempt}): Full solution redraft")
            if self._draft_solution(node):
                node.debug_history.append(
                    f"Attempt {attempt}: full redraft OK"
                )
                return True
            node.debug_history.append(f"Attempt {attempt}: full redraft FAILED")

        self.log(f"All debug strategies failed at attempt {attempt}")
        return False

    def _apply_fixes(self, node: TreeNode, fixed_files: Dict[str, str]):
        """Apply fixes to a node's solution in-place.

        If any module (non-main) was regenerated AND main.py was NOT explicitly
        fixed by the LLM, refreshes main.py to sync with updated function
        signatures — avoiding stale API calls.

        If the LLM fixed main.py directly, we trust that fix and do NOT
        overwrite it with a regenerated version.
        """
        modules_changed = []
        main_explicitly_fixed = False

        # First pass: detect what the LLM actually changed
        original_files = node.solution.get_all_files()
        for filename, code in fixed_files.items():
            original_code = original_files.get(filename, "")
            if filename == "main.py":
                if code != original_code:
                    main_explicitly_fixed = True
                node.solution.main_script = code
            else:
                if code != original_code:
                    modules_changed.append(filename)
                node.solution.modules[filename] = code

        self.log(f"Applied fix to {len(fixed_files)} files")

        # If a module changed but main.py was NOT explicitly fixed, regenerate
        # main.py so it stays in sync with updated module APIs.
        # If main.py WAS explicitly fixed, trust the LLM's version.
        if modules_changed and not main_explicitly_fixed:
            self.log(
                f"  Modules changed: {modules_changed} — regenerating main.py "
                f"to sync with updated APIs"
            )
            self._refresh_main_script(node)
        elif modules_changed and main_explicitly_fixed:
            self.log(
                f"  Modules changed: {modules_changed} but main.py was explicitly "
                f"fixed by LLM — skipping regeneration to preserve fix"
            )

    def _validate_metric_bounds(self, node: TreeNode):
        """
        Reject metric values that are physically impossible for the metric type.

        Uses the metric name to infer expected bounds and flags outliers.
        """
        metric = node.metric_value
        name = (self.metric_name or "").lower()

        # Metrics that must be in [0, 1]
        bounded_01 = ("auc", "roc", "accuracy", "acc", "f1", "precision",
                      "recall", "r2", "iou", "jaccard", "dice", "kappa",
                      "mcc", "map", "ndcg")
        # Metrics that must be >= 0
        non_negative = ("mae", "mse", "rmse", "loss", "error", "logloss",
                        "cross_entropy", "bce", "ce")

        if any(k in name for k in bounded_01):
            if not (0.0 <= metric <= 1.0):
                self.log(
                    f"  [Bounds] Metric {self.metric_name}={metric:.4f} out of [0,1] — "
                    f"likely a bug (data leakage or wrong scale). Invalidating."
                )
                node.execution_result.validation_metric_valid = False
                node.status = NodeStatus.BUGGY
                node.metric_value = None
                return
            # Suspiciously perfect score warning
            if metric > 0.995:
                self.log(
                    f"  [Bounds] WARNING: {self.metric_name}={metric:.4f} is suspiciously high "
                    f"— possible data leakage."
                )

        elif any(k in name for k in non_negative):
            if metric < 0:
                self.log(
                    f"  [Bounds] Metric {self.metric_name}={metric:.4f} is negative — "
                    f"impossible for a loss metric. Invalidating."
                )
                node.execution_result.validation_metric_valid = False
                node.status = NodeStatus.BUGGY
                node.metric_value = None

    def _classify_test_failure(self, test_result) -> str:
        """
        Classify a module test failure into one of:
        - 'timeout'    : process was killed by time limit
        - 'import'     : missing module / import error
        - 'syntax'     : syntax / indentation error
        - 'logic'      : assertion / runtime error (fixable by debug)
        """
        error_text = (test_result.error_message or "") + (test_result.stderr or "")
        if "timeout" in error_text.lower() or not test_result.stderr:
            if test_result.execution_time >= 110:  # close to 120s limit
                return "timeout"
        if any(k in error_text for k in ("ModuleNotFoundError", "ImportError", "No module named")):
            return "import"
        if any(k in error_text for k in ("SyntaxError", "IndentationError", "TabError")):
            return "syntax"
        return "logic"

    def _refresh_main_script(self, node: TreeNode):
        """
        Regenerate main.py using the current (post-debug) module state.

        Called after a module is changed during inline debugging to keep
        main.py in sync with the updated function signatures.
        Skips silently if the solution or idea is not available.
        """
        if not node.solution:
            return
        try:
            new_main = self.coding_agent.implement_main_script(
                self.problem_description,
                node.solution.idea,
                node.solution.modules,
                self.lesson_pool,
                eda_report=self.eda_report,
                metric_name=self.metric_name,
                data_schema=self.data_schema,
            )
            if new_main:
                node.solution.main_script = new_main
                self.log("  main.py regenerated with updated module signatures")
            else:
                self.log("  main.py regeneration returned None — keeping existing")
        except Exception as e:
            self.log(f"  main.py regeneration failed ({e}) — keeping existing")

    def _extract_error_file(self, error: str) -> Optional[str]:
        """Extract the filename that caused the error from traceback."""
        matches = re.findall(r'File\s+"[^"]*[\\/](\w+\.py)"', error)
        if matches:
            return matches[-1]  # Last match = deepest in call stack
        return None

    def _validate_cross_module_imports(
        self,
        filename: str,
        code: str,
        existing_modules: Dict[str, str],
    ) -> List[str]:
        """
        Check that imports from sibling modules reference names that actually exist.

        For example, if module B does `from feature_eng import FeatureEngineer`
        but feature_eng.py doesn't define FeatureEngineer, this returns an error
        before the buggy code enters the solution — avoiding a runtime NameError.

        Returns list of issue strings (empty = all OK).
        """
        issues = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []  # Syntax errors already handled by CodingAgent

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module_ref = node.module or ""
            sibling_file = f"{module_ref}.py"
            if sibling_file not in existing_modules:
                continue  # External library — not our problem

            sibling_code = existing_modules[sibling_file]
            # Build set of names exported by the sibling
            try:
                sibling_tree = ast.parse(sibling_code)
            except SyntaxError:
                continue

            exported_names = set()
            for snode in ast.walk(sibling_tree):
                if isinstance(snode, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    exported_names.add(snode.name)
                elif isinstance(snode, ast.Assign):
                    for target in snode.targets:
                        if isinstance(target, ast.Name):
                            exported_names.add(target.id)
                elif isinstance(snode, ast.AnnAssign) and isinstance(snode.target, ast.Name):
                    exported_names.add(snode.target.id)

            for alias in node.names:
                name = alias.name
                if name == "*":
                    continue
                if name not in exported_names:
                    issues.append(
                        f"'{name}' not found in {sibling_file} "
                        f"(available: {sorted(exported_names)[:8]})"
                    )
        return issues

    def _fix_cross_module_imports(
        self,
        filename: str,
        code: str,
        issues: List[str],
        existing_modules: Dict[str, str],
        idea: str,
        module_desc: str,
    ) -> Optional[str]:
        """
        Ask the CodingAgent to regenerate a module whose cross-module imports
        don't resolve, providing the actual API of the dependency.
        """
        issues_text = "\n".join(f"  - {i}" for i in issues)
        # Build a compact API summary of all sibling modules
        api_summary = []
        for sib_file, sib_code in existing_modules.items():
            try:
                tree = ast.parse(sib_code)
            except SyntaxError:
                continue
            names = []
            for n in ast.walk(tree):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.append(n.name)
            if names:
                api_summary.append(f"{sib_file}: {', '.join(names)}")

        fix_prompt = f"""The module {filename} has import errors — it tries to import names that
don't exist in the sibling modules. Fix the imports to use the ACTUAL exported names.

IMPORT ISSUES:
{issues_text}

ACTUAL API OF SIBLING MODULES:
{chr(10).join(api_summary)}

CURRENT CODE OF {filename}:
```python
{code}
```

MODULE PURPOSE: {module_desc}

Rules:
1. Fix ONLY the broken import statements and any code that uses the wrong names
2. Do NOT add new functionality
3. Output the COMPLETE fixed file in a ```python block
4. ASCII only, no emojis
"""
        self.log(f"  Attempting cross-module import fix for {filename}")
        response = self.coding_agent.call_llm(
            user_message=fix_prompt,
            temperature=0,
            max_tokens=8192,
        )
        fixed = self.coding_agent._extract_code_from_response(response["content"])
        if fixed:
            is_valid, _ = self.coding_agent.parser.validate_syntax(fixed)
            if is_valid:
                self.log(f"  Cross-module import fix applied to {filename}")
                return fixed
        self.log(f"  Cross-module import fix failed for {filename}")
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
                data_schema=self.data_schema,
                eda_report=self.eda_report,
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

            # Use auto-install for module tests too
            test_result = self.executor.execute_with_auto_install(
                test_dir / f"test_{filename}",
                working_dir=test_dir,
                timeout=120,
                max_install_attempts=2,
                env_vars=self._data_env_vars(),
            )

            if test_result.success:
                self.log(f"  Module {module_name} passed tests")
                tested_modules[filename] = code
            else:
                # ── Classify failure type before debugging ────────────
                failure_type = self._classify_test_failure(test_result)
                self.log(f"  Module {module_name} failed tests [{failure_type}], attempting fix")

                if failure_type == "timeout":
                    self.log(
                        f"  Module {module_name} timed out — keeping as-is "
                        f"(likely heavy computation; will tune at runtime)"
                    )
                    tested_modules[filename] = code
                    continue

                error_text = test_result.stderr or test_result.error_message or ""
                fixed_code = self._debug_module(
                    idea, filename, code, error_text,
                    modules_desc.get(module_name, ""),
                    all_modules=implemented_modules,
                )
                if fixed_code:
                    self.log(f"  Module {module_name} fixed successfully")
                    tested_modules[filename] = fixed_code
                else:
                    self.log(
                        f"  WARNING: Module {module_name} fix failed — "
                        f"using original code (known broken). "
                        f"Error: {error_text[:200].strip()}"
                    )
                    tested_modules[filename] = code

        return tested_modules

    def _debug_module(
        self,
        idea: str,
        filename: str,
        code: str,
        error: str,
        description: str,
        all_modules: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Attempt to fix a single module that failed unit testing.

        Passes ALL sibling modules as context so the debug agent can verify
        that function calls match the actual signatures of dependencies.
        """
        # Include all sibling modules so the agent sees the full API surface
        context_files = dict(all_modules) if all_modules else {}
        context_files[filename] = code  # Ensure the failing module is included

        error_analysis = self.debug_agent.analyze_error(
            self.problem_description,
            context_files,
            error,
            self.lesson_pool,
            eda_report=self.eda_report,
            data_schema=self.data_schema,
            metric_name=self.metric_name,
        )
        if not error_analysis:
            return None

        fixed_files = self.debug_agent.fix_error(
            self.problem_description,
            context_files,
            error,
            error_analysis,
            self.lesson_pool,
            eda_report=self.eda_report,
            data_schema=self.data_schema,
            metric_name=self.metric_name,
        )

        fixed_code = fixed_files.get(filename) if fixed_files else None

        # Extract a debug lesson regardless of whether the fix succeeded.
        # This teaches the system about module-level error patterns.
        try:
            self.lesson_extractor.extract_module_debug_lesson(
                module_name=filename,
                original_code=code,
                fixed_code=fixed_code,
                error=error,
                error_analysis=error_analysis,
                fixed=bool(fixed_code),
            )
        except Exception as e:
            self.log(f"Module debug lesson extraction failed (non-blocking): {e}")

        return fixed_code

    def _data_env_vars(self) -> Dict[str, str]:
        """Return env vars that point generated code to the original data paths."""
        return {
            "DATA_DIR": str(self.data_dir.absolute()) if self.data_dir else ".",
            "METADATA_DIR": str(self.metadata_dir.absolute()),
        }

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

        # Link metadata (train/val splits) into solution dir
        if self.metadata_dir.exists():
            self.file_manager.copy_metadata(solution_dir, self.metadata_dir)

        # Create working subdirectories
        self.file_manager.create_working_subdirs(solution_dir)

        # Execute main script with auto-install for missing dependencies
        main_path = solution_dir / "main.py"
        result = self.executor.execute_with_auto_install(
            main_path,
            working_dir=solution_dir,
            max_install_attempts=3,
            env_vars=self._data_env_vars(),
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
                metric_name=self.metric_name,
                lower_is_better=self.lower_is_better,
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

                # Heuristic bounds validation
                if node.metric_value is not None:
                    self._validate_metric_bounds(node)

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
    # EDA Augmentation
    # ==================================================================

    def _augment_eda_if_needed(self):
        """
        Enrich the shared EDA report with insights extracted from solution
        lessons every _eda_refresh_interval iterations.

        The raw EDA (statistics) never changes because the CSV doesn't change,
        but accumulated lessons reveal what data characteristics matter most.
        Appending them here means every downstream agent (IdeaAgent, CodingAgent,
        DebugAgent) automatically benefits from this growing knowledge.
        """
        if self.iterations - self._last_eda_refresh < self._eda_refresh_interval:
            return

        solution_lessons = self.lesson_pool.get_recent_lessons(
            LessonType.SOLUTION, k=5
        )
        if not solution_lessons:
            return

        insights = []
        for lesson in solution_lessons:
            if hasattr(lesson, "key_lesson") and lesson.key_lesson:
                insights.append(f"- {lesson.key_lesson.strip()}")

        if not insights:
            return

        # Use a unique sentinel that cannot appear in normal EDA output
        marker = "\n\n## [MARS_AUGMENT] Insights from Accumulated Solutions"
        augmentation = (
            marker + "\n"
            + "\n".join(insights)
        )

        # Replace previous augmentation section (if any) to avoid duplication
        base = self.eda_report.split(marker)[0]
        self.eda_report = base + augmentation

        self._last_eda_refresh = self.iterations
        self.log(
            f"EDA augmented with {len(insights)} lesson insight(s) "
            f"(iteration {self.iterations})"
        )

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

        # Model exploration statistics (NEW)
        print(f"\nModel Exploration:")
        print(f"  Total models discovered: {mcts_stats.get('total_models', 0)}")
        print(f"  Models explored: {mcts_stats.get('models_explored', 0)}")
        if self.mcts.discovered_models:
            print(f"  Models detail:")
            for idx, stats in self.mcts.model_exploration_stats.items():
                model_name = self.mcts.discovered_models[int(idx)].get("name", "Unknown") if int(idx) < len(self.mcts.discovered_models) else "Unknown"
                best = stats.get("best_metric")
                valid = stats.get("valid_count", 0)
                explored = "✓" if stats.get("explored", False) else "○"
                best_str = f"{best:.6f}" if best is not None else "N/A"
                print(f"    [{explored}] {model_name}: {valid} valid nodes, best={best_str}")

        lesson_stats = self.lesson_pool.get_statistics()
        print(f"\nLesson Statistics:")
        print(f"  Solution lessons: {lesson_stats['solution_lessons']}")
        print(f"  Debug lessons: {lesson_stats['debug_lessons']}")

        if self.mcts.best_node:
            print(f"\nBest Solution:")
            print(f"  Node ID: {self.mcts.best_node.id}")
            print(f"  Metric: {self.mcts.best_node.metric_value}")
            print(f"  Model: {self.mcts.best_node.model_name}")
            print(f"  Execution time: {self.mcts.best_node.execution_time:.1f}s")

    def _generate_tree_visualization(self):
        """Render and save the MCTS search tree as a PNG."""
        try:
            viz = TreeVisualizer()
            output_path = self.working_dir / "mcts_tree.png"
            viz.generate(
                root=self.mcts.root,
                best_node=self.mcts.best_node,
                output_path=output_path,
                metric_name=self.metric_name,
                lower_is_better=self.lower_is_better,
            )
        except Exception as e:
            self.log(f"Tree visualization failed (non-blocking): {e}")

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
