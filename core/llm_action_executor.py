# core/llm_action_executor.py
"""
LLM Action Executor - Executes validated action requests from the LLM.

This module processes action requests parsed by LLMActionParser and executes
them as SUGGESTIONS (the system decides whether to accept or reject).

All actions are logged for transparency and debugging.
"""

from typing import TYPE_CHECKING, Dict, Any, Optional, List
from datetime import datetime

from core.llm_action_parser import (
    LLMAction,
    ActionResult,
    ActionType,
    ActionStatus,
    LLMActionLogger,
)

if TYPE_CHECKING:
    from orchestrator import MARSOrchestrator


class LLMActionExecutor:
    """
    Executor for LLM action requests.

    Operates in SUGGESTION MODE: all actions are suggestions that the system
    evaluates before deciding whether to accept.

    Rate limiting:
    - SEARCH_MODELS: Max 1 per 5 iterations
    - SWITCH_MODEL: Max 1 per 3 iterations
    - SIGNAL_CONVERGENCE: Requires confidence > 0.8
    """

    # Cooldown periods (iterations between allowed executions)
    COOLDOWNS = {
        ActionType.SEARCH_MODELS: 5,
        ActionType.SWITCH_MODEL: 3,
        ActionType.CHANGE_PHASE: 5,
    }

    # Minimum confidence for convergence signal
    MIN_CONVERGENCE_CONFIDENCE = 0.8

    def __init__(self, working_dir: str = "."):
        """
        Initialize executor.

        Note: Orchestrator reference is set later via set_orchestrator()
        to avoid circular imports.
        """
        self.orchestrator: Optional["MARSOrchestrator"] = None
        self.logger = LLMActionLogger(f"{working_dir}/llm_actions.json")

        # Track last execution iteration for cooldowns
        self.last_execution: Dict[ActionType, int] = {}

        # Track search requests per session
        self.search_count = 0
        self.max_searches_per_session = 5

        # Convergence signal tracking
        self.convergence_signals: List[Dict] = []

    def set_orchestrator(self, orchestrator: "MARSOrchestrator"):
        """Set orchestrator reference for action execution."""
        self.orchestrator = orchestrator

    def execute(
        self,
        action: LLMAction,
        iteration: int,
        node_id: Optional[str] = None
    ) -> ActionResult:
        """
        Execute a validated action request.

        Args:
            action: Validated LLMAction to execute
            iteration: Current MCTS iteration
            node_id: ID of current node being processed

        Returns:
            ActionResult with success status and data
        """
        # Log the request
        self.logger.log_action(action, iteration, node_id)

        # Check cooldown
        if not self._check_cooldown(action.action_type, iteration):
            action.status = ActionStatus.REJECTED
            action.rejection_reason = f"Cooldown active (wait {self._remaining_cooldown(action.action_type, iteration)} iterations)"
            result = ActionResult(
                action=action,
                success=False,
                message=action.rejection_reason,
            )
            self.logger.log_result(result, iteration, node_id)
            return result

        # Execute based on action type
        executor_map = {
            ActionType.SEARCH_MODELS: self._execute_search_models,
            ActionType.SWITCH_MODEL: self._execute_switch_model,
            ActionType.CHANGE_PHASE: self._execute_change_phase,
            ActionType.FILTER_LESSONS: self._execute_filter_lessons,
            ActionType.SIGNAL_CONVERGENCE: self._execute_signal_convergence,
            ActionType.REQUEST_ANALYSIS: self._execute_request_analysis,
            ActionType.ADJUST_STRATEGY: self._execute_adjust_strategy,
        }

        executor = executor_map.get(action.action_type)
        if not executor:
            action.status = ActionStatus.REJECTED
            action.rejection_reason = "No executor for action type"
            result = ActionResult(action=action, success=False, message=action.rejection_reason)
            self.logger.log_result(result, iteration, node_id)
            return result

        try:
            result = executor(action, iteration)
            if result.success:
                self.last_execution[action.action_type] = iteration
            self.logger.log_result(result, iteration, node_id)
            return result
        except Exception as e:
            action.status = ActionStatus.FAILED
            result = ActionResult(
                action=action,
                success=False,
                message=f"Execution failed: {str(e)}",
            )
            self.logger.log_result(result, iteration, node_id)
            return result

    def _check_cooldown(self, action_type: ActionType, iteration: int) -> bool:
        """Check if action is allowed (not in cooldown)."""
        cooldown = self.COOLDOWNS.get(action_type, 0)
        if cooldown == 0:
            return True

        last = self.last_execution.get(action_type, -cooldown)
        return (iteration - last) >= cooldown

    def _remaining_cooldown(self, action_type: ActionType, iteration: int) -> int:
        """Get remaining cooldown iterations."""
        cooldown = self.COOLDOWNS.get(action_type, 0)
        last = self.last_execution.get(action_type, -cooldown)
        remaining = cooldown - (iteration - last)
        return max(0, remaining)

    # =========================================================================
    # Action Executors
    # =========================================================================

    def _execute_search_models(self, action: LLMAction, iteration: int) -> ActionResult:
        """
        Execute SEARCH_MODELS action.

        Triggers a new SOTA model search with the specified keywords.
        Limited to max_searches_per_session total.
        """
        if self.search_count >= self.max_searches_per_session:
            action.status = ActionStatus.REJECTED
            action.rejection_reason = f"Max searches ({self.max_searches_per_session}) reached for this session"
            return ActionResult(action=action, success=False, message=action.rejection_reason)

        if not self.orchestrator:
            action.status = ActionStatus.REJECTED
            action.rejection_reason = "Orchestrator not available"
            return ActionResult(action=action, success=False, message=action.rejection_reason)

        keywords = action.params.get("keywords", [])
        if not keywords:
            action.status = ActionStatus.REJECTED
            action.rejection_reason = "No keywords provided"
            return ActionResult(action=action, success=False, message=action.rejection_reason)

        # Execute search via SearchAgent
        try:
            # Build search query from keywords
            query = " ".join(keywords)
            new_models = self.orchestrator.idea_agent.search_agent.search_sota_models(
                problem_description=f"Find models for: {query}",
                num_candidates=3,
                alternative_search=True,
            )

            if new_models:
                # Register new models with MCTS
                existing_names = {
                    m.get("name", "").lower()
                    for m in self.orchestrator.idea_agent.discovered_models
                }
                added = 0
                for model in new_models:
                    if model.get("name", "").lower() not in existing_names:
                        self.orchestrator.idea_agent.discovered_models.append(model)
                        added += 1

                if added > 0:
                    # Update MCTS with new models
                    self.orchestrator.mcts.set_discovered_models(
                        self.orchestrator.idea_agent.discovered_models
                    )

                self.search_count += 1
                action.status = ActionStatus.EXECUTED
                action.execution_result = f"Added {added} new models"
                return ActionResult(
                    action=action,
                    success=True,
                    message=f"Search completed: found {len(new_models)} models, added {added} new",
                    data=new_models,
                )
            else:
                action.status = ActionStatus.EXECUTED
                action.execution_result = "No new models found"
                return ActionResult(
                    action=action,
                    success=True,
                    message="Search completed but no new models found",
                )

        except Exception as e:
            action.status = ActionStatus.FAILED
            return ActionResult(action=action, success=False, message=f"Search failed: {e}")

    def _execute_switch_model(self, action: LLMAction, iteration: int) -> ActionResult:
        """
        Execute SWITCH_MODEL action.

        Suggests switching to a different model. The system evaluates if the
        model exists and is different from the current one.
        """
        if not self.orchestrator:
            action.status = ActionStatus.REJECTED
            return ActionResult(action=action, success=False, message="Orchestrator not available")

        model_name = action.params.get("model_name", "")
        reason = action.params.get("reason", "LLM suggestion")

        if not model_name:
            action.status = ActionStatus.REJECTED
            return ActionResult(action=action, success=False, message="No model name provided")

        # Find model in discovered models
        discovered = self.orchestrator.idea_agent.discovered_models
        target_idx = None
        for idx, model in enumerate(discovered):
            if model.get("name", "").lower() == model_name.lower():
                target_idx = idx
                break

        if target_idx is None:
            # Model not found - suggest adding it
            action.status = ActionStatus.REJECTED
            action.rejection_reason = f"Model '{model_name}' not in discovered models"
            return ActionResult(
                action=action,
                success=False,
                message=f"Model '{model_name}' not found. Consider using SEARCH_MODELS first.",
            )

        # Check if already on this model
        current_idx = self.orchestrator.mcts.current_model_index
        if target_idx == current_idx:
            action.status = ActionStatus.REJECTED
            action.rejection_reason = "Already using this model"
            return ActionResult(action=action, success=False, message="Already using this model")

        # Request switch via MCTS
        success = self.orchestrator.mcts.request_model_switch(target_idx, reason)

        if success:
            action.status = ActionStatus.EXECUTED
            action.execution_result = f"Switched to model index {target_idx}"
            return ActionResult(
                action=action,
                success=True,
                message=f"Model switch suggested: {model_name} (reason: {reason})",
                data={"new_model_index": target_idx},
            )
        else:
            action.status = ActionStatus.REJECTED
            return ActionResult(
                action=action,
                success=False,
                message="Model switch rejected by MCTS",
            )

    def _execute_change_phase(self, action: LLMAction, iteration: int) -> ActionResult:
        """
        Execute CHANGE_PHASE action.

        Suggests changing the exploration phase. System validates if the
        transition makes sense given current state.
        """
        if not self.orchestrator:
            action.status = ActionStatus.REJECTED
            return ActionResult(action=action, success=False, message="Orchestrator not available")

        from agents.idea_agent import ExplorationPhase

        phase_name = action.params.get("phase", "")
        reason = action.params.get("reason", "LLM suggestion")

        try:
            target_phase = ExplorationPhase(phase_name.lower())
        except ValueError:
            action.status = ActionStatus.REJECTED
            return ActionResult(
                action=action,
                success=False,
                message=f"Invalid phase: {phase_name}",
            )

        current_phase = self.orchestrator.idea_agent.current_phase

        # Don't allow going backwards (except RADICAL_PIVOT which can happen anytime)
        phase_order = [
            ExplorationPhase.BREADTH_SEARCH,
            ExplorationPhase.DEEP_EXPLOIT,
            ExplorationPhase.RADICAL_PIVOT,
            ExplorationPhase.ENSEMBLE_SYNTHESIS,
        ]

        current_idx = phase_order.index(current_phase)
        target_idx = phase_order.index(target_phase)

        # Allow RADICAL_PIVOT from anywhere, otherwise only forward
        if target_phase != ExplorationPhase.RADICAL_PIVOT and target_idx < current_idx:
            action.status = ActionStatus.REJECTED
            return ActionResult(
                action=action,
                success=False,
                message=f"Cannot go from {current_phase.value} back to {target_phase.value}",
            )

        # Apply phase change
        self.orchestrator.idea_agent.current_phase = target_phase
        self.orchestrator.idea_agent.phase_history.append(target_phase)

        action.status = ActionStatus.EXECUTED
        action.execution_result = f"Changed to {target_phase.value}"
        return ActionResult(
            action=action,
            success=True,
            message=f"Phase changed: {current_phase.value} -> {target_phase.value} (reason: {reason})",
            data={"new_phase": target_phase.value},
        )

    def _execute_filter_lessons(self, action: LLMAction, iteration: int) -> ActionResult:
        """
        Execute FILTER_LESSONS action.

        Returns lessons filtered by keywords. Does not modify state.
        """
        if not self.orchestrator:
            action.status = ActionStatus.REJECTED
            return ActionResult(action=action, success=False, message="Orchestrator not available")

        keywords = action.params.get("keywords", [])
        count = action.params.get("count", 10)

        if not keywords:
            action.status = ActionStatus.REJECTED
            return ActionResult(action=action, success=False, message="No keywords provided")

        # Filter lessons from pool
        filtered = self.orchestrator.lesson_pool.filter_by_keywords(keywords, count)

        action.status = ActionStatus.EXECUTED
        action.execution_result = f"Found {len(filtered)} lessons"
        return ActionResult(
            action=action,
            success=True,
            message=f"Filtered {len(filtered)} lessons matching: {keywords}",
            data=filtered,
        )

    def _execute_signal_convergence(self, action: LLMAction, iteration: int) -> ActionResult:
        """
        Execute SIGNAL_CONVERGENCE action.

        Records the LLM's belief that search has converged. System considers
        this along with other metrics to decide on early stopping.
        """
        if not self.orchestrator:
            action.status = ActionStatus.REJECTED
            return ActionResult(action=action, success=False, message="Orchestrator not available")

        confidence = action.params.get("confidence", 0.0)
        reason = action.params.get("reason", "")

        # Require minimum confidence
        if confidence < self.MIN_CONVERGENCE_CONFIDENCE:
            action.status = ActionStatus.REJECTED
            return ActionResult(
                action=action,
                success=False,
                message=f"Confidence {confidence} below minimum {self.MIN_CONVERGENCE_CONFIDENCE}",
            )

        # Record signal
        signal = {
            "iteration": iteration,
            "confidence": confidence,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        }
        self.convergence_signals.append(signal)

        # Check for consistent convergence signals
        recent_signals = [s for s in self.convergence_signals if s["iteration"] > iteration - 5]
        if len(recent_signals) >= 2:
            # Multiple high-confidence signals - suggest early stop
            self.orchestrator.mcts.signal_convergence()
            action.status = ActionStatus.EXECUTED
            action.execution_result = "Convergence signal accepted (multiple confirmations)"
            return ActionResult(
                action=action,
                success=True,
                message=f"Convergence signal recorded (confidence: {confidence}). Early stop suggested.",
                data={"total_signals": len(self.convergence_signals)},
            )
        else:
            action.status = ActionStatus.EXECUTED
            action.execution_result = "Convergence signal recorded (awaiting confirmation)"
            return ActionResult(
                action=action,
                success=True,
                message=f"Convergence signal recorded (confidence: {confidence}). Need more confirmation.",
                data={"total_signals": len(self.convergence_signals)},
            )

    def _execute_request_analysis(self, action: LLMAction, iteration: int) -> ActionResult:
        """
        Execute REQUEST_ANALYSIS action.

        Requests additional data analysis. Currently returns guidance;
        actual analysis would require EDA agent integration.
        """
        analysis_type = action.params.get("type", "")
        target_features = action.params.get("target_features", [])

        # For now, return guidance on how to perform the analysis
        # Full implementation would call EDA agent
        guidance = {
            "feature_importance": "Use model.feature_importances_ or SHAP values",
            "correlation": "Use df.corr() with target variable",
            "distribution": "Use df.describe() and histograms",
            "missing_values": "Use df.isnull().sum() / len(df)",
        }

        action.status = ActionStatus.EXECUTED
        action.execution_result = f"Analysis guidance for: {analysis_type}"
        return ActionResult(
            action=action,
            success=True,
            message=f"Analysis guidance: {guidance.get(analysis_type, 'Unknown type')}",
            data={"type": analysis_type, "guidance": guidance.get(analysis_type)},
        )

    def _execute_adjust_strategy(self, action: LLMAction, iteration: int) -> ActionResult:
        """
        Execute ADJUST_STRATEGY action.

        Suggests a debugging strategy. Returns the suggestion for the
        debug loop to consider.
        """
        strategy = action.params.get("strategy", "")
        reason = action.params.get("reason", "")
        target_file = action.params.get("target_file", "main.py")

        # Map strategy names to debug strategy indices
        strategy_map = {
            "regenerate": 1,  # Strategy 1: Regenerate file
            "diff": 2,        # Strategy 2: Diff-based fix
            "redraft": 3,     # Strategy 3: Full redraft
            "skip": 0,        # Skip debugging
        }

        strategy_idx = strategy_map.get(strategy, 1)

        action.status = ActionStatus.EXECUTED
        action.execution_result = f"Suggested strategy: {strategy} (index {strategy_idx})"
        return ActionResult(
            action=action,
            success=True,
            message=f"Debug strategy suggestion: {strategy} for {target_file} (reason: {reason})",
            data={
                "strategy": strategy,
                "strategy_index": strategy_idx,
                "target_file": target_file,
            },
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            "search_count": self.search_count,
            "max_searches": self.max_searches_per_session,
            "convergence_signals": len(self.convergence_signals),
            "last_executions": {k.value: v for k, v in self.last_execution.items()},
            **self.logger.get_statistics(),
        }
