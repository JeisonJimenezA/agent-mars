# core/llm_action_parser.py
"""
LLM Action Parser - Detects and extracts action requests from LLM responses.

This module enables autonomous LLM decision-making by parsing special action
markers that the LLM can include in its responses to request system actions.

Supported actions:
- SEARCH_MODELS: Request additional model/technique search
- SWITCH_MODEL: Suggest switching to a different ML model
- CHANGE_PHASE: Suggest changing exploration phase
- FILTER_LESSONS: Request specific lessons from the pool
- SIGNAL_CONVERGENCE: Indicate belief that search has converged
- REQUEST_ANALYSIS: Request additional data analysis
- ADJUST_STRATEGY: Suggest debugging strategy
"""

import re
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class ActionType(Enum):
    """Supported LLM action types."""
    SEARCH_MODELS = "SEARCH_MODELS"
    SWITCH_MODEL = "SWITCH_MODEL"
    CHANGE_PHASE = "CHANGE_PHASE"
    FILTER_LESSONS = "FILTER_LESSONS"
    SIGNAL_CONVERGENCE = "SIGNAL_CONVERGENCE"
    REQUEST_ANALYSIS = "REQUEST_ANALYSIS"
    ADJUST_STRATEGY = "ADJUST_STRATEGY"


class ActionStatus(Enum):
    """Status of action execution."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"


@dataclass
class LLMAction:
    """Represents a parsed action request from the LLM."""
    action_type: ActionType
    params: Dict[str, Any]
    raw_text: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: ActionStatus = ActionStatus.PENDING
    rejection_reason: Optional[str] = None
    execution_result: Optional[Any] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "action_type": self.action_type.value,
            "params": self.params,
            "raw_text": self.raw_text,
            "timestamp": self.timestamp,
            "status": self.status.value,
            "rejection_reason": self.rejection_reason,
            "execution_result": str(self.execution_result) if self.execution_result else None,
        }


@dataclass
class ActionResult:
    """Result of action execution."""
    action: LLMAction
    success: bool
    message: str
    data: Optional[Any] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "action": self.action.to_dict(),
            "success": self.success,
            "message": self.message,
            "data": str(self.data) if self.data else None,
        }


class LLMActionParser:
    """
    Parser for LLM action requests.

    Detects action markers in the format:
    @@ACTION:ACTION_TYPE:{"param": "value"}@@

    Example:
    @@ACTION:SEARCH_MODELS:{"keywords": ["transformer", "attention"]}@@
    """

    # Regex pattern for action markers
    ACTION_PATTERN = re.compile(
        r'@@ACTION:(\w+):(\{[^}]+\})@@',
        re.MULTILINE
    )

    # Rate limiting: max actions per response
    MAX_ACTIONS_PER_RESPONSE = 3

    # Validation schemas for each action type
    PARAM_SCHEMAS: Dict[ActionType, Dict[str, Any]] = {
        ActionType.SEARCH_MODELS: {
            "required": ["keywords"],
            "optional": ["num_results"],
            "types": {"keywords": list, "num_results": int},
        },
        ActionType.SWITCH_MODEL: {
            "required": ["model_name"],
            "optional": ["reason"],
            "types": {"model_name": str, "reason": str},
        },
        ActionType.CHANGE_PHASE: {
            "required": ["phase"],
            "optional": ["reason"],
            "types": {"phase": str, "reason": str},
            "allowed_values": {
                "phase": ["BREADTH_SEARCH", "DEEP_EXPLOIT", "RADICAL_PIVOT", "ENSEMBLE_SYNTHESIS"]
            },
        },
        ActionType.FILTER_LESSONS: {
            "required": ["keywords"],
            "optional": ["count", "lesson_type"],
            "types": {"keywords": list, "count": int, "lesson_type": str},
        },
        ActionType.SIGNAL_CONVERGENCE: {
            "required": ["confidence"],
            "optional": ["reason"],
            "types": {"confidence": (int, float), "reason": str},
            "constraints": {"confidence": {"min": 0.0, "max": 1.0}},
        },
        ActionType.REQUEST_ANALYSIS: {
            "required": ["type"],
            "optional": ["target_features"],
            "types": {"type": str, "target_features": list},
            "allowed_values": {
                "type": ["feature_importance", "correlation", "distribution", "missing_values"]
            },
        },
        ActionType.ADJUST_STRATEGY: {
            "required": ["strategy"],
            "optional": ["reason", "target_file"],
            "types": {"strategy": str, "reason": str, "target_file": str},
            "allowed_values": {
                "strategy": ["regenerate", "diff", "redraft", "skip"]
            },
        },
    }

    def __init__(self):
        self.last_parse_count = 0

    def parse(self, response: str) -> List[LLMAction]:
        """
        Parse LLM response for action requests.

        Args:
            response: Full LLM response text

        Returns:
            List of parsed LLMAction objects (max MAX_ACTIONS_PER_RESPONSE)
        """
        if not response:
            return []

        actions = []
        matches = self.ACTION_PATTERN.findall(response)

        for action_type_str, params_str in matches:
            try:
                # Validate action type
                action_type = ActionType(action_type_str)

                # Parse JSON params
                params = json.loads(params_str)

                action = LLMAction(
                    action_type=action_type,
                    params=params,
                    raw_text=f"@@ACTION:{action_type_str}:{params_str}@@",
                )
                actions.append(action)

            except (ValueError, json.JSONDecodeError) as e:
                # Invalid action type or malformed JSON - skip silently
                continue

            # Rate limiting
            if len(actions) >= self.MAX_ACTIONS_PER_RESPONSE:
                break

        self.last_parse_count = len(actions)
        return actions

    def validate(self, action: LLMAction) -> bool:
        """
        Validate action parameters against schema.

        Args:
            action: LLMAction to validate

        Returns:
            True if valid, False otherwise (sets rejection_reason)
        """
        schema = self.PARAM_SCHEMAS.get(action.action_type)
        if not schema:
            action.rejection_reason = f"Unknown action type: {action.action_type}"
            return False

        params = action.params

        # Check required params
        for required in schema.get("required", []):
            if required not in params:
                action.rejection_reason = f"Missing required param: {required}"
                return False

        # Check types
        type_specs = schema.get("types", {})
        for param_name, expected_type in type_specs.items():
            if param_name in params:
                value = params[param_name]
                if isinstance(expected_type, tuple):
                    if not isinstance(value, expected_type):
                        action.rejection_reason = f"Invalid type for {param_name}: expected {expected_type}, got {type(value)}"
                        return False
                elif not isinstance(value, expected_type):
                    action.rejection_reason = f"Invalid type for {param_name}: expected {expected_type.__name__}, got {type(value).__name__}"
                    return False

        # Check allowed values
        allowed = schema.get("allowed_values", {})
        for param_name, allowed_list in allowed.items():
            if param_name in params:
                value = params[param_name]
                if value not in allowed_list:
                    action.rejection_reason = f"Invalid value for {param_name}: {value}. Allowed: {allowed_list}"
                    return False

        # Check constraints
        constraints = schema.get("constraints", {})
        for param_name, constraint in constraints.items():
            if param_name in params:
                value = params[param_name]
                if "min" in constraint and value < constraint["min"]:
                    action.rejection_reason = f"{param_name} must be >= {constraint['min']}"
                    return False
                if "max" in constraint and value > constraint["max"]:
                    action.rejection_reason = f"{param_name} must be <= {constraint['max']}"
                    return False

        return True

    def extract_content(self, response: str) -> str:
        """
        Extract clean content from response, removing action markers.

        Args:
            response: Full LLM response

        Returns:
            Response text without action markers
        """
        if not response:
            return ""

        # Remove action markers
        clean = self.ACTION_PATTERN.sub("", response)

        # Clean up extra whitespace
        clean = re.sub(r'\n{3,}', '\n\n', clean)

        return clean.strip()

    def format_actions_for_log(self, actions: List[LLMAction]) -> List[Dict]:
        """Format actions for JSON logging."""
        return [action.to_dict() for action in actions]


class LLMActionLogger:
    """
    Detailed logger for LLM actions.

    Maintains a separate log file (llm_actions.json) with full action history.
    """

    def __init__(self, log_path: str = "llm_actions.json"):
        self.log_path = log_path
        self.action_log: List[Dict] = []
        self._load_existing_log()

    def _load_existing_log(self):
        """Load existing log file if present."""
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                self.action_log = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.action_log = []

    def log_action(self, action: LLMAction, iteration: int, node_id: Optional[str] = None):
        """
        Log an action request.

        Args:
            action: The LLMAction to log
            iteration: Current MCTS iteration number
            node_id: ID of the node being processed
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "node_id": node_id,
            **action.to_dict(),
        }
        self.action_log.append(entry)
        self._save_log()

    def log_result(self, result: ActionResult, iteration: int, node_id: Optional[str] = None):
        """
        Log an action result.

        Args:
            result: The ActionResult to log
            iteration: Current MCTS iteration number
            node_id: ID of the node being processed
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "node_id": node_id,
            **result.to_dict(),
        }
        self.action_log.append(entry)
        self._save_log()

    def _save_log(self):
        """Save log to file."""
        try:
            with open(self.log_path, 'w', encoding='utf-8') as f:
                json.dump(self.action_log, f, indent=2, ensure_ascii=False)
        except Exception:
            pass  # Non-blocking

    def get_statistics(self) -> Dict[str, Any]:
        """Get action statistics."""
        stats = {
            "total_actions": len(self.action_log),
            "by_type": {},
            "by_status": {},
        }

        for entry in self.action_log:
            action_type = entry.get("action_type", "unknown")
            status = entry.get("status", "unknown")

            stats["by_type"][action_type] = stats["by_type"].get(action_type, 0) + 1
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

        return stats
