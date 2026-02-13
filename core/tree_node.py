# core/tree_node.py
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid
import math

class NodeStatus(Enum):
    """Status of a tree node"""
    PENDING = "pending"           # Not yet executed
    EXECUTING = "executing"       # Currently running
    VALID = "valid"              # Executed successfully
    BUGGY = "buggy"              # Execution failed
    FULLY_EXPANDED = "fully_expanded"  # No more children to explore

class ActionType(Enum):
    """Types of actions in MCTS"""
    DRAFT = "draft"              # Create new solution from scratch
    IMPROVE = "improve"          # Improve existing valid solution
    DEBUG = "debug"              # Fix buggy solution

@dataclass
class ExecutionResult:
    """Results from executing a solution"""
    success: bool
    metric_value: Optional[float] = None
    execution_time: float = 0.0
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""
    validation_metric_valid: bool = False
    review_findings: str = ""  # Stage 1: Review findings for two-stage lesson extraction

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "metric_value": self.metric_value,
            "execution_time": self.execution_time,
            "validation_metric_valid": self.validation_metric_valid,
            "error_message": self.error_message,
            "review_findings": self.review_findings,
        }

@dataclass
class Solution:
    """Represents a modular solution (set of files + main script)"""
    idea: str = ""                          # Natural language description
    modules: Dict[str, str] = field(default_factory=dict)  # filename -> code
    main_script: str = ""                   # Orchestration script
    module_descriptions: Dict[str, str] = field(default_factory=dict)  # filename -> description
    
    def get_all_files(self) -> Dict[str, str]:
        """Get all files including main script"""
        files = self.modules.copy()
        if self.main_script:
            files["main.py"] = self.main_script
        return files
    
    def to_dict(self) -> dict:
        return {
            "idea": self.idea,
            "modules": self.modules,
            "main_script": self.main_script,
            "module_descriptions": self.module_descriptions
        }

class TreeNode:
    """
    Node in the MCTS tree representing a solution state.
    Implements the Budget-Aware MCTS algorithm from the paper.
    """
    
    def __init__(
        self,
        solution: Optional[Solution] = None,
        parent: Optional['TreeNode'] = None,
        action_type: Optional[ActionType] = None,
        depth: int = 0
    ):
        # Identity
        self.id: str = str(uuid.uuid4())[:8]
        self.depth: int = depth
        self.parent: Optional[TreeNode] = parent
        self.children: List[TreeNode] = []
        
        # Solution content
        self.solution: Solution = solution or Solution()
        self.action_type: Optional[ActionType] = action_type
        
        # Execution state
        self.status: NodeStatus = NodeStatus.PENDING
        self.execution_result: Optional[ExecutionResult] = None
        self.execution_count: int = 0
        
        # MCTS statistics
        self.visit_count: int = 0
        self.total_reward: float = 0.0
        self.q_value: float = 0.0  # Average reward
        
        # Metrics
        self.metric_value: Optional[float] = None
        self.execution_time: float = 0.0
        self.time_limit: float = 3600.0  # Default 1 hour
        
        # Timestamps
        self.created_at: float = time.time()
        self.executed_at: Optional[float] = None
        
        # Expansion control
        self.improvement_attempts: int = 0  # Number of IMPROVE children
        self.debug_attempts: int = 0        # Number of DEBUG attempts
        self.is_fully_expanded: bool = False
    
    def add_child(self, child: 'TreeNode') -> 'TreeNode':
        """Add a child node"""
        self.children.append(child)
        
        # Track attempt counts
        if child.action_type == ActionType.IMPROVE:
            self.improvement_attempts += 1
        elif child.action_type == ActionType.DEBUG:
            self.debug_attempts += 1
        
        return child
    
    def update_stats(self, reward: float):
        """Update MCTS statistics (backpropagation)"""
        self.visit_count += 1
        self.total_reward += reward
        self.q_value = self.total_reward / self.visit_count if self.visit_count > 0 else 0.0
    
    def set_execution_result(self, result: ExecutionResult):
        """Set execution result and update status"""
        self.execution_result = result
        self.execution_count += 1
        self.executed_at = time.time()
        
        if result.success and result.validation_metric_valid:
            self.status = NodeStatus.VALID
            self.metric_value = result.metric_value
            self.execution_time = result.execution_time
        else:
            self.status = NodeStatus.BUGGY
    
    def compute_reward(
        self, global_min: float, global_max: float,
        w: float = -0.07, lower_is_better: bool = False
    ) -> float:
        """
        Compute efficiency-guided reward (Equation 4 from paper).

        R(v) = G(v) · [t(v)/L(v)]^w

        where:
        - G(v): normalized score (Equation 3)
        - t(v): execution time
        - L(v): time limit
        - w: penalty weight (default -0.07)
        - lower_is_better: if True, invert G(v) so lower metric = higher reward
        """
        if self.metric_value is None or self.execution_time == 0:
            return 0.0

        # Global normalized score G(v) - Equation 3
        if global_max == global_min:
            g_score = 0.5
        else:
            g_score = (self.metric_value - global_min) / (global_max - global_min)
            # Invert for minimize metrics (e.g. logloss, MSE)
            if lower_is_better:
                g_score = 1.0 - g_score

        # Time penalty factor
        time_ratio = self.execution_time / self.time_limit
        time_penalty = time_ratio ** w

        # Final reward
        reward = g_score * time_penalty

        return reward
    
    def uct_value(self, c_uct: float = 1.414, parent_visits: int = 1) -> float:
        """
        Compute UCT value for node selection (Equation 6 from paper).
        
        UCT = Q(s,a) + c_uct * sqrt(ln(N(s)) / N(s,a))
        """
        if self.visit_count == 0:
            return float('inf')  # Prioritize unvisited nodes
        
        exploitation = self.q_value
        exploration = c_uct * math.sqrt(math.log(parent_visits) / self.visit_count)
        
        return exploitation + exploration
    
    def can_improve(self, max_improvements: int) -> bool:
        """Check if node can have more IMPROVE children"""
        return (
            self.status == NodeStatus.VALID and
            self.improvement_attempts < max_improvements and
            not self.is_fully_expanded
        )
    
    def can_debug(self, max_debug_attempts: int) -> bool:
        """Check if node can have more DEBUG children"""
        return (
            self.status == NodeStatus.BUGGY and
            self.debug_attempts < max_debug_attempts and
            not self.is_fully_expanded
        )
    
    def mark_fully_expanded(self):
        """Mark node as fully expanded"""
        self.is_fully_expanded = True
        self.status = NodeStatus.FULLY_EXPANDED
    
    def get_path_from_root(self) -> List['TreeNode']:
        """Get path from root to this node"""
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))
    
    def to_dict(self, include_solution: bool = False) -> dict:
        """Export node as dictionary"""
        data = {
            "id": self.id,
            "depth": self.depth,
            "status": self.status.value,
            "action_type": self.action_type.value if self.action_type else None,
            "visit_count": self.visit_count,
            "q_value": self.q_value,
            "metric_value": self.metric_value,
            "execution_time": self.execution_time,
            "improvement_attempts": self.improvement_attempts,
            "debug_attempts": self.debug_attempts,
            "num_children": len(self.children),
        }
        
        if include_solution:
            data["solution"] = self.solution.to_dict()
        
        if self.execution_result:
            data["execution_result"] = self.execution_result.to_dict()
        
        return data
    
    def __repr__(self) -> str:
        return (
            f"Node(id={self.id}, depth={self.depth}, status={self.status.value}, "
            f"metric={self.metric_value}, visits={self.visit_count})"
        )