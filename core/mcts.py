# core/mcts.py
import time
import math
from typing import Optional, List, Tuple
from core.tree_node import TreeNode, NodeStatus, ActionType
from core.config import Config
import json
from pathlib import Path

class MCTSEngine:
    """
    Budget-Aware Monte Carlo Tree Search implementation.
    Follows Algorithm 1 and 2 from the MARS paper.
    """
    
    def __init__(
        self,
        time_budget: int = 86400,  # 24 hours default
        max_improvements: int = None,
        max_debug_attempts: int = None,
        penalty_weight: float = None,
        c_uct: float = None,
        stagnation_threshold: int = None,
        lower_is_better: bool = None,
    ):
        # Hyperparameters (from Config or override)
        self.time_budget = time_budget
        self.max_improvements = max_improvements or Config.NI
        self.max_debug_attempts = max_debug_attempts or Config.ND
        self.penalty_weight = penalty_weight or Config.W
        self.c_uct = c_uct or Config.C_UCT
        self.stagnation_threshold = stagnation_threshold or Config.NS
        self.lower_is_better = lower_is_better if lower_is_better is not None else Config.LOWER_IS_BETTER
        
        # Tree structure
        self.root: TreeNode = TreeNode(depth=0)
        self.root.status = NodeStatus.PENDING
        
        # Best solution tracking
        self.best_node: Optional[TreeNode] = None
        self.best_metric: float = float('-inf')
        
        # Global statistics for normalization
        self.explored_nodes: List[TreeNode] = []
        self.global_min_metric: float = float('inf')
        self.global_max_metric: float = float('-inf')
        
        # Search control
        self.start_time: float = 0
        self.iterations: int = 0
        self.valid_nodes_since_improvement: int = 0
        
        # Logging
        self.search_log: List[dict] = []
    
    def search(self) -> TreeNode:
        """
        Main MCTS search loop (Algorithm 2 from paper).
        
        Returns:
            Best node found
        """
        self.start_time = time.time()
        print(f"Starting MCTS search with {self.time_budget}s budget...")
        
        while not self._should_stop():
            self.iterations += 1
            
            # 1. SELECTION: Choose node to expand
            selected_node = self._select_node()
            
            if selected_node is None:
                print("No nodes available for expansion. Ending search.")
                break
            
            # 2. EXPANSION: Create new child node
            new_node = self._expand(selected_node)
            
            if new_node is None:
                continue
            
            # 3. SIMULATION: Execute the solution (handled externally)
            # This will be done by the orchestrator
            # For now, we just mark it as ready for execution
            new_node.status = NodeStatus.PENDING
            
            # Log iteration
            self._log_iteration(selected_node, new_node)
            
            print(f"Iteration {self.iterations}: Created node {new_node.id} "
                  f"(action={new_node.action_type.value if new_node.action_type else 'root'})")
            
            # Return node for execution (orchestrator will call update_after_execution)
            yield new_node
        
        elapsed = time.time() - self.start_time
        print(f"\nMCTS search completed:")
        print(f"  Time elapsed: {elapsed:.1f}s")
        print(f"  Iterations: {self.iterations}")
        print(f"  Nodes explored: {len(self.explored_nodes)}")
        print(f"  Best metric: {self.best_metric}")
        
        return self.best_node
    
    def _select_node(self) -> Optional[TreeNode]:
        """
        SELECT phase: Traverse tree using UCT until finding expandable node.
        Implements node selection strategy from Section 4.4.2.
        """
        node = self.root
        
        # Check if root should be reactivated
        if self._should_reactivate_root():
            print(f"Reactivating root (stagnation: {self.valid_nodes_since_improvement} nodes)")
            self.root.is_fully_expanded = False
            self.valid_nodes_since_improvement = 0
            return self.root
        
        # Traverse down the tree
        while not self._is_expandable(node):
            if len(node.children) == 0:
                # Leaf node that's fully expanded -> reactivate root
                return self.root
            
            # Select best child using UCT
            node = self._select_best_child(node)
        
        return node
    
    def _select_best_child(self, node: TreeNode) -> TreeNode:
        """Select child with highest UCT value"""
        best_child = None
        best_uct = float('-inf')
        
        for child in node.children:
            uct = child.uct_value(self.c_uct, node.visit_count)
            if uct > best_uct:
                best_uct = uct
                best_child = child
        
        return best_child
    
    def _is_expandable(self, node: TreeNode) -> bool:
        """
        Check if node can be expanded (Section 4.4.2).

        With the internal debugging loop, MCTS only expands via DRAFT
        (from root) and IMPROVE (from valid nodes). Buggy nodes are
        handled inline by the orchestrator, not as MCTS actions.
        """
        if node.is_fully_expanded:
            return False

        if node == self.root:
            return not node.is_fully_expanded

        if node.status == NodeStatus.VALID:
            return node.can_improve(self.max_improvements)

        # Buggy nodes are NOT expandable in MCTS — they are debugged
        # inline by the orchestrator before the node is returned.
        return False
    
    def _expand(self, node: TreeNode) -> Optional[TreeNode]:
        """
        EXPANSION phase: Create new child node.
        Only DRAFT (from root) and IMPROVE (from valid nodes).
        DEBUG is handled inline by the orchestrator.
        """
        if node == self.root:
            action_type = ActionType.DRAFT
        elif node.status == NodeStatus.VALID:
            action_type = ActionType.IMPROVE
        else:
            return None

        # Create child node with time_limit from budget
        child = TreeNode(
            parent=node,
            action_type=action_type,
            depth=node.depth + 1,
        )
        child.time_limit = float(self.time_budget)

        node.add_child(child)

        # Check if parent is now fully expanded
        if node != self.root:
            if node.status == NodeStatus.VALID and node.improvement_attempts >= self.max_improvements:
                node.mark_fully_expanded()

        return child
    
    def _is_metric_improvement(self, new_metric: float, old_metric: float) -> bool:
        """Check if new_metric is better than old_metric considering direction."""
        if self.lower_is_better:
            return new_metric < old_metric
        else:
            return new_metric > old_metric

    def update_after_execution(self, node: TreeNode, execution_result):
        """
        Update tree after executing a node (BACKPROPAGATION phase).
        Called by orchestrator after running the solution.
        """
        # Set execution result
        node.set_execution_result(execution_result)
        self.explored_nodes.append(node)

        # Update global statistics
        if node.status == NodeStatus.VALID and node.metric_value is not None:
            self.global_min_metric = min(self.global_min_metric, node.metric_value)
            self.global_max_metric = max(self.global_max_metric, node.metric_value)

            # Track stagnation (direction-aware comparison)
            if self.best_node is None or self._is_metric_improvement(node.metric_value, self.best_metric):
                self.best_metric = node.metric_value
                self.best_node = node
                self.valid_nodes_since_improvement = 0
                print(f"  New best solution! Metric: {self.best_metric:.6f}")
            else:
                self.valid_nodes_since_improvement += 1

        # Compute reward (direction-aware)
        reward = node.compute_reward(
            self.global_min_metric,
            self.global_max_metric,
            self.penalty_weight,
            self.lower_is_better,
        )

        # Backpropagate reward
        self._backpropagate(node, reward)
    
    def _backpropagate(self, node: TreeNode, reward: float):
        """Update statistics along path to root"""
        current = node
        while current is not None:
            current.update_stats(reward)
            current = current.parent
    
    def _should_reactivate_root(self) -> bool:
        """Check if root should be reactivated (Section 4.4.2)"""
        return (
            self.valid_nodes_since_improvement >= self.stagnation_threshold and
            self.best_node is not None
        )
    
    def _should_stop(self) -> bool:
        """Check if search should terminate"""
        elapsed = time.time() - self.start_time
        if elapsed >= self.time_budget:
            print(f"Time budget exceeded ({elapsed:.1f}s >= {self.time_budget}s)")
            return True
        return False
    
    def _log_iteration(self, parent: TreeNode, child: TreeNode):
        """Log iteration details"""
        self.search_log.append({
            "iteration": self.iterations,
            "parent_id": parent.id,
            "child_id": child.id,
            "action": child.action_type.value if child.action_type else None,
            "depth": child.depth,
            "timestamp": time.time() - self.start_time,
        })
    
    def save_search_log(self, filepath: Path):
        """Save search log to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.search_log, f, indent=2)
    
    def get_statistics(self) -> dict:
        """Get search statistics"""
        return {
            "iterations": self.iterations,
            "nodes_explored": len(self.explored_nodes),
            "valid_nodes": sum(1 for n in self.explored_nodes if n.status == NodeStatus.VALID),
            "buggy_nodes": sum(1 for n in self.explored_nodes if n.status == NodeStatus.BUGGY),
            "best_metric": self.best_metric,
            "time_elapsed": time.time() - self.start_time,
        }