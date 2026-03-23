# core/mcts.py
import time
import math
from typing import Optional, List, Tuple, Dict
from core.tree_node import TreeNode, NodeStatus, ActionType
from core.config import Config
import json
from pathlib import Path


class MCTSEngine:
    """
    Budget-Aware Monte Carlo Tree Search implementation.
    Follows Algorithm 1 and 2 from the MARS paper.

    Enhanced with model-aware exploration: tracks which ML models have been
    explored and rotates to new models when a branch stagnates.
    """

    # Number of valid iterations without improvement before switching model
    MODEL_STAGNATION_THRESHOLD: int = 4

    def __init__(
        self,
        time_budget: int = 86400,  # 24 hours default
        max_improvements: int = None,
        max_debug_attempts: int = None,
        penalty_weight: float = None,
        c_uct: float = None,
        stagnation_threshold: int = None,
        lower_is_better: bool = None,
        max_depth: int = None,
    ):
        # Hyperparameters (from Config or override)
        self.time_budget = time_budget
        self.max_improvements = max_improvements or Config.NI
        self.max_debug_attempts = max_debug_attempts or Config.ND
        self.penalty_weight = penalty_weight or Config.W
        self.c_uct = c_uct or Config.C_UCT
        self.stagnation_threshold = stagnation_threshold or Config.NS
        self.lower_is_better = lower_is_better if lower_is_better is not None else Config.LOWER_IS_BETTER
        self.max_depth = max_depth if max_depth is not None else Config.MAX_DEPTH

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

        # ═══════════════════════════════════════════════════════════════
        # Model exploration tracking (NEW)
        # ═══════════════════════════════════════════════════════════════
        self.discovered_models: List[Dict[str, str]] = []  # From SearchAgent
        self.current_model_index: int = 0  # Index of model being explored
        self.model_exploration_stats: Dict[int, Dict] = {}  # {model_idx: {best_metric, valid_count, stagnant}}
        self.models_exhausted: bool = False  # True when all models explored

        # Logging (enhanced with metrics)
        self.search_log: List[dict] = []
        self.metrics_log: List[dict] = []  # Detailed metrics per node
    
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

        Priority order:
        1. If root should be reactivated (stagnation), return root for new DRAFT
        2. Find an expandable VALID node for IMPROVE
        3. If no expandable nodes exist, reactivate root for new DRAFT
        """
        # Check if root should be reactivated due to stagnation
        if self._should_reactivate_root():
            print(f"  [MCTS] Reactivating root (stagnation: {self.valid_nodes_since_improvement} valid nodes without improvement)")
            self.root.is_fully_expanded = False
            self.valid_nodes_since_improvement = 0
            return self.root

        # If root is not fully expanded (no DRAFT yet or reactivated), use it
        if not self.root.is_fully_expanded:
            return self.root

        # ═══════════════════════════════════════════════════════════════
        # Root is fully expanded: search for VALID nodes to IMPROVE
        # ═══════════════════════════════════════════════════════════════
        expandable_node = self._find_expandable_node_in_tree()

        if expandable_node:
            print(f"  [MCTS] Selected node {expandable_node.id} for IMPROVE (depth={expandable_node.depth})")
            return expandable_node

        # ═══════════════════════════════════════════════════════════════
        # No expandable nodes found: reactivate root for new DRAFT
        # ═══════════════════════════════════════════════════════════════
        print(f"  [MCTS] No expandable nodes in tree, reactivating root for new DRAFT")
        self.root.is_fully_expanded = False
        return self.root

    def _find_expandable_node_in_tree(self) -> Optional[TreeNode]:
        """
        Find the best expandable node in the tree using UCT traversal.

        Returns the highest-UCT expandable node (VALID that can still IMPROVE).
        Returns None if no expandable nodes exist.
        """
        # Collect all expandable nodes
        expandable_nodes = []
        self._collect_expandable_nodes(self.root, expandable_nodes)

        if not expandable_nodes:
            return None

        # If only one, return it
        if len(expandable_nodes) == 1:
            return expandable_nodes[0]

        # Select best by UCT value
        best_node = None
        best_uct = float('-inf')
        for node in expandable_nodes:
            parent_visits = node.parent.visit_count if node.parent else 1
            uct = node.uct_value(self.c_uct, parent_visits)
            if uct > best_uct:
                best_uct = uct
                best_node = node

        return best_node

    def _collect_expandable_nodes(self, node: TreeNode, result: List[TreeNode]):
        """Recursively collect all expandable nodes in the subtree."""
        # Check if this node is expandable (only VALID nodes can IMPROVE)
        if node != self.root and self._is_expandable(node):
            result.append(node)

        # Recurse into children
        for child in node.children:
            self._collect_expandable_nodes(child, result)
    
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

        Depth limit (MAX_DEPTH) balances tree breadth vs depth:
        valid nodes beyond max_depth are marked fully expanded,
        forcing the search back to root for new DRAFTs.

        CRITICAL: After creating a DRAFT, the root is marked as
        fully_expanded to force exploration of IMPROVE actions on
        valid children before creating another DRAFT.
        """
        if node == self.root:
            action_type = ActionType.DRAFT
        elif node.status == NodeStatus.VALID:
            # Depth limit: prevent infinite deepening of improvements
            if node.depth >= self.max_depth:
                print(f"  [MCTS] Node {node.id} at max depth ({self.max_depth}), forcing breadth exploration")
                node.mark_fully_expanded()
                return None
            action_type = ActionType.IMPROVE
        else:
            return None

        # Create child node with time_limit = per-solution execution cap
        # (used as L(v) in the reward function Eq 4: R = G · [t/L]^w)
        child = TreeNode(
            parent=node,
            action_type=action_type,
            depth=node.depth + 1,
        )
        child.time_limit = float(Config.MAX_EXECUTION_TIME)

        # ═══════════════════════════════════════════════════════════════
        # Propagate model info from parent (IMPROVE) or current model (DRAFT)
        # ═══════════════════════════════════════════════════════════════
        if action_type == ActionType.IMPROVE:
            if node.model_name:
                # IMPROVE: inherit model from parent
                child.model_name = node.model_name
                child.model_index = node.model_index
            else:
                # Fallback for legacy nodes without model info
                current_model = self.get_current_model()
                if current_model:
                    child.model_name = current_model.get("name", "Unknown")
                    child.model_index = self.current_model_index
        elif action_type == ActionType.DRAFT:
            # DRAFT: use current model from exploration
            current_model = self.get_current_model()
            if current_model:
                child.model_name = current_model.get("name", "Unknown")
                child.model_index = self.current_model_index

        node.add_child(child)

        # ═══════════════════════════════════════════════════════════════
        # CRITICAL FIX: After DRAFT, mark root as fully_expanded to force
        # exploration of the new node via IMPROVE before creating more DRAFTs.
        # Root will be reactivated by _should_reactivate_root() when:
        # - Stagnation occurs (NS valid nodes without improvement)
        # - Model needs to be switched (MODEL_STAGNATION_THRESHOLD)
        # - No expandable nodes remain in the tree
        # ═══════════════════════════════════════════════════════════════
        if node == self.root:
            self.root.is_fully_expanded = True
            print(f"  [MCTS] Root marked fully_expanded after DRAFT (will explore children)")

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

        Enhanced with model-aware tracking and detailed metrics logging.
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

            # ═══════════════════════════════════════════════════════════════
            # Model exploration tracking (NEW)
            # ═══════════════════════════════════════════════════════════════
            # Update branch-level tracking
            node.branch_valid_count = (node.parent.branch_valid_count + 1) if node.parent else 1

            # Update branch best metric
            parent_best = node.parent.branch_best_metric if node.parent else None
            if parent_best is None or self._is_metric_improvement(node.metric_value, parent_best):
                node.branch_best_metric = node.metric_value
            else:
                node.branch_best_metric = parent_best

            # Update model-level statistics
            self.update_model_stats(node)

        # Compute reward (direction-aware)
        reward = node.compute_reward(
            self.global_min_metric,
            self.global_max_metric,
            self.penalty_weight,
            self.lower_is_better,
        )

        # Backpropagate reward
        self._backpropagate(node, reward)

        # Log detailed metrics (NEW)
        self.log_node_metrics(node)
    
    def _backpropagate(self, node: TreeNode, reward: float):
        """Update statistics along path to root"""
        current = node
        while current is not None:
            current.update_stats(reward)
            current = current.parent
    
    def _should_reactivate_root(self) -> bool:
        """
        Check if root should be reactivated (Section 4.4.2).

        Root is reactivated when NS valid nodes have been explored
        without improvement to the global best metric.

        NOTE: Model stagnation is handled separately by the orchestrator
        which calls switch_to_next_model() after execution. That function
        directly sets root.is_fully_expanded = False.
        """
        if self.valid_nodes_since_improvement >= self.stagnation_threshold and self.best_node is not None:
            print(f"  [MCTS] Global stagnation detected ({self.valid_nodes_since_improvement} >= {self.stagnation_threshold})")
            return True
        return False
    
    def _should_stop(self) -> bool:
        """Check if search should terminate"""
        elapsed = time.time() - self.start_time
        if elapsed >= self.time_budget:
            print(f"Time budget exceeded ({elapsed:.1f}s >= {self.time_budget}s)")
            return True
        return False
    
    def _log_iteration(self, parent: TreeNode, child: TreeNode):
        """Log iteration details with enhanced metrics"""
        self.search_log.append({
            "iteration": self.iterations,
            "parent_id": parent.id,
            "child_id": child.id,
            "action": child.action_type.value if child.action_type else None,
            "depth": child.depth,
            "timestamp": time.time() - self.start_time,
            # Enhanced fields
            "model_name": child.model_name,
            "model_index": child.model_index,
        })

    def log_node_metrics(self, node: TreeNode):
        """Log detailed metrics for a node after execution (for mcts_log)"""
        entry = {
            "iteration": self.iterations,
            "node_id": node.id,
            "parent_id": node.parent.id if node.parent else None,
            "action": node.action_type.value if node.action_type else None,
            "depth": node.depth,
            "status": node.status.value,
            "metric_value": node.metric_value,
            "execution_time": node.execution_time,
            "model_name": node.model_name,
            "model_index": node.model_index,
            "branch_valid_count": node.branch_valid_count,
            "branch_best_metric": node.branch_best_metric,
            "global_best_metric": self.best_metric,
            "timestamp": time.time() - self.start_time,
            "idea_summary": (node.solution.idea[:200] + "...") if node.solution and node.solution.idea else "",
        }
        self.metrics_log.append(entry)

    # ═══════════════════════════════════════════════════════════════
    # Model Exploration Management (NEW)
    # ═══════════════════════════════════════════════════════════════

    def set_discovered_models(self, models: List[Dict[str, str]]):
        """Register discovered ML models for systematic exploration"""
        self.discovered_models = models
        self.current_model_index = 0
        self.models_exhausted = False
        # Initialize stats for each model
        for i in range(len(models)):
            self.model_exploration_stats[i] = {
                "best_metric": None,
                "valid_count": 0,
                "stagnant_count": 0,
                "explored": False,
            }
        print(f"[MCTS] Registered {len(models)} models for exploration:")
        for i, m in enumerate(models):
            print(f"  [{i}] {m.get('name', 'Unknown')}")

    def get_current_model(self) -> Optional[Dict[str, str]]:
        """Get the current model to explore"""
        if not self.discovered_models:
            return None
        if self.current_model_index >= len(self.discovered_models):
            return None
        return self.discovered_models[self.current_model_index]

    def get_current_model_name(self) -> str:
        """Get name of current model"""
        model = self.get_current_model()
        return model.get("name", "Unknown") if model else "Unknown"

    def should_switch_model(self) -> bool:
        """
        Check if we should switch to the next model.

        Returns True if:
        - Current model has MODEL_STAGNATION_THRESHOLD valid nodes without improvement
        - All improvements on current model exhausted
        """
        if not self.discovered_models:
            return False

        stats = self.model_exploration_stats.get(self.current_model_index, {})
        stagnant = stats.get("stagnant_count", 0)

        if stagnant >= self.MODEL_STAGNATION_THRESHOLD:
            print(f"[MCTS] Model '{self.get_current_model_name()}' stagnated "
                  f"({stagnant} valid nodes without improvement)")
            return True

        return False

    def switch_to_next_model(self) -> bool:
        """
        Switch to the next unexplored model.

        Returns True if switched successfully, False if all models exhausted.
        """
        # Mark current as explored
        if self.current_model_index < len(self.discovered_models):
            self.model_exploration_stats[self.current_model_index]["explored"] = True

        # Find next unexplored model
        for i in range(len(self.discovered_models)):
            next_idx = (self.current_model_index + 1 + i) % len(self.discovered_models)
            stats = self.model_exploration_stats.get(next_idx, {})
            if not stats.get("explored", False):
                old_name = self.get_current_model_name()
                self.current_model_index = next_idx
                new_name = self.get_current_model_name()
                print(f"[MCTS] Switching model: '{old_name}' -> '{new_name}' (index {next_idx})")
                # Force root reactivation for new DRAFT with new model
                self.root.is_fully_expanded = False
                return True

        # All models explored - switch to the BEST performing model
        self.models_exhausted = True
        best_idx = self._get_best_model_index()
        if best_idx != self.current_model_index:
            old_name = self.get_current_model_name()
            self.current_model_index = best_idx
            new_name = self.get_current_model_name()
            print(f"[MCTS] All models explored. Returning to best: '{new_name}' (was '{old_name}')")
            # Reset stagnation for best model to allow more exploration
            self.model_exploration_stats[best_idx]["stagnant_count"] = 0
            self.root.is_fully_expanded = False
        else:
            print("[MCTS] All models explored. Already on best model.")
        return False

    def _get_best_model_index(self) -> int:
        """Find the model index with the best metric."""
        best_idx = 0
        best_metric = None

        for idx, stats in self.model_exploration_stats.items():
            metric = stats.get("best_metric")
            if metric is None:
                continue
            if best_metric is None:
                best_metric = metric
                best_idx = int(idx)
            elif self._is_metric_improvement(metric, best_metric):
                best_metric = metric
                best_idx = int(idx)

        return best_idx

    def update_model_stats(self, node: TreeNode):
        """Update model exploration statistics after a valid node execution"""
        if node.model_index < 0 or node.model_index >= len(self.discovered_models):
            return

        stats = self.model_exploration_stats[node.model_index]
        stats["valid_count"] += 1

        # Check if this node improved the branch
        improved = False
        if node.metric_value is not None:
            current_best = stats.get("best_metric")
            if current_best is None:
                improved = True
            elif self._is_metric_improvement(node.metric_value, current_best):
                improved = True

            if improved:
                stats["best_metric"] = node.metric_value
                stats["stagnant_count"] = 0
                print(f"[MCTS] Model '{node.model_name}' improved: {node.metric_value:.6f}")
            else:
                stats["stagnant_count"] += 1
                print(f"[MCTS] Model '{node.model_name}' stagnant: "
                      f"{stats['stagnant_count']}/{self.MODEL_STAGNATION_THRESHOLD}")
    
    def save_search_log(self, filepath: Path):
        """Save search log and metrics log to JSON"""
        # Save main search log
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.search_log, f, indent=2)

        # Save detailed metrics log (mcts_metrics.json)
        metrics_path = filepath.parent / "mcts_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metrics_log": self.metrics_log,
                "model_exploration_stats": {
                    str(k): v for k, v in self.model_exploration_stats.items()
                },
                "discovered_models": [
                    {"index": i, "name": m.get("name", "Unknown")}
                    for i, m in enumerate(self.discovered_models)
                ],
                "final_best_metric": self.best_metric,
                "total_iterations": self.iterations,
                "total_nodes_explored": len(self.explored_nodes),
            }, f, indent=2)
    
    def get_statistics(self) -> dict:
        """Get search statistics including model exploration info"""
        # Count models explored
        models_explored = sum(
            1 for stats in self.model_exploration_stats.values()
            if stats.get("explored", False) or stats.get("valid_count", 0) > 0
        )

        return {
            "iterations": self.iterations,
            "nodes_explored": len(self.explored_nodes),
            "valid_nodes": sum(1 for n in self.explored_nodes if n.status == NodeStatus.VALID),
            "buggy_nodes": sum(1 for n in self.explored_nodes if n.status == NodeStatus.BUGGY),
            "best_metric": self.best_metric,
            "time_elapsed": time.time() - self.start_time,
            # Model exploration stats
            "total_models": len(self.discovered_models),
            "models_explored": models_explored,
            "current_model": self.get_current_model_name(),
            "current_model_index": self.current_model_index,
            "models_exhausted": self.models_exhausted,
        }