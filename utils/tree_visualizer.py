# utils/tree_visualizer.py
"""
MCTS tree visualization for MARS.

Renders a PNG showing:
  - VALID nodes (green)
  - BUGGY nodes (red)
  - Metric value inside each node
  - Gold border on the best node
  - Legend with best node info
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

from core.tree_node import TreeNode, NodeStatus, ActionType


class TreeVisualizer:
    """Visualizes the MCTS search tree as a PNG."""

    # ── Colors ───────────────────────────────────────────────────────────
    COLOR_VALID   = "#27ae60"
    COLOR_BUGGY   = "#e74c3c"
    COLOR_PENDING = "#bdc3c7"
    COLOR_ROOT    = "#2c3e50"
    COLOR_BEST    = "#f39c12"
    COLOR_EDGE    = "#7f8c8d"
    COLOR_TEXT    = "white"

    H_GAP = 1.4   # horizontal spacing between sibling nodes (data units)
    V_GAP = 1.0   # vertical spacing between depth levels (data units)

    # ─────────────────────────────────────────────────────────────────────

    def generate(
        self,
        root: TreeNode,
        best_node: Optional[TreeNode],
        output_path: Path,
        metric_name: str = "metric",
        lower_is_better: bool = False,
    ) -> None:
        if not _MPL_OK:
            print("[TreeVisualizer] matplotlib not installed - skipping")
            return

        all_nodes = self._bfs(root)
        if len(all_nodes) <= 1:
            print("[TreeVisualizer] No explored nodes - skipping")
            return

        positions = self._layout(root)

        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Count leaves and depth for figure sizing
        n_leaves = max(1, sum(1 for n in all_nodes if not n.children))
        max_depth = int(round(-y_min / self.V_GAP)) + 1

        fig_w = max(12, n_leaves * 2.2)
        fig_h = max(6,  max_depth * 2.2)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        # Set axis limits with padding so nodes near the border don't clip
        x_pad = self.H_GAP * 0.8
        y_pad = self.V_GAP * 1.2
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        # Node radius in data units — proportional to spacing
        r = self.H_GAP * 0.30

        self._draw_edges(ax, all_nodes, positions)
        self._draw_nodes(ax, all_nodes, positions, best_node, root.id, r)

        # ── Title ────────────────────────────────────────────────────────
        valid_n = sum(1 for n in all_nodes if n.status == NodeStatus.VALID)
        buggy_n = sum(1 for n in all_nodes if n.status == NodeStatus.BUGGY)
        total_n = len(all_nodes) - 1
        ax.set_title(
            f"MARS - MCTS Search Tree\n"
            f"Nodes: {total_n}  |  Valid: {valid_n}  |  Buggy: {buggy_n}",
            fontsize=12, fontweight="bold", pad=10,
        )

        # ── Legend (inside the figure, bottom) ───────────────────────────
        entries = [
            mpatches.Patch(color=self.COLOR_ROOT,    label="Root"),
            mpatches.Patch(color=self.COLOR_VALID,   label="Valid (OK)"),
            mpatches.Patch(color=self.COLOR_BUGGY,   label="Buggy"),
            mpatches.Patch(color=self.COLOR_PENDING, label="Pending"),
        ]
        if best_node and best_node.metric_value is not None:
            direction = "lower=better" if lower_is_better else "higher=better"
            entries.append(mpatches.Patch(
                edgecolor=self.COLOR_BEST, facecolor="none", linewidth=2,
                label=(
                    f"Best [{direction}]:  "
                    f"id={best_node.id}  "
                    f"{metric_name}={best_node.metric_value:.6f}"
                ),
            ))

        ax.legend(
            handles=entries,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=min(len(entries), 5),
            fontsize=8,
            framealpha=0.85,
        )

        fig.subplots_adjust(top=0.88, bottom=0.12, left=0.02, right=0.98)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[TreeVisualizer] Tree saved: {output_path}")

    # ── Layout ───────────────────────────────────────────────────────────

    def _layout(self, root: TreeNode) -> Dict[str, Tuple[float, float]]:
        """
        Post-order DFS:
        - Leaves get sequential x (spaced H_GAP apart).
        - Parents centre over their children.
        - y = -depth * V_GAP  (root at top = y=0).
        """
        node_x: Dict[str, float] = {}
        cursor = [0.0]

        def _assign(node: TreeNode) -> float:
            if not node.children:
                x = cursor[0]
                cursor[0] += self.H_GAP
            else:
                xs = [_assign(c) for c in node.children]
                x = sum(xs) / len(xs)
            node_x[node.id] = x
            return x

        def _collect(node: TreeNode, depth: int) -> Dict[str, Tuple[float, float]]:
            pos = {node.id: (node_x[node.id], -depth * self.V_GAP)}
            for c in node.children:
                pos.update(_collect(c, depth + 1))
            return pos

        _assign(root)
        return _collect(root, 0)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _bfs(root: TreeNode) -> List[TreeNode]:
        nodes, q = [], deque([root])
        while q:
            n = q.popleft()
            nodes.append(n)
            q.extend(n.children)
        return nodes

    def _node_color(self, node: TreeNode, root_id: str) -> str:
        if node.id == root_id:
            return self.COLOR_ROOT
        if node.status == NodeStatus.VALID:
            return self.COLOR_VALID
        if node.status == NodeStatus.BUGGY:
            return self.COLOR_BUGGY
        return self.COLOR_PENDING

    # ── Drawing ──────────────────────────────────────────────────────────

    def _draw_edges(
        self,
        ax,
        nodes: List[TreeNode],
        pos: Dict[str, Tuple[float, float]],
    ) -> None:
        for node in nodes:
            if node.id not in pos:
                continue
            px, py = pos[node.id]
            for child in node.children:
                if child.id not in pos:
                    continue
                cx, cy = pos[child.id]
                ax.annotate(
                    "",
                    xy=(cx, cy),
                    xytext=(px, py),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=self.COLOR_EDGE,
                        lw=1.2,
                        mutation_scale=12,
                        shrinkA=18,
                        shrinkB=18,
                    ),
                    zorder=1,
                )

    def _draw_nodes(
        self,
        ax,
        nodes: List[TreeNode],
        pos: Dict[str, Tuple[float, float]],
        best_node: Optional[TreeNode],
        root_id: str,
        r: float,
    ) -> None:
        # We draw nodes using scatter so they're always circular in screen space.
        # Convert data-unit radius to scatter marker size (points^2).
        # This is approximate: we use a fixed display size and adjust text accordingly.
        MARKER_SIZE   = 4500   # base scatter size (points^2)
        BEST_RING     = 6500   # ring around best node

        xs_all = [pos[n.id][0] for n in nodes if n.id in pos]
        ys_all = [pos[n.id][1] for n in nodes if n.id in pos]

        for node in nodes:
            if node.id not in pos:
                continue
            x, y    = pos[node.id]
            is_best = best_node is not None and node.id == best_node.id
            color   = self._node_color(node, root_id)

            # Main filled circle
            ax.scatter(x, y, s=MARKER_SIZE, c=color, zorder=3, linewidths=0)

            # Gold ring for best node
            if is_best:
                ax.scatter(
                    x, y,
                    s=BEST_RING,
                    facecolors="none",
                    edgecolors=self.COLOR_BEST,
                    linewidths=3,
                    zorder=4,
                )

            # Label
            if node.id == root_id:
                label = "ROOT"
            else:
                action = (
                    node.action_type.value[:3].upper()
                    if node.action_type else "?"
                )
                if node.metric_value is not None:
                    metric_str = f"{node.metric_value:.4f}"
                elif node.status == NodeStatus.BUGGY:
                    metric_str = "BUG"
                else:
                    metric_str = "--"
                label = f"{action}:{node.id}\n{metric_str}"

            ax.text(
                x, y, label,
                ha="center", va="center",
                fontsize=8,
                color=self.COLOR_TEXT,
                fontweight="bold" if is_best else "normal",
                zorder=5,
            )
