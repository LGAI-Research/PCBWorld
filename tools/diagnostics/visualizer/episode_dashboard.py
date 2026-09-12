"""Episode-level trend dashboard — time-series charts from step sidecars.

Reads all step JSON sidecars from an episode directory and produces
up to 6 charts plus an HTML index for browser viewing.
"""

from __future__ import annotations

import json
import os
from typing import Any


def _load_sidecars(episode_dir: str) -> list[dict[str, Any]]:
    """Load all step_NNNN.json sidecars sorted by step number."""
    frames_dir = os.path.join(episode_dir, "frames")
    if not os.path.isdir(frames_dir):
        frames_dir = episode_dir  # fallback: sidecars in root

    sidecars = []
    for f in sorted(os.listdir(frames_dir)):
        if f.startswith("step_") and f.endswith(".json"):
            with open(os.path.join(frames_dir, f)) as fh:
                sidecars.append(json.load(fh))

    sidecars.sort(key=lambda s: s.get("step", 0))
    return sidecars


def generate_episode_dashboard(
    episode_dir: str,
    output_dir: str | None = None,
) -> dict[str, str]:
    """Generate episode trend charts from step sidecar JSONs.

    Creates up to 6 charts + an HTML index:
    1. Reward + Value curve
    2. Action mask evolution heatmap
    3. Router state timeline
    4. DRC violation trend
    5. Spatial action trajectory
    6. Board state metrics

    Args:
        episode_dir: Directory containing frames/step_NNNN.json files.
        output_dir: Where to save charts. Defaults to ``{episode_dir}/dashboard/``.

    Returns:
        Dict mapping chart name to output path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if output_dir is None:
        output_dir = os.path.join(episode_dir, "dashboard")
    os.makedirs(output_dir, exist_ok=True)

    sidecars = _load_sidecars(episode_dir)
    if not sidecars:
        return {}

    outputs: dict[str, str] = {}
    steps = [s.get("step", i + 1) for i, s in enumerate(sidecars)]

    # ---- Chart 1: Reward + Value curve ----
    from tools.diagnostics.visualizer.step_visualizer import plot_reward_curve

    rewards = [s.get("transition", {}).get("r_t", 0.0) for s in sidecars]
    values = [s.get("policy_data", {}).get("V_s_t") for s in sidecars]
    has_values = any(v is not None for v in values)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    plot_reward_curve(ax1, steps, rewards)

    title_suffix = ""
    if has_values:
        v_steps = [s for s, v in zip(steps, values) if v is not None]
        v_vals = [v for v in values if v is not None]
        ax2 = ax1.twinx()
        ax2.plot(v_steps, v_vals, "g-o", linewidth=1.0, markersize=3, label="V(s_t)")
        ax2.set_ylabel("V(s_t)", color="green")
        ax2.legend(loc="upper left")
    else:
        title_suffix = " (no value estimates)"

    ax1.set_title(f"Reward & Value Curve{title_suffix}")
    ax1.legend(loc="lower right")
    path = os.path.join(output_dir, "reward_value_curve.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    outputs["reward_value_curve"] = path

    # ---- Chart 2: Action mask evolution heatmap ----
    masks = [s.get("action_mask") for s in sidecars]
    if any(m is not None for m in masks):
        n_actions = max(len(m) for m in masks if m is not None)
        heatmap = np.full((n_actions, len(sidecars)), 0.5)  # 0.5 = gray (missing)
        for t, m in enumerate(masks):
            if m is not None:
                for a in range(min(len(m), n_actions)):
                    heatmap[a, t] = float(m[a])

        fig, ax = plt.subplots(figsize=(10, 5))
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(["#cc3333", "#cccccc", "#33aa33"])
        ax.imshow(heatmap, cmap=cmap, aspect="auto", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_xlabel("Step")
        ax.set_ylabel("Action ID")
        ax.set_title("Action Mask Evolution")
        ax.set_yticks(range(n_actions))
        path = os.path.join(output_dir, "action_mask_evolution.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        outputs["action_mask_evolution"] = path

    # ---- Chart 3: Router state timeline ----
    states = []
    for s in sidecars:
        t = s.get("transition", {})
        s_next = t.get("s_t_plus_1", {})
        is_dragging = s_next.get("is_dragging", False)
        is_routing = s_next.get("is_routing", False)
        if is_dragging:
            states.append("DRAGGING")
        elif is_routing:
            states.append("ROUTING")
        else:
            states.append("IDLE")

    state_colors = {"IDLE": "#999999", "ROUTING": "#33aa33", "DRAGGING": "#ff8800"}
    fig, ax = plt.subplots(figsize=(10, 2))
    for i, st in enumerate(states):
        ax.barh(0, 1, left=steps[i] - 0.5, color=state_colors.get(st, "#999999"), height=0.6)
    ax.set_xlabel("Step")
    ax.set_yticks([])
    ax.set_title("Router State Timeline")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=s) for s, c in state_colors.items()]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    path = os.path.join(output_dir, "router_state_timeline.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    outputs["router_state_timeline"] = path

    # ---- Chart 4: DRC violation trend ----
    drc_counts = []
    has_drc = False
    for s in sidecars:
        obs = s.get("obs_summary")
        if obs is not None and "drc_count" in obs:
            drc_counts.append(float(obs["drc_count"]))
            has_drc = True
        else:
            drc_counts.append(None)

    if has_drc:
        fig, ax = plt.subplots(figsize=(10, 3))
        valid_steps = [st for st, d in zip(steps, drc_counts) if d is not None]
        valid_counts = [d for d in drc_counts if d is not None]
        ax.plot(valid_steps, valid_counts, "r-o", linewidth=1.5, markersize=4)
        ax.fill_between(valid_steps, valid_counts, alpha=0.2, color="red")
        ax.set_xlabel("Step")
        ax.set_ylabel("DRC Count")
        ax.set_title("DRC Violation Trend")
        ax.grid(True, alpha=0.3)
        path = os.path.join(output_dir, "drc_trend.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        outputs["drc_trend"] = path

    # ---- Chart 5: Spatial action trajectory ----
    action_xs, action_ys, action_steps = [], [], []
    for i, s in enumerate(sidecars):
        a_t = s.get("transition", {}).get("a_t", {})
        x = a_t.get("x_mm")
        y = a_t.get("y_mm")
        if x is not None and y is not None and (abs(x) > 1e-9 or abs(y) > 1e-9):
            action_xs.append(float(x))
            action_ys.append(float(y))
            action_steps.append(steps[i])

    if action_xs:
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(action_xs, action_ys, c=action_steps, cmap="viridis",
                        s=30, edgecolors="black", linewidth=0.5, zorder=5)
        ax.plot(action_xs, action_ys, "k-", alpha=0.3, linewidth=0.8, zorder=1)
        plt.colorbar(sc, ax=ax, label="Step")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Action Spatial Trajectory")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        path = os.path.join(output_dir, "spatial_trajectory.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        outputs["spatial_trajectory"] = path

    # ---- Chart 6: Board state metrics ----
    track_counts = []
    unrouted_counts = []
    for s in sidecars:
        s_next = s.get("transition", {}).get("s_t_plus_1", {})
        track_counts.append(s_next.get("track_count", 0))
        unrouted_counts.append(s_next.get("unrouted_count", 0))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, track_counts, "b-o", linewidth=1.5, markersize=4, label="Tracks")
    ax.plot(steps, unrouted_counts, "r-s", linewidth=1.5, markersize=4, label="Unrouted")
    ax.set_xlabel("Step")
    ax.set_ylabel("Count")
    ax.set_title("Board State Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "board_metrics.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    outputs["board_metrics"] = path

    # ---- HTML index ----
    _generate_html_index(output_dir, outputs, sidecars, has_drc)

    return outputs


def _generate_html_index(
    output_dir: str,
    chart_paths: dict[str, str],
    sidecars: list[dict],
    has_drc: bool,
) -> str:
    """Generate an HTML page embedding all dashboard charts."""
    html_path = os.path.join(output_dir, "index.html")

    chart_sections = []
    chart_order = [
        ("reward_value_curve", "Reward & Value Curve"),
        ("action_mask_evolution", "Action Mask Evolution"),
        ("router_state_timeline", "Router State Timeline"),
        ("drc_trend", "DRC Violation Trend"),
        ("spatial_trajectory", "Spatial Action Trajectory"),
        ("board_metrics", "Board State Metrics"),
    ]

    for key, title in chart_order:
        if key in chart_paths:
            fname = os.path.basename(chart_paths[key])
            chart_sections.append(
                f'<div class="chart"><h2>{title}</h2>'
                f'<img src="{fname}" alt="{title}"></div>'
            )
        elif key == "drc_trend" and not has_drc:
            chart_sections.append(
                f'<div class="chart"><h2>{title}</h2>'
                f'<p class="note">DRC data not available in sidecars.</p></div>'
            )

    ep_len = len(sidecars)
    total_r = sum(s.get("transition", {}).get("r_t", 0.0) for s in sidecars)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Episode Dashboard</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 1100px;
         margin: 0 auto; padding: 20px; background: #fafafa; }}
  h1 {{ color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
  .summary {{ background: #fff; padding: 15px; border-radius: 8px;
              box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }}
  .chart {{ background: #fff; padding: 15px; border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }}
  .chart img {{ max-width: 100%; height: auto; }}
  .chart h2 {{ color: #555; font-size: 1.1em; margin-top: 0; }}
  .note {{ color: #999; font-style: italic; }}
</style>
</head>
<body>
<h1>Episode Dashboard</h1>
<div class="summary">
  <strong>Episode Length:</strong> {ep_len} steps &nbsp;|&nbsp;
  <strong>Total Reward:</strong> {total_r:.4f}
</div>
{''.join(chart_sections)}
</body>
</html>"""

    with open(html_path, "w") as f:
        f.write(html)

    return html_path
