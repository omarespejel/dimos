# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run path-follower controllers against the path battery, in sim or on hardware.

Two modes (selected with `--mode`):

* `--mode sim` (default): sweep the full cohort matrix against the full path
  battery (10 cohorts x N paths). Produces per-cohort scored JSON, per-path
  trajectory SVGs, and a single index.html aggregating all composites.

* `--mode hw`: run a single controller (selected via `--controller`) with
  optional `--pi` / `--ff` toggles. Operator gets prompted to park the robot
  at each path's start. Onboard Go2 odom drifts on long/curvy paths so the
  recommended hardware battery is short paths only (straight_5m,
  single_corner, circle_R0.5).

Examples:
    # Full sim sweep
    python -m dimos.utils.benchmarking.run_battery --mode sim

    # One hardware run
    Terminal A: dimos run unitree-go2-coordinator
    Terminal B: python -m dimos.utils.benchmarking.run_battery --mode hw \\
                  --controller baseline --k_angular 1.0 --path straight_5m
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dimos.msgs.nav_msgs.Path import Path as NavPath

from dimos.control.tasks.feedforward_gain_compensator import FeedforwardGainConfig
from dimos.control.tasks.velocity_tracking_pid import (
    VelocityPIDConfig,
    VelocityTrackingConfig,
)
from dimos.utils.benchmarking.paths import (
    default_battery,
    multi_trajectory_to_svg,
    trajectory_to_svg,
)
from dimos.utils.benchmarking.runner import (
    HwRunOptions,
    run_baseline_hw,
    run_baseline_sim,
    run_lyapunov_hw,
    run_lyapunov_sim,
    run_mpc_hw,
    run_pure_pursuit_hw,
    run_pure_pursuit_sim,
    run_rpp_hw,
    run_rpp_sim,
    run_rpp_tuned_hw,
    run_setpoint_hw,
    run_setpoint_sim,
)
from dimos.utils.benchmarking.scoring import ExecutedTrajectory, score_run

# Session 3 wz PI gains, from
# ~/char_runs/session_20260425-131525/modeling/tuning/tuning_summary.json
SESSION3_WZ_PID = VelocityPIDConfig(
    kp=0.346,
    ki=1.343,
    kd=0.0,
    max_integral=0.5,
    output_min=-1.5,
    output_max=1.5,
)
PASSTHROUGH_PID = VelocityPIDConfig(kp=0.0, ki=0.0)
SESSION3_VELOCITY_TRACKING = VelocityTrackingConfig(
    vx=PASSTHROUGH_PID,
    vy=PASSTHROUGH_PID,
    wz=SESSION3_WZ_PID,
    dt=0.1,
)

# Strategy B: static plant-gain feedforward. Derived from the single
# source of truth (dimos.utils.benchmarking.plant_models) so it can never
# go stale again — a hardcoded copy here (K_wz=2.175) silently diverged
# from the vendored fit (2.453) and made every FF controller compensate
# with the wrong gain. Read it; don't copy it.
from dimos.utils.benchmarking.plant_models import GO2_PLANT_FITTED as _GO2_PLANT

GO2_FEEDFORWARD = FeedforwardGainConfig(
    K_vx=_GO2_PLANT.vx.K,
    K_vy=_GO2_PLANT.vy.K,
    K_wz=_GO2_PLANT.wz.K,
)


# Sim cohort matrix: each entry produces a callable (path, timeout) -> ExecutedTrajectory.
# Slim 4 + 1 optional set: production anchor, the three FF controllers, and
# pure_pursuit (no FF) kept as the FF-vs-no-FF demo.
SIM_COHORTS: dict[str, Callable[[Path, float], ExecutedTrajectory]] = {
    "baseline_k0.5": lambda p, t: run_baseline_sim(p, timeout_s=t, k_angular=0.5),
    "pure_pursuit": lambda p, t: run_pure_pursuit_sim(p, timeout_s=t),
    "pure_pursuit_ff": lambda p, t: run_pure_pursuit_sim(p, timeout_s=t, ff_config=GO2_FEEDFORWARD),
    "rpp_ff": lambda p, t: run_rpp_sim(p, timeout_s=t, ff_config=GO2_FEEDFORWARD),
    "lyapunov_ff": lambda p, t: run_lyapunov_sim(p, timeout_s=t, ff_config=GO2_FEEDFORWARD),
    "setpoint": lambda p, t: run_setpoint_sim(p, timeout_s=t),
}

# Hw controller dispatch: maps `--controller` value to its runner factory.
HW_RUNNERS: dict[str, Callable] = {
    "baseline": run_baseline_hw,
    "lyapunov": run_lyapunov_hw,
    "pure_pursuit": run_pure_pursuit_hw,
    "rpp": run_rpp_hw,
    "rpp_tuned": run_rpp_tuned_hw,
    "mpc": run_mpc_hw,
    "setpoint": run_setpoint_hw,
}


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# --- Sim mode entry point ---


def _run_sim(args: argparse.Namespace) -> None:
    """Run the cohort matrix in-process, persist trajectories + scores, render."""
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    sha = _git_short_sha()

    battery = default_battery()
    all_results: dict[str, dict[str, dict]] = {}
    per_path_executed: dict[str, dict[str, list[tuple[float, float]]]] = {
        name: {} for name in battery
    }

    for cohort_name, runner_fn in SIM_COHORTS.items():
        print(f"\n{'#' * 60}\n# Cohort: {cohort_name}\n{'#' * 60}")
        cohort_dir = out / cohort_name
        cohort_dir.mkdir(parents=True, exist_ok=True)
        cohort_results: dict[str, dict] = {}

        for name, path in battery.items():
            print(f"\n=== {name} ({len(path.poses)} poses) ===")
            traj = runner_fn(path, args.timeout)
            score = score_run(path, traj)
            last = traj.ticks[-1] if traj.ticks else None
            tail = (
                f"  arrived={score.arrived}  ticks={score.n_ticks}  t={score.time_to_complete:.2f}s"
            )
            if last is not None:
                tail += f"  last=({last.pose.position.x:.2f},{last.pose.position.y:.2f})"
            print(tail)
            print(
                f"  CTE rms={score.cte_rms * 100:.1f}cm max={score.cte_max * 100:.1f}cm  "
                f"head_err rms={score.heading_err_rms:.3f}rad  "
                f"v_lin_rms={score.linear_speed_rms:.2f}m/s  "
                f"smoothness(Sum|dcmd|)={score.cmd_rate_integral:.2f}"
            )
            cohort_results[name] = asdict(score)
            executed_xy = [(t.pose.position.x, t.pose.position.y) for t in traj.ticks]
            # Persist the executed trajectory next to the SVG so plots can be
            # regenerated later via `--mode replot` without re-simulating.
            (cohort_dir / f"{name}_traj.json").write_text(json.dumps(executed_xy))
            per_path_executed[name][cohort_name] = executed_xy

        all_results[cohort_name] = cohort_results

    # Persist scores BEFORE rendering so a render failure doesn't lose them.
    summary_path = out / f"sim_baseline_{sha}.json"
    summary_path.write_text(json.dumps(all_results, indent=2))

    _render_sim_artifacts(
        out, sha, battery, list(SIM_COHORTS.keys()), all_results, per_path_executed
    )
    print(f"\nWrote {summary_path}")
    print(f"Wrote composite SVGs to {out}/composite/")
    print(f"Open in browser: file://{out.resolve()}/index.html")

    # Comparison table
    print(f"\n{'=' * 90}\nHead-to-head: cte_rms (cm)\n{'=' * 90}")
    cohort_names = list(SIM_COHORTS.keys())
    print(f"{'path':<22}  " + "  ".join(f"{c:>20}" for c in cohort_names))
    for path_name in battery:
        row = [path_name]
        for c in cohort_names:
            r = all_results[c][path_name]
            row.append(f"{r['cte_rms'] * 100:5.1f}cm/{r['time_to_complete']:5.1f}s")
        print(f"{row[0]:<22}  " + "  ".join(f"{r:>20}" for r in row[1:]))


def _render_sim_artifacts(
    out: Path,
    sha: str,
    battery: dict[str, NavPath],
    cohort_names: list[str],
    all_results: dict[str, dict[str, dict]],
    per_path_executed: dict[str, dict[str, list[tuple[float, float]]]],
) -> None:
    """Write per-cohort SVGs, composite SVGs, and index.html from in-memory data.

    Pulled out of `_run_sim` so `_run_replot` can call the same logic with data
    loaded from disk instead of from a fresh simulation.
    """
    # Per-cohort, per-path SVGs.
    for cohort_name in cohort_names:
        cohort_dir = out / cohort_name
        cohort_dir.mkdir(parents=True, exist_ok=True)
        for path_name, path in battery.items():
            executed_xy = per_path_executed.get(path_name, {}).get(cohort_name)
            if executed_xy is None:
                continue
            (cohort_dir / f"{path_name}.svg").write_text(trajectory_to_svg(path, executed_xy))

    # Composite: one SVG per path with all available cohorts overlaid.
    composite_dir = out / "composite"
    composite_dir.mkdir(parents=True, exist_ok=True)
    for path_name, path in battery.items():
        svg = multi_trajectory_to_svg(
            path,
            per_path_executed.get(path_name, {}),
            size_px=600,
            title=path_name,
        )
        (composite_dir / f"{path_name}.svg").write_text(svg)

    # Single index.html aggregating all composites in a grid.
    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Sim baseline {sha}</title>",
        "<style>body{font-family:monospace;margin:20px;background:#fafafa}"
        "h1{margin:8px 0}h2{margin:24px 0 8px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(620px,1fr));gap:16px}"
        ".card{background:white;border:1px solid #ddd;padding:8px}"
        "table{border-collapse:collapse;font-size:12px}"
        "th,td{border:1px solid #ccc;padding:4px 8px;text-align:right}"
        "th:first-child,td:first-child{text-align:left}"
        "</style></head><body>",
        f"<h1>Sim baseline (commit {sha})</h1>",
        "<h2>Head-to-head: cte_rms (cm)</h2>",
        "<table><thead><tr><th>path</th>"
        + "".join(f"<th>{c}</th>" for c in cohort_names)
        + "</tr></thead><tbody>",
    ]
    for path_name in battery:
        html_parts.append(f"<tr><td>{path_name}</td>")
        for c in cohort_names:
            r = (all_results.get(c) or {}).get(path_name) or {}
            cte = r.get("cte_rms")
            ttc = r.get("time_to_complete")
            if cte is None or ttc is None:
                html_parts.append("<td>-</td>")
            else:
                html_parts.append(f"<td>{cte * 100:.1f} / {ttc:.1f}s</td>")
        html_parts.append("</tr>")
    html_parts.append("</tbody></table>")

    html_parts.append("<h2>Trajectories (per path, all cohorts)</h2>")
    html_parts.append("<div class='grid'>")
    for path_name in battery:
        html_parts.append(
            f"<div class='card'><img src='composite/{path_name}.svg' "
            f"width='600' alt='{path_name}'/></div>"
        )
    html_parts.append("</div></body></html>")

    (out / "index.html").write_text("\n".join(html_parts))


# --- Replot mode entry point ---


def _run_replot(args: argparse.Namespace) -> None:
    """Re-render every SVG and the index.html from existing on-disk data.

    Reads `sim_baseline_<sha>.json` (most recent if multiple) and each
    `<cohort>/<path>_traj.json` produced by a prior `--mode sim` run, and
    regenerates trajectory_to_svg + multi_trajectory_to_svg + index.html.
    No simulation, no scoring - typically completes in seconds.
    """
    out: Path = args.out
    if not out.is_dir():
        raise SystemExit(f"--out {out} does not exist; nothing to replot.")

    # Pick the most recent sim_baseline_<sha>.json in the dir.
    summaries = sorted(out.glob("sim_baseline_*.json"), key=lambda p: p.stat().st_mtime)
    if not summaries:
        raise SystemExit(f"No sim_baseline_*.json found in {out}. Run `--mode sim` first.")
    summary_path = summaries[-1]
    sha = summary_path.stem.removeprefix("sim_baseline_")
    all_results = json.loads(summary_path.read_text())
    print(f"Replotting from {summary_path}")

    # Load executed trajectories from per-cohort traj.json files.
    battery = default_battery()
    cohort_names = list(all_results.keys())
    per_path_executed: dict[str, dict[str, list[tuple[float, float]]]] = {
        name: {} for name in battery
    }
    n_loaded = 0
    n_missing = 0
    for cohort_name in cohort_names:
        cohort_dir = out / cohort_name
        if not cohort_dir.is_dir():
            n_missing += len(battery)
            print(f"  [skip] {cohort_name}: dir missing")
            continue
        for path_name in battery:
            traj_path = cohort_dir / f"{path_name}_traj.json"
            if not traj_path.exists():
                n_missing += 1
                continue
            try:
                xy = json.loads(traj_path.read_text())
                per_path_executed[path_name][cohort_name] = [(float(x), float(y)) for x, y in xy]
                n_loaded += 1
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                print(f"  [skip] {cohort_name}/{path_name}: {type(e).__name__}: {e}")
                n_missing += 1
    print(f"Loaded {n_loaded} trajectories ({n_missing} missing/skipped)")

    if n_loaded == 0:
        raise SystemExit(
            f"No traj.json files found under {out}/<cohort>/<path>_traj.json. "
            "Older sim runs (before the replot feature) didn't persist trajectories - "
            "you'll need to re-run `--mode sim` once to populate them."
        )

    _render_sim_artifacts(out, sha, battery, cohort_names, all_results, per_path_executed)
    print(f"Re-rendered SVGs + index.html in {out}")
    print(f"Open in browser: file://{out.resolve()}/index.html")


# --- Hardware mode entry point ---


def _run_hw(args: argparse.Namespace) -> None:
    if args.pi and args.ff:
        raise SystemExit("--pi and --ff are mutually exclusive")
    if not args.controller:
        raise SystemExit("--mode hw requires --controller")

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    sha = _git_short_sha()

    battery = default_battery()
    if args.path is not None:
        if args.path not in battery:
            raise SystemExit(f"Unknown path {args.path!r}; choices: {sorted(battery)}")
        battery = {args.path: battery[args.path]}

    opts = HwRunOptions(
        timeout_s=args.timeout,
        speed=args.speed,
        k_angular=args.k_angular,
        pid_config=SESSION3_VELOCITY_TRACKING if args.pi else None,
        ff_config=GO2_FEEDFORWARD if args.ff else None,
    )

    cohort_label = (
        f"hw_{args.controller}"
        f"{'_k' + str(args.k_angular) if args.controller == 'baseline' else ''}"
        f"{'_pi' if args.pi else ''}"
        f"{'_ff' if args.ff else ''}"
    )
    runner_fn = HW_RUNNERS[args.controller]
    cohort_dir = out / cohort_label
    cohort_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for name, path in battery.items():
        print(f"\n{'#' * 60}\n# {cohort_label} :: {name} ({len(path.poses)} poses)\n{'#' * 60}")
        path_used, traj = runner_fn(path, opts)
        score = score_run(path_used, traj)
        last = traj.ticks[-1] if traj.ticks else None
        tail = f"  arrived={score.arrived}  ticks={score.n_ticks}  t={score.time_to_complete:.2f}s"
        if last is not None:
            tail += f"  last=({last.pose.position.x:.2f},{last.pose.position.y:.2f})"
        print(tail)
        print(
            f"  CTE rms={score.cte_rms * 100:.1f}cm max={score.cte_max * 100:.1f}cm  "
            f"head_err rms={score.heading_err_rms:.3f}rad  "
            f"v_lin_rms={score.linear_speed_rms:.2f}m/s"
        )
        results[name] = asdict(score)
        executed_xy = [(t.pose.position.x, t.pose.position.y) for t in traj.ticks]
        (cohort_dir / f"{name}.svg").write_text(trajectory_to_svg(path_used, executed_xy))

        # Raw per-tick dump for offline alpha-tuning / cmd-vs-actual analysis.
        ticks_dump = [
            {
                "t": tk.t,
                "x": tk.pose.position.x,
                "y": tk.pose.position.y,
                "yaw": tk.pose.orientation.euler[2],
                "cmd_vx": tk.cmd_twist.linear.x,
                "cmd_wz": tk.cmd_twist.angular.z,
                "est_vx": tk.actual_twist.linear.x,
                "est_vy": tk.actual_twist.linear.y,
                "est_wz": tk.actual_twist.angular.z,
            }
            for tk in traj.ticks
        ]
        (cohort_dir / f"{name}_ticks.json").write_text(json.dumps(ticks_dump))

    summary = out / f"hw_baseline_{sha}_{cohort_label}.json"
    summary.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {summary}")
    print(f"Per-path SVGs in {cohort_dir}/")


# --- CLI ---


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["sim", "hw", "replot"],
        default="sim",
        help=(
            "sim: sweep the full cohort matrix; "
            "hw: run one controller; "
            "replot: regenerate SVGs + index.html from existing on-disk data "
            "(no simulation). Default: sim."
        ),
    )
    # Common
    parser.add_argument(
        "--out", type=Path, default=None, help="Output dir (default depends on --mode)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-run timeout in seconds (default: 60s sim, 30s hw)",
    )
    # Hw-only
    parser.add_argument(
        "--controller",
        choices=list(HW_RUNNERS.keys()),
        default=None,
        help="(hw mode) Which controller to run.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="(hw mode) Single path name; omit to run full battery.",
    )
    parser.add_argument(
        "--k_angular", type=float, default=0.5, help="(hw mode) Baseline k_angular gain."
    )
    parser.add_argument("--speed", type=float, default=0.55, help="(hw mode) Target linear speed.")
    parser.add_argument(
        "--pi",
        action="store_true",
        help="(hw mode) Enable Session-3 wz PI inner loop (Strategy D).",
    )
    parser.add_argument(
        "--ff",
        action="store_true",
        help="(hw mode) Enable static feedforward plant-gain (Strategy B).",
    )
    args = parser.parse_args()

    # Mode-specific defaults. replot shares sim's output dir since it
    # operates on what sim produced.
    if args.out is None:
        args.out = Path(
            "/tmp/benchmarking_hw" if args.mode == "hw" else "/tmp/benchmarking_baseline"
        )
    if args.timeout is None:
        args.timeout = 60.0 if args.mode == "sim" else 30.0

    if args.mode == "sim":
        _run_sim(args)
    elif args.mode == "hw":
        _run_hw(args)
    else:  # replot
        _run_replot(args)


if __name__ == "__main__":
    main()
