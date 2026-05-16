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

"""Map a diagnostic session's classification breakdown to a phase-2 action.

Reads a session produced by the trajectory diagnostic, re-runs the
classifier (no plotting), aggregates per-trial labels, and prints a
short text recommendation. This is a *hint*, not a verdict — the owner
reviews the report and decides what to fund.

Usage::

    python -m dimos.utils.characterization.processing.recommend <session_dir>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import NamedTuple

from dimos.utils.benchmarking.scoring import score_run_with_trajectory
from dimos.utils.characterization.processing.diagnose import (
    TrialDiagnosis,
    build_executed_trajectory,
    classify,
    load_session,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------


class Action(NamedTuple):
    classification: str
    summary: str
    estimate: str
    note: str


_ACTIONS: dict[str, Action] = {
    "clean": Action(
        classification="clean",
        summary="No controls work warranted — tracking is at the plant's physical floor.",
        estimate="none",
        note=(
            "Along-track lag is at or under the FOPDT floor (tau+L)*v — the plant "
            "plus a trivial low-gain controller is already tracking as well as the "
            "physics allows. No saturation, estimation noise, jitter, or odom bias "
            "fired. Building MPC or retuning controllers here would chase a "
            "bottleneck that does not exist. If the original controller benchmark "
            "still shows controllers ≈ baseline, the cause is upstream "
            "(e.g. a mis-calibrated feedforward gain), not controller architecture — "
            "re-benchmark with the corrected plant_models before any controls work."
        ),
    ),
    "saturation": Action(
        classification="saturation",
        summary="Engage the existing velocity profiler.",
        estimate="~1 day",
        note=(
            "Path is `dimos/control/tasks/velocity_profiler.py` — already wired into "
            "`path_follower_task.py`. Tune the curvature-speed cap and accel limits so "
            "the commanded velocity profile stays inside the saturation envelope. No new "
            "controller architecture needed."
        ),
    ),
    "deadtime_lag": Action(
        classification="deadtime_lag",
        summary="MPC with FOPDT prediction model.",
        estimate="3-4 weeks",
        note=(
            "This is the canonical case for model predictive control: the controller "
            "needs to act on a *predicted future* pose, not current. Linear MPC with "
            "the FOPDT plant as prediction model is the cleanest path. Vendored plant "
            "params live in `dimos/utils/benchmarking/plant_models.py` — confirm they're "
            "still current before building."
        ),
    ),
    "estimation_noise": Action(
        classification="estimation_noise",
        summary="State-estimation work — separate ticket.",
        estimate="not a controls ticket",
        note=(
            "The pre-roll cross-track noise floor exceeds active-window cross-track "
            "RMS — the estimator is the binding limit. No controller change helps. "
            "Investigate odom source (filter, frequency, fusion) before any controller "
            "engineering."
        ),
    ),
    "odom_bias": Action(
        classification="odom_bias",
        summary="Fix the estimator — separate ticket.",
        estimate="not a controls ticket",
        note=(
            "Pre-roll (zero command) shows secular pose drift — the estimator has bias. "
            "All controller scores are contaminated by this drift. Fix the estimator "
            "first; rerun the diagnostic."
        ),
    ),
    "jitter": Action(
        classification="jitter",
        summary="Document as a comms ceiling; no controller fix.",
        estimate="n/a",
        note=(
            "Intersample dt std exceeds the threshold — commands aren't arriving at the "
            "robot at a uniform cadence. Likely WebRTC delivery jitter. Without "
            "firmware access there's no controller-side fix. Document the ceiling."
        ),
    ),
}


# ---------------------------------------------------------------------------


def aggregate_classifications(diagnoses: list[TrialDiagnosis]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in diagnoses:
        counts[d.classification] = counts.get(d.classification, 0) + 1
    return counts


def recommend(session_dir: Path) -> str:
    """Return a multi-line recommendation string for ``session_dir``."""
    session_dir = Path(session_dir).expanduser().resolve()
    runs = load_session(session_dir)
    if not runs:
        return f"no trajectory runs found in {session_dir}"

    diagnoses: list[TrialDiagnosis] = []
    for run in runs:
        executed = build_executed_trajectory(run)
        score = score_run_with_trajectory(executed, run.anchored_ref_fn, duration_s=run.duration_s)
        diagnoses.append(classify(run, score))

    counts = aggregate_classifications(diagnoses)
    total = len(diagnoses)
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])

    lines: list[str] = []
    lines.append(f"diagnostic session: {session_dir}")
    lines.append(f"trials analyzed:    {total}")
    lines.append("")
    lines.append("classification breakdown:")
    for name, n in sorted_counts:
        lines.append(f"  {name:<20} {n}/{total} trials")
    lines.append("")

    dominant, dominant_n = sorted_counts[0]
    is_multi_regime = sum(1 for _, n in sorted_counts if n > 0) > 1 and (dominant_n < total)

    if is_multi_regime:
        lines.append("MULTIPLE REGIMES detected — no single phase-2 action covers everything.")
        lines.append("Scope each regime separately. Per-regime recommendation:")
        lines.append("")
        for name, n in sorted_counts:
            action = _ACTIONS.get(name)
            if action is None:
                continue
            lines.append(f"- **{name}** ({n}/{total}): {action.summary} ({action.estimate})")
            lines.append(f"    {action.note}")
            lines.append("")
        lines.append("")
        lines.append(
            "The owner should weigh which regime is the binding bottleneck for the "
            "operating points that matter. Trial-by-trial in the diagnose report:"
        )
        for d in diagnoses:
            lines.append(
                f"  {d.run.run_id:<28} → {d.classification:<20} (mode={d.run.controller_mode})"
            )
    else:
        action = _ACTIONS.get(dominant)
        if action is None:
            lines.append(f"unrecognized classification: {dominant!r}")
        else:
            lines.append(f"DOMINANT REGIME: {dominant} ({dominant_n}/{total} trials)")
            lines.append("")
            lines.append(f"  → {action.summary}")
            lines.append(f"  → estimate: {action.estimate}")
            lines.append("")
            lines.append(action.note)

    lines.append("")
    lines.append(
        "This is a hint, not a verdict. The owner reviews the diagnose report and "
        "decides whether to fund the recommended action."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Map a diagnostic session's classification breakdown to a phase-2 action."
    )
    parser.add_argument("session_dir", type=Path, help="Path to a session_* directory.")
    args = parser.parse_args()
    print(recommend(args.session_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
