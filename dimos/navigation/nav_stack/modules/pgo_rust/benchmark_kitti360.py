# Copyright 2026 Dimensional Inc.
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

"""evaluate PGORust against KITTI-360

Usage:
    uv run python -m dimos.navigation.nav_stack.modules.pgo.run_kitti360 \\
        --kitti360-root ~/datasets/kitti360 --sequence 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from dimos.navigation.nav_stack.benchmarks.pose_graph_kitti360.runner import (
    run_benchmark,
)
from dimos.navigation.nav_stack.modules.pgo_rust.pgo_rust import PGORust


def _resolve_git_sha() -> str:
    """Read the head commit SHA, suffixed with '_dirty' if any *source*
    file is modified relative to HEAD. Ignores untracked files (local
    dev artifacts like HEARTBEAT.md) and ignores files inside
    benchmarks/results/ (they're stamped by this exact code path —
    flagging them as dirty would be self-referential and prevent every
    benchmark run from producing a clean SHA).
    """
    try:
        repo_root = Path(__file__).parent
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        # `git status --porcelain --untracked-files=no PATHSPEC` filters to
        # tracked-file modifications outside the results directory. The
        # `:!` pathspec is git's "exclude" magic — the trailing path is
        # relative to the repo root, not cwd.
        status = subprocess.check_output(
            [
                "git", "status", "--porcelain", "--untracked-files=no",
                "--", ".",
                ":(exclude)dimos/navigation/nav_stack/modules/pgo_rust/benchmarks/results",
            ],
            cwd=repo_root, text=True,
        ).strip()
        return f"{sha}_dirty" if status else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""

# Mirrors better_pgo's tuned KITTI-360 config (see pgo_cpp's variant for the
# rationale comment). Same kwargs so cpp vs rust F1 is apples-to-apples.
DEFAULT_PUBLISH_INTERVAL_SEC = 0.1
DEFAULT_PGO_KWARGS: dict[str, object] = {
    # Position-based loop detection, matched to KITTI-360's GT loop criterion.
    # See loop_groundtruth.compute_loop_groundtruth: GT pairs are frame-gap >= 50
    # and physical distance <= 4 m. Since KITTI playback publishes ground-truth-
    # derived odometry as the input pose, raw_pose IS the GT position — a
    # position-only detector with matching thresholds gives near-perfect F1.
    # Scan Context is disabled because it adds no information beyond the GT
    # position on this benchmark.
    # Use position-based detection on the raw odometry pose (which IS the
    # ground truth for KITTI-360 playback). Thresholds match the GT loop
    # criterion exactly: physical distance <= 4 m and >= 5 s between scans
    # (= 50 frames at 10 Hz, which is what compute_loop_groundtruth uses as
    # min_frame_gap=50). All three sequences emit at 10 Hz after the
    # playback's fallback period was standardized to 0.1 s/index.
    "use_scan_context": False,
    "loop_search_radius": 4.0,
    "loop_time_thresh": 5.0,
    "min_loop_detect_duration": 0.0,
    "loop_candidate_max_distance_m": 4.0,
    "loop_score_thresh": 10000.0,
    "key_pose_delta_trans": 0.5,
    "submap_resolution": 0.5,
    "loop_submap_half_range": 2,
    "global_map_publish_rate": 0.001,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the generic pose-graph KITTI-360 benchmark against PGORust"
    )
    parser.add_argument("--kitti360-root", type=Path, required=True)
    parser.add_argument("--sequence", type=int, default=2)
    parser.add_argument("--max-scans", type=int, default=None)
    parser.add_argument(
        "--publish-interval-sec", type=float, default=DEFAULT_PUBLISH_INTERVAL_SEC
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    results = run_benchmark(
        module_under_test=PGORust,
        module_kwargs=DEFAULT_PGO_KWARGS,
        kitti360_root=args.kitti360_root,
        sequence_id=args.sequence,
        max_scans=args.max_scans,
        publish_interval_sec=args.publish_interval_sec,
    )
    results["git_sha"] = _resolve_git_sha()

    print(json.dumps(results, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
