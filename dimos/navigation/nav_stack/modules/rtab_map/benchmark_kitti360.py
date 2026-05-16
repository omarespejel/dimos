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

"""evaluate RtabMap against KITTI-360

Usage:
    uv run python -m dimos.navigation.nav_stack.modules.rtab_map.benchmark_kitti360 \\
        --kitti360-root ~/datasets/kitti360 --sequence 2

The pose-graph KITTI-360 benchmark scaffolding (playback + scoring) lives in
``dimos/navigation/nav_stack/benchmarks/pose_graph_kitti360/`` and consumes
any blueprint that exposes the pose-graph wire contract:

  in:  registered_scan: In[PointCloud2], odometry: In[Odometry]
  out: pose_graph_edges: Out[NavPath], loop_closure: Out[NavPath]

RtabMap publishes both via the C++ binary (see ``cpp/main.cpp``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dimos.navigation.nav_stack.benchmarks.pose_graph_kitti360.runner import (
    run_benchmark,
)
from dimos.navigation.nav_stack.modules.rtab_map.rtab_map import RtabMap


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the pose-graph KITTI-360 benchmark against RtabMap"
    )
    parser.add_argument("--kitti360-root", type=Path, required=True)
    parser.add_argument("--sequence", type=int, default=2)
    parser.add_argument("--max-scans", type=int, default=None)

    # RtabMap-side knobs that materially affect loop-closure recall/precision.
    # Defaults track the wrapper's defaults; override here when benchmarking.
    parser.add_argument(
        "--rtabmap-process-period",
        type=float,
        default=0.0,
        help="seconds between keyframes (0 = admit every scan; use 0 for tests)",
    )
    parser.add_argument(
        "--rgbd-proximity-path-max-neighbors",
        type=int,
        default=10,
        help="rtabmap one-to-many proximity detection neighbor count",
    )
    parser.add_argument(
        "--grid-cell-size",
        type=float,
        default=0.1,
        help="OctoMap cell size in meters",
    )
    parser.add_argument("--debug", action="store_true", help="verbose C++ stderr")
    parser.add_argument(
        "--loop-min-id-gap",
        type=int,
        default=50,
        help="wrapper-side filter: drop detected loops whose signature-id gap "
        "is below this (default 50, matching KITTI-360 GT min frame gap)",
    )

    # Benchmark-runner knobs.
    parser.add_argument("--publish-interval-sec", type=float, default=0.02)
    parser.add_argument("--drain-sec", type=float, default=10.0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    rtab_blueprint = RtabMap.blueprint(
        rtabmap_process_period=args.rtabmap_process_period,
        rgbd_proximity_path_max_neighbors=args.rgbd_proximity_path_max_neighbors,
        grid_cell_size=args.grid_cell_size,
        loop_min_id_gap=args.loop_min_id_gap,
        debug=args.debug,
    )

    results = run_benchmark(
        module_under_test=rtab_blueprint,
        kitti360_root=args.kitti360_root,
        sequence_id=args.sequence,
        max_scans=args.max_scans,
        publish_interval_sec=args.publish_interval_sec,
        drain_sec=args.drain_sec,
    )

    print(json.dumps(results, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
