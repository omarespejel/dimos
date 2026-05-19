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

"""Native Rust PGO module — Rust port of the C++ PGO via GTSAM cxx FFI + KISS-ICP.

Parallel to ``pgo_cpp.PGOCpp``: same LoopClosure protocol, same outputs.
nav_stack picks one of the two via the ``loop_closure`` parameter.
"""

from __future__ import annotations

from pathlib import Path

from dimos.core.core import rpc
from dimos.core.native_module import NativeModule, NativeModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.nav_msgs.Graph3D import Graph3D
from dimos.msgs.nav_msgs.GraphDelta3D import GraphDelta3D
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.navigation.nav_stack.specs import LoopClosure
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class PGORustConfig(NativeModuleConfig):
    cwd: str | None = str(Path(__file__).resolve().parent / "rust")
    executable: str = "target/release/pgo_rust"
    # Uses `nix develop --command` so the cxx FFI build sees the gtsam +
    # cephes-gtsam + tbb + eigen include / lib paths from the per-module
    # flake. A bare `cargo build` fails outside nix (missing gtsam headers).
    build_command: str | None = "nix develop --command cargo build --release"
    stdin_config: bool = True
    ready_timeout_sec: float = 10.0

    frame_id: str = "map"
    child_frame_id: str = "start_point"
    parent_frame: str = "world"
    body_frame: str = "current_point"
    tf_channel: str = "/tf#tf2_msgs.TFMessage"

    key_pose_delta_deg: float = 10.0
    key_pose_delta_trans: float = 0.5

    loop_search_radius: float = 1.0
    loop_time_thresh: float = 60.0
    loop_score_thresh: float = 0.15
    loop_submap_half_range: int = 5
    submap_resolution: float = 0.1
    min_loop_detect_duration: float = 5.0

    unregister_input: bool = True

    global_map_voxel_size: float = 0.1
    global_map_publish_rate: float = 1.0

    use_scan_context: bool = True
    scan_context_num_rings: int = 20
    scan_context_num_sectors: int = 60
    scan_context_max_range_m: float = 80.0
    scan_context_top_k: int = 10
    scan_context_match_threshold: float = 0.4
    scan_context_lidar_height_m: float = 2.0

    debug: bool = False


class PGORust(NativeModule, LoopClosure):
    """Pose graph optimization with loop closure using GTSAM iSAM2 (FFI) + KISS-ICP."""

    config: PGORustConfig

    registered_scan: In[PointCloud2]
    odometry: In[Odometry]
    corrected_odometry: Out[Odometry]
    global_map: Out[PointCloud2]
    pose_graph: Out[Graph3D]
    loop_closure_event: Out[GraphDelta3D]

    @rpc
    def start(self) -> None:
        super().start()
        if self.config.debug:
            logger.info("PGO native module started (Rust + GTSAM FFI + KISS-ICP)")

    @rpc
    def stop(self) -> None:
        super().stop()
