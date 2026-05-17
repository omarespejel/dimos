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

"""ApplyClosure: warp a DynamicCloud global map by a pose-graph correction.

Inputs:
- ``global_map``: the voxel map to warp (DynamicCloud)
- ``previous_pose_graph``: graph poses before optimization (Path of PoseStamped)
- ``next_pose_graph``: graph poses after optimization (same nodes, corrected poses)

For each pose-graph node ``i`` the correction is ``delta_i = next_i @ prev_i^-1``.
Each voxel is bound to the pose-graph timeline by its latest event timestamp
(``per_point_latest_timestamp``), and its warp is a two-nearest-neighbor LBS
blend: lerp on translation, slerp on rotation between the bracketing nodes.

The effect: voxels with recent event timestamps follow the latest pose
corrections, older voxels barely move — matching the way pose-graph drift
accumulates along a trajectory.

Voxels without any event (timestamp 0) clip to the earliest node and get the
smallest correction, which is the conservative choice for "unknown age".
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
from reactivex.disposable import Disposable
from scipy.spatial.transform import Rotation, Slerp

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.nav_msgs.DynamicCloud import DynamicCloud
from dimos.msgs.nav_msgs.Path import Path
from dimos.navigation.nav_stack.frames import FRAME_MAP
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


# Pose-graph node timestamps within this many seconds of each other are
# treated as identical for matching prev/next graphs.
_NODE_TIME_MATCH_TOL = 1e-3
# Slerp requires strictly increasing input times. If two pose-graph nodes
# share a timestamp (degenerate input), bump later ones by this epsilon so
# the interpolator stays well-defined.
_TIME_DEDUP_EPS = 1e-9


def pose_stamped_to_matrix(pose: PoseStamped) -> np.ndarray:
    """Pack a PoseStamped into a 4x4 homogeneous transform."""
    quat = np.array(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(quat))
    if norm == 0.0:
        quat = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        quat = quat / norm
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_quat(quat).as_matrix()
    T[:3, 3] = [pose.x, pose.y, pose.z]
    return T


def path_to_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps[N], transforms[N, 4, 4]) for the path's poses."""
    n = len(path.poses)
    ts = np.empty(n, dtype=np.float64)
    transforms = np.empty((n, 4, 4), dtype=np.float64)
    for i, pose in enumerate(path.poses):
        ts[i] = float(pose.ts)
        transforms[i] = pose_stamped_to_matrix(pose)
    return ts, transforms


def invert_transforms(transforms: np.ndarray) -> np.ndarray:
    """Invert a batch of rigid 4x4 transforms (R^T, -R^T t)."""
    inv = np.empty_like(transforms)
    R = transforms[:, :3, :3]
    t = transforms[:, :3, 3]
    R_t = np.transpose(R, (0, 2, 1))
    inv[:, :3, :3] = R_t
    inv[:, :3, 3] = -np.einsum("nij,nj->ni", R_t, t)
    inv[:, 3, :] = 0.0
    inv[:, 3, 3] = 1.0
    return inv


def compute_node_deltas(prev_T: np.ndarray, next_T: np.ndarray) -> np.ndarray:
    """Per-node correction transforms: delta_i = next_i @ prev_i^-1."""
    return next_T @ invert_transforms(prev_T)


def _dedupe_times(times: np.ndarray) -> np.ndarray:
    """Bump any duplicate timestamps so the sequence is strictly increasing."""
    out = times.astype(np.float64).copy()
    for i in range(1, out.size):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + _TIME_DEDUP_EPS
    return out


def lbs_warp_positions(
    positions: np.ndarray,
    position_times: np.ndarray,
    node_times: np.ndarray,
    node_deltas: np.ndarray,
) -> np.ndarray:
    """Apply two-nearest-neighbor LBS to ``positions`` (M, 3) using ``node_deltas``.

    For each point, find the two pose-graph nodes whose timestamps bracket
    the point's time, slerp the rotations and lerp the translations between
    them by the time-fraction, and apply the blended delta. Points outside
    the node-time range clip to the nearest endpoint.

    Args:
        positions: (M, 3) world-space positions.
        position_times: (M,) per-position timestamps (seconds).
        node_times: (N,) strictly increasing node timestamps.
        node_deltas: (N, 4, 4) correction transforms per node.

    Returns:
        (M, 3) warped positions.
    """
    if positions.shape[0] == 0:
        return positions.astype(np.float64, copy=True)
    if node_times.shape[0] == 0:
        return positions.astype(np.float64, copy=True)
    if node_times.shape[0] == 1:
        delta = node_deltas[0]
        homog = np.concatenate(
            [positions, np.ones((positions.shape[0], 1), dtype=positions.dtype)], axis=1
        )
        return (homog @ delta.T)[:, :3]

    node_times_safe = _dedupe_times(node_times)
    clipped = np.clip(position_times, node_times_safe[0], node_times_safe[-1])

    node_R = Rotation.from_matrix(node_deltas[:, :3, :3])
    slerp = Slerp(node_times_safe, node_R)
    point_R = slerp(clipped)

    node_t = node_deltas[:, :3, 3]
    tx = np.interp(clipped, node_times_safe, node_t[:, 0])
    ty = np.interp(clipped, node_times_safe, node_t[:, 1])
    tz = np.interp(clipped, node_times_safe, node_t[:, 2])
    translation = np.stack([tx, ty, tz], axis=1)

    return point_R.apply(positions.astype(np.float64)) + translation


def merge_duplicate_voxels(
    voxels: np.ndarray,
    quantity: np.ndarray,
    event_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse voxels that share an integer position.

    Two original voxels can warp into the same int32 grid cell. Sum their
    ``quantity``, and remap ``event_indices`` so events still point at the
    surviving merged voxel.

    Returns ``(unique_voxels, merged_quantity, remapped_event_indices)``.
    """
    if voxels.shape[0] == 0:
        return voxels, quantity, event_indices
    unique, inverse = np.unique(voxels, axis=0, return_inverse=True)
    merged_q = np.zeros(unique.shape[0], dtype=np.uint64)
    np.add.at(merged_q, inverse, quantity.astype(np.uint64))
    merged_q = np.minimum(merged_q, np.iinfo(np.uint32).max).astype(np.uint32)
    if event_indices.size == 0:
        new_events = event_indices
    else:
        new_events = inverse[event_indices.astype(np.intp)].astype(np.uint32)
    return unique.astype(np.int32), merged_q, new_events


def apply_closure_to_cloud(
    cloud: DynamicCloud,
    previous_pose_graph: Path,
    next_pose_graph: Path,
) -> DynamicCloud:
    """Warp ``cloud`` by the per-node correction implied by the two graphs.

    Raises:
        ValueError: if the graphs disagree in length or in node timestamps.
    """
    if len(previous_pose_graph.poses) != len(next_pose_graph.poses):
        raise ValueError(
            f"pose graph length mismatch: previous={len(previous_pose_graph.poses)}, "
            f"next={len(next_pose_graph.poses)}"
        )
    if len(previous_pose_graph.poses) == 0:
        # No correction available — pass through.
        return cloud

    prev_ts, prev_T = path_to_arrays(previous_pose_graph)
    next_ts, next_T = path_to_arrays(next_pose_graph)
    if not np.allclose(prev_ts, next_ts, atol=_NODE_TIME_MATCH_TOL):
        raise ValueError("pose graph node timestamps do not match between prev and next")

    order = np.argsort(prev_ts, kind="stable")
    prev_ts = prev_ts[order]
    prev_T = prev_T[order]
    next_T = next_T[order]

    deltas = compute_node_deltas(prev_T, next_T)

    world = cloud.world_positions().astype(np.float64)
    latest_ns = cloud.per_point_latest_timestamp()
    point_times = latest_ns.astype(np.float64) / 1_000_000_000.0

    new_world = lbs_warp_positions(world, point_times, prev_ts, deltas)
    new_voxels = np.rint(new_world / cloud.voxel_size).astype(np.int32)

    voxels, quantity, event_indices = merge_duplicate_voxels(
        new_voxels, cloud.quantity, cloud.event_indices
    )

    # event_timestamps is unchanged (the events still refer to the same physical
    # observations, just at remapped voxel indices). DynamicCloud copies/normalizes
    # the array internally so sharing the reference is safe.
    return DynamicCloud(
        voxels=voxels,
        quantity=quantity,
        event_indices=event_indices,
        event_timestamps=cloud.event_timestamps,
        voxel_size=cloud.voxel_size,
        frame_id=cloud.frame_id,
        ts=cloud.ts,
    )


class ApplyClosureConfig(ModuleConfig):
    world_frame: str = FRAME_MAP
    # Log a one-line summary every time a correction is applied.
    log_each_apply: bool = True


class ApplyClosure(Module):
    """Warp the global voxel map by a pose-graph loop-closure correction."""

    config: ApplyClosureConfig

    global_map: In[DynamicCloud]
    previous_pose_graph: In[Path]
    next_pose_graph: In[Path]
    corrected_global_map: Out[DynamicCloud]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._latest_map: DynamicCloud | None = None
        self._latest_prev: Path | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.global_map.subscribe(self._on_global_map)))
        self.register_disposable(
            Disposable(self.previous_pose_graph.subscribe(self._on_previous_pose_graph))
        )
        self.register_disposable(
            Disposable(self.next_pose_graph.subscribe(self._on_next_pose_graph))
        )
        logger.info("ApplyClosure started")

    @rpc
    def stop(self) -> None:
        super().stop()

    def _on_global_map(self, msg: DynamicCloud) -> None:
        with self._lock:
            self._latest_map = msg

    def _on_previous_pose_graph(self, msg: Path) -> None:
        with self._lock:
            self._latest_prev = msg

    def _on_next_pose_graph(self, msg: Path) -> None:
        """Loop-closure trigger: apply correction to the latched map."""
        with self._lock:
            cloud = self._latest_map
            prev = self._latest_prev
        if cloud is None or prev is None:
            return
        t0 = time.monotonic()
        try:
            corrected = apply_closure_to_cloud(cloud, prev, msg)
        except ValueError as exc:
            logger.warning("ApplyClosure skipped", reason=str(exc))
            return
        corrected.ts = time.time()
        self.corrected_global_map.publish(corrected)
        if self.config.log_each_apply:
            logger.info(
                "ApplyClosure applied",
                num_nodes=len(msg.poses),
                num_points_in=len(cloud),
                num_points_out=len(corrected),
                elapsed_ms=round((time.monotonic() - t0) * 1000.0, 2),
            )
