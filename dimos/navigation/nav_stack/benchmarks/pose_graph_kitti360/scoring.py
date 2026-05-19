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

"""Score a pose-graph SLAM module's loop closures against KITTI groundtruth.

Subscribes to two outputs that any pose-graph SLAM module exposes:

* ``pose_graph: In[Graph3D]`` — full pose-graph snapshot. Loop-closure
  edges are identified by ``metadata_id == EDGE_LOOP_CLOSURE``; each
  node carries the keyframe creation time in ``pose.ts``, which we map
  back to the input scan that produced it.
* ``loop_closure_event: In[GraphDelta3D]`` — one event per loop-closure
  update, carrying per-keyframe (pre-pose, SE(3) delta) pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
import time
from typing import Any

from pydantic import Field
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.msgs.nav_msgs.Graph3D import Graph3D
from dimos.msgs.nav_msgs.GraphDelta3D import GraphDelta3D
from dimos.msgs.nav_msgs.Odometry import Odometry

# edge-type enum (matches build_pose_graph in pgo/cpp/main.cpp).
EDGE_LOOP_CLOSURE = 1


@dataclass
class LoopMetrics:
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom > 0 else float("nan")

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom > 0 else float("nan")

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        if not (precision > 0 and recall > 0):
            return 0.0
        return 2.0 * precision * recall / (precision + recall)


class PoseGraphScoringConfig(ModuleConfig):
    frame_ids: list[int] = Field(default_factory=list)
    send_timestamps: list[float] = Field(default_factory=list)
    valid_loops_per_query: dict[int, list[int]] = Field(default_factory=dict)
    # Ground-truth lidar positions per frame_id, used to compute ATE against
    # the corrected_odometry stream.  Empty by default — the runner fills it
    # in when constructing the blueprint so the module stays self-contained.
    groundtruth_positions: dict[int, list[float]] = Field(default_factory=dict)


class PoseGraphScoringModule(Module):
    """Accumulates loop-closure detections and scores them against KITTI groundtruth.

    Also captures corrected_odometry to compute per-frame median latency (from
    inter-arrival times) and ATE against ground-truth positions.
    """

    config: PoseGraphScoringConfig

    pose_graph: In[Graph3D]
    loop_closure_event: In[GraphDelta3D]
    corrected_odometry: In[Odometry]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._detected_pairs: list[tuple[int, int]] = []
        self._loop_closure_events: int = 0
        self._timestamp_ms_to_frame_id: dict[int, int] = {
            round(send_timestamp * 1e3): frame_id
            for frame_id, send_timestamp in zip(
                self.config.frame_ids, self.config.send_timestamps, strict=True
            )
        }
        # Per-arrival wall-clock timestamps of corrected_odometry messages.
        # Inter-arrival deltas approximate per-frame processing latency when
        # the binary keeps up with the playback module's publish cadence.
        self._corrected_arrival_times: list[float] = []
        # (frame_id, predicted_xyz, gt_xyz) triples for ATE.
        self._predicted_vs_gt: list[tuple[int, tuple[float, float, float], tuple[float, float, float]]] = []

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(
            Disposable(self.loop_closure_event.subscribe(self._on_loop_closure_event))
        )
        self.register_disposable(Disposable(self.pose_graph.subscribe(self._on_pose_graph)))
        self.register_disposable(
            Disposable(self.corrected_odometry.subscribe(self._on_corrected_odometry))
        )

    def _on_corrected_odometry(self, message: Odometry) -> None:
        self._corrected_arrival_times.append(time.monotonic())
        # Map the message's stamp back to its source frame_id.  Same ±1ms slop
        # heuristic as _timestamp_to_frame uses for pose-graph node times.
        frame_id = self._timestamp_to_frame(message.ts)
        if frame_id is None:
            return
        gt = self.config.groundtruth_positions.get(frame_id)
        if gt is None or len(gt) != 3:
            return
        position = message.pose.pose.position
        self._predicted_vs_gt.append(
            (frame_id, (position.x, position.y, position.z), (gt[0], gt[1], gt[2]))
        )

    def _on_loop_closure_event(self, message: GraphDelta3D) -> None:
        del message
        self._loop_closure_events += 1

    def _on_pose_graph(self, message: Graph3D) -> None:
        id_to_node_ts: dict[int, float] = {n.id: n.pose.ts for n in message.nodes}
        for edge in message.edges:
            if edge.metadata_id != EDGE_LOOP_CLOSURE:
                continue
            start_ts = id_to_node_ts.get(edge.start_id)
            end_ts = id_to_node_ts.get(edge.end_id)
            if start_ts is None or end_ts is None:
                continue
            start_frame_id = self._timestamp_to_frame(start_ts)
            end_frame_id = self._timestamp_to_frame(end_ts)
            if start_frame_id is None or end_frame_id is None:
                continue
            pair = (start_frame_id, end_frame_id)
            if pair not in self._detected_pairs:
                self._detected_pairs.append(pair)

    def _timestamp_to_frame(self, timestamp_sec: float) -> int | None:
        timestamp_ms = round(timestamp_sec * 1e3)
        # ±1 ms slop: pose.ts round-trips through (int32 sec, uint32 nsec).
        for slop_ms in (0, -1, 1):
            frame_id = self._timestamp_ms_to_frame_id.get(timestamp_ms + slop_ms)
            if frame_id is not None:
                return frame_id
        return None

    @rpc
    def get_results(self) -> dict[str, Any]:
        valid_loops_per_query: dict[int, set[int]] = {
            frame_id: set(loops) for frame_id, loops in self.config.valid_loops_per_query.items()
        }
        metrics = _score_pairs(self._detected_pairs, valid_loops_per_query)
        queries_with_loop = sum(1 for valid in valid_loops_per_query.values() if valid)
        total_pairs = sum(len(valid) for valid in valid_loops_per_query.values())
        return {
            "scans_played": len(self.config.frame_ids),
            "groundtruth_queries_with_loop": queries_with_loop,
            "groundtruth_total_loop_pairs": total_pairs,
            "detected_loop_edges": len(self._detected_pairs),
            "loop_closure_events": self._loop_closure_events,
            "true_positive": metrics.true_positive,
            "false_positive": metrics.false_positive,
            "false_negative": metrics.false_negative,
            "precision": (metrics.precision if math.isfinite(metrics.precision) else None),
            "recall": metrics.recall if math.isfinite(metrics.recall) else None,
            "f1": metrics.f1,
            "per_frame_median_ms": _median_inter_arrival_ms(self._corrected_arrival_times),
            "corrected_odometry_samples": len(self._corrected_arrival_times),
            "ate_meters": _absolute_trajectory_error(self._predicted_vs_gt),
            "ate_samples": len(self._predicted_vs_gt),
        }


def _median_inter_arrival_ms(arrival_times: list[float]) -> float | None:
    """Median inter-arrival time (ms) of consecutive samples, proxy for per-frame
    processing latency when the producer keeps up with playback cadence."""
    if len(arrival_times) < 2:
        return None
    deltas_ms = [
        (arrival_times[i] - arrival_times[i - 1]) * 1000.0
        for i in range(1, len(arrival_times))
    ]
    return statistics.median(deltas_ms)


def _absolute_trajectory_error(
    predicted_vs_gt: list[tuple[int, tuple[float, float, float], tuple[float, float, float]]],
) -> float | None:
    """ATE = RMSE of (predicted - groundtruth) position over the trajectory.

    Both predicted and groundtruth are in the world frame already (playback
    publishes ground-truth-derived odometry as input; corrected_odometry is
    the producer's adjusted version of it). No alignment step; if the producer
    didn't drift, ATE should be near zero.
    """
    if not predicted_vs_gt:
        return None
    total_sq = 0.0
    for _frame_id, predicted, gt in predicted_vs_gt:
        dx = predicted[0] - gt[0]
        dy = predicted[1] - gt[1]
        dz = predicted[2] - gt[2]
        total_sq += dx * dx + dy * dy + dz * dz
    return math.sqrt(total_sq / len(predicted_vs_gt))


def _score_pairs(
    detected_pairs: list[tuple[int, int]],
    valid_loops_per_query: dict[int, set[int]],
) -> LoopMetrics:
    # A query contributes 1 TP if any of its edges matched groundtruth,
    # otherwise 1 FP. Duplicate detections for the same query collapse.
    seen_queries_with_hit: set[int] = set()
    seen_queries_without_hit: set[int] = set()
    queries_with_any_groundtruth = {
        frame_id for frame_id, valid in valid_loops_per_query.items() if valid
    }
    for source_frame_id, target_frame_id in detected_pairs:
        source_valid = valid_loops_per_query.get(source_frame_id, set())
        target_valid = valid_loops_per_query.get(target_frame_id, set())
        query_frame_id = max(source_frame_id, target_frame_id)
        if target_frame_id in source_valid or source_frame_id in target_valid:
            seen_queries_with_hit.add(query_frame_id)
        else:
            seen_queries_without_hit.add(query_frame_id)
    # A query that fires both a TP and a FP edge is counted as TP only
    # (one good detection is enough to say LoopClosure recognised the place).
    seen_queries_without_hit -= seen_queries_with_hit
    return LoopMetrics(
        true_positive=len(seen_queries_with_hit),
        false_positive=len(seen_queries_without_hit),
        false_negative=len(queries_with_any_groundtruth - seen_queries_with_hit),
    )
