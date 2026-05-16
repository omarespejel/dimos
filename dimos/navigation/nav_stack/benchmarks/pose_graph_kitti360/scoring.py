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

"""Module that scores a pose-graph SLAM module's loop closures against KITTI groundtruth.

Subscribes to two outputs that any pose-graph SLAM module exposes:

* ``pose_graph_edges: In[NavPath]`` — pose-graph edges where loop closures
  are tagged with ``orientation.w == 0.4`` (odometry edges use ``1.0``).
* ``loop_closure: In[NavPath]`` — one event per loop-closure update with
  per-keyframe deltas.

The scoring module needs to know, for each edge endpoint, which input scan
produced that keyframe. The producer publishes a timestamp on each endpoint's
``PoseStamped`` header — we keep a (timestamp → frame_id) cache built from
the playback module's send schedule so we can map back unambiguously even
after iSAM2 has shifted the optimized keyframe positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.msgs.nav_msgs.Path import Path as NavPath

# Default tag value used by the PGO publisher to mark loop-closure edges in
# the orientation.w field of pose_graph_edges PoseStamped pairs (odometry
# edges use 1.0). Both knobs are exposed on PoseGraphScoringConfig so any
# other pose-graph producer can dial in its own marker.
DEFAULT_LOOP_CLOSURE_TRAVERSABILITY = 0.4
DEFAULT_TRAVERSABILITY_TOLERANCE = 0.05


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
    frame_ids: list[int] = field(default_factory=list)
    send_timestamps: list[float] = field(default_factory=list)
    # JSON-friendly form of LoopGroundtruth.valid_loops_per_query:
    # frame_id → list of frame_ids that form valid loop pairs.
    valid_loops_per_query: dict[int, list[int]] = field(default_factory=dict)
    # Tag value the publisher writes into orientation.w to mark a
    # pose_graph_edges PoseStamped pair as a loop closure (vs the
    # odometry-edge default of 1.0). Both fields are config-driven so
    # different pose-graph SLAM producers can plug in their own marker.
    loop_closure_traversability: float = DEFAULT_LOOP_CLOSURE_TRAVERSABILITY
    traversability_tolerance: float = DEFAULT_TRAVERSABILITY_TOLERANCE


class PoseGraphScoringModule(Module):
    """Accumulates loop-closure detections and scores them against KITTI groundtruth."""

    config: PoseGraphScoringConfig

    pose_graph_edges: In[NavPath]
    loop_closure: In[NavPath]

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

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.loop_closure.subscribe(self._on_loop_closure)))
        self.register_disposable(
            Disposable(self.pose_graph_edges.subscribe(self._on_pose_graph_edges))
        )

    def _on_loop_closure(self, message: NavPath) -> None:
        del message
        self._loop_closure_events += 1

    def _on_pose_graph_edges(self, message: NavPath) -> None:
        pose_index = 0
        while pose_index + 1 < len(message.poses):
            start_pose = message.poses[pose_index]
            end_pose = message.poses[pose_index + 1]
            traversability = float(start_pose.orientation.w)
            if (
                abs(traversability - self.config.loop_closure_traversability)
                < self.config.traversability_tolerance
            ):
                start_frame_id = self._timestamp_to_frame(start_pose.ts)
                end_frame_id = self._timestamp_to_frame(end_pose.ts)
                if start_frame_id is not None and end_frame_id is not None:
                    pair = (start_frame_id, end_frame_id)
                    if pair not in self._detected_pairs:
                        self._detected_pairs.append(pair)
            pose_index += 2

    def _timestamp_to_frame(self, timestamp_sec: float) -> int | None:
        timestamp_ms = round(timestamp_sec * 1e3)
        # ±1 ms slop: PoseStamped.ts round-trips through (int32 sec, uint32 nsec).
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
        }


def _score_pairs(
    detected_pairs: list[tuple[int, int]],
    valid_loops_per_query: dict[int, set[int]],
) -> LoopMetrics:
    true_positives = 0
    false_positives = 0
    seen_queries_with_hit: set[int] = set()
    queries_with_any_groundtruth = {
        frame_id for frame_id, valid in valid_loops_per_query.items() if valid
    }
    for source_frame_id, target_frame_id in detected_pairs:
        source_valid = valid_loops_per_query.get(source_frame_id, set())
        target_valid = valid_loops_per_query.get(target_frame_id, set())
        if target_frame_id in source_valid or source_frame_id in target_valid:
            true_positives += 1
            seen_queries_with_hit.add(max(source_frame_id, target_frame_id))
        else:
            false_positives += 1
    false_negatives = len(queries_with_any_groundtruth - seen_queries_with_hit)
    return LoopMetrics(
        true_positive=true_positives,
        false_positive=false_positives,
        false_negative=false_negatives,
    )
