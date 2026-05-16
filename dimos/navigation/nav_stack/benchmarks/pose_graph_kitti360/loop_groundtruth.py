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

"""Compute groundtruth loop closures from a trajectory.

A pair (i, j) with j < i is a groundtruth loop iff:
* ``i - j >= MIN_FRAME_GAP`` (default 50) — exclude near-temporal neighbours
* ``|pose(i).t - pose(j).t| <= MAX_LOOP_DISTANCE_M`` (default 4.0m) —
  spatial threshold matching the LCDNet / KITTI-360 convention.

Output: per-i list of valid earlier indices, so the eval can score whether
PGO's detected pair (i, j_detected) hits any of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_MIN_FRAME_GAP = 50
DEFAULT_MAX_LOOP_DISTANCE_M = 4.0


@dataclass
class LoopGroundtruth:
    """Per-query (index i) the set of valid loop indices j < i."""

    min_frame_gap: int
    max_distance_m: float
    valid_loops_per_query: dict[int, set[int]]

    @property
    def queries_with_loop(self) -> int:
        return sum(1 for v in self.valid_loops_per_query.values() if v)

    @property
    def total_loop_pairs(self) -> int:
        return sum(len(v) for v in self.valid_loops_per_query.values())


def compute_loop_groundtruth(
    frame_ids: list[int],
    positions_xyz: np.ndarray,
    min_frame_gap: int = DEFAULT_MIN_FRAME_GAP,
    max_distance_m: float = DEFAULT_MAX_LOOP_DISTANCE_M,
) -> LoopGroundtruth:
    """Compute the groundtruth-loops table.

    Args:
        frame_ids: ordered list of frame IDs (e.g. KITTI frame indices).
        positions_xyz: (N, 3) world-frame translation of each frame.
        min_frame_gap: minimum index distance (in this list) to count.
        max_distance_m: spatial radius for a positive loop.

    Returns:
        ``LoopGroundtruth`` with ``valid_loops_per_query``: query frame_id
        → set of earlier frame_ids that satisfy both thresholds.
    """
    if positions_xyz.shape != (len(frame_ids), 3):
        raise ValueError(
            f"positions_xyz shape {positions_xyz.shape} doesn't match "
            f"len(frame_ids)={len(frame_ids)}"
        )

    valid: dict[int, set[int]] = {fid: set() for fid in frame_ids}
    for i in range(len(frame_ids)):
        if i < min_frame_gap:
            continue
        # Bound the search: any j with |i - j| >= min_frame_gap.
        upper_j = i - min_frame_gap
        if upper_j < 0:
            continue
        deltas = positions_xyz[: upper_j + 1] - positions_xyz[i]
        distances = np.linalg.norm(deltas, axis=1)
        matches = np.where(distances <= max_distance_m)[0]
        for j_index in matches:
            valid[frame_ids[i]].add(frame_ids[int(j_index)])

    return LoopGroundtruth(
        min_frame_gap=min_frame_gap,
        max_distance_m=max_distance_m,
        valid_loops_per_query=valid,
    )


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
        p, r = self.precision, self.recall
        if not (p > 0 and r > 0):
            return 0.0
        return 2.0 * p * r / (p + r)


def score_detected_loops(
    detected_pairs: list[tuple[int, int]],
    groundtruth: LoopGroundtruth,
) -> LoopMetrics:
    """Score detected (query_id, candidate_id) pairs against groundtruth.

    A detection is a true positive iff its candidate is in the
    groundtruth set for that query (or vice-versa — order-agnostic).

    Recall denominator = number of queries that have at least one
    valid loop. (We don't count "valid pairs missed" because a single
    correct detection per query is enough to count.)
    """
    tp = 0
    fp = 0
    seen_queries_with_hit: set[int] = set()
    queries_with_any_gt = {q for q, valid in groundtruth.valid_loops_per_query.items() if valid}

    for src, dst in detected_pairs:
        # Order-agnostic: PGO may report (target, source) or (source, target).
        src_valid = groundtruth.valid_loops_per_query.get(src, set())
        dst_valid = groundtruth.valid_loops_per_query.get(dst, set())
        if dst in src_valid or src in dst_valid:
            tp += 1
            # For per-query recall, mark whichever side was the "later" query.
            query = max(src, dst)
            seen_queries_with_hit.add(query)
        else:
            fp += 1

    fn = len(queries_with_any_gt - seen_queries_with_hit)
    return LoopMetrics(true_positive=tp, false_positive=fp, false_negative=fn)
