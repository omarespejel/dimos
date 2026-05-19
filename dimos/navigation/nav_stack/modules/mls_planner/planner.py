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

"""Pure functions for the multi-level surface map (MLS) planner.

No LCM, no Module — just numpy in, numpy/dicts out, so this is unit-testable
without the framework.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SurfacePatch:
    """A standable surface in a single (x, y) voxel column.

    ``z_top`` is the voxel index of the top voxel of the solid the robot would
    stand on. World height of the surface center is ``(z_top + 0.5) * voxel_size``.
    """

    z_top: int


# Voxel column → list of surfaces (sorted by z_top ascending).
MLS = dict[tuple[int, int], list[SurfacePatch]]


def points_to_mls(
    points: np.ndarray,
    voxel_size: float,
    robot_height_voxels: int,
) -> MLS:
    """Extract a multi-level surface map from a voxel-center point cloud.

    Buckets points by (x_cell, y_cell), then walks each column bottom-to-top
    and emits a surface patch every time we leave a contiguous run of solid
    voxels with enough clear air above for the robot to stand. Surfaces with
    no upper bound (top of the column) are always emitted.
    """
    if points.size == 0:
        return {}
    if voxel_size <= 0.0:
        raise ValueError(f"voxel_size must be > 0, got {voxel_size}")
    if robot_height_voxels <= 0:
        raise ValueError(f"robot_height_voxels must be > 0, got {robot_height_voxels}")

    indices = np.floor(points / voxel_size).astype(np.int64)
    columns: dict[tuple[int, int], list[int]] = defaultdict(list)
    for kx, ky, kz in indices:
        columns[(int(kx), int(ky))].append(int(kz))

    mls: MLS = {}
    for col, zs in columns.items():
        surfaces = _extract_surfaces(zs, robot_height_voxels)
        if surfaces:
            mls[col] = surfaces
    return mls


def _extract_surfaces(z_indices: list[int], robot_height_voxels: int) -> list[SurfacePatch]:
    """Walk one column's z-indices and emit surface candidates.

    Algorithm: for each gap between consecutive populated voxels, if the gap is
    at least ``robot_height_voxels`` cells of clear air, the lower voxel is the
    top of a standable surface. The topmost populated voxel is always emitted
    (infinite air above).
    """
    z_sorted = sorted(set(z_indices))
    if not z_sorted:
        return []

    surfaces: list[SurfacePatch] = []
    prev_z = z_sorted[0]
    for z in z_sorted[1:]:
        gap = z - prev_z - 1
        if gap >= robot_height_voxels:
            surfaces.append(SurfacePatch(z_top=prev_z))
        prev_z = z
    surfaces.append(SurfacePatch(z_top=prev_z))
    return surfaces


def robot_height_in_voxels(robot_height: float, voxel_size: float) -> int:
    """Conservative clearance: round up so we never accept a too-cramped surface."""
    return max(1, math.ceil(robot_height / voxel_size))


def surface_centers(mls: MLS, voxel_size: float) -> np.ndarray:
    """Flatten an MLS to an (N, 3) array of surface-patch world-frame centers.

    Useful for publishing the MLS as a debug PointCloud2.
    """
    if not mls:
        return np.zeros((0, 3), dtype=np.float32)
    half = 0.5 * voxel_size
    out = np.empty((sum(len(v) for v in mls.values()), 3), dtype=np.float32)
    i = 0
    for (kx, ky), patches in mls.items():
        for p in patches:
            out[i, 0] = kx * voxel_size + half
            out[i, 1] = ky * voxel_size + half
            out[i, 2] = p.z_top * voxel_size + half
            i += 1
    return out
