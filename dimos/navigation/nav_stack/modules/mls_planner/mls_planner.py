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

"""Multi-Level Surface (MLS) path planner.

Extracts walkable surfaces from a voxelized global map, builds a sparse
waypoint graph over those surfaces, and plans paths via local A* plus
shortest-path search on the graph. Skeleton — algorithm is filled in
piecewise.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.nav_msgs.Path import Path
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class MLSPlannerConfig(ModuleConfig):
    world_frame: str = "map"
    voxel_size: float = 0.1
    robot_height: float = 1.0


def _extract_surfaces(points: np.ndarray, voxel_size: float, robot_height: float) -> np.ndarray:
    """Find walkable surface tops in a voxelized point cloud.

    Iterate through all the columns, find continuous areas of
    free space. If the free space column is at least robot height,
    add the bottom of this range as a surface.
    """
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    indices = np.floor(points / voxel_size).astype(np.int64)
    ix, iy, iz = indices[:, 0], indices[:, 1], indices[:, 2]

    order = np.lexsort((iz, iy, ix))
    sx, sy, sz = ix[order], iy[order], iz[order]

    height_cells = int(np.ceil(robot_height / voxel_size))

    next_same_col = np.zeros(len(sx), dtype=bool)
    next_same_col[:-1] = (sx[:-1] == sx[1:]) & (sy[:-1] == sy[1:])

    gap = np.empty(len(sx), dtype=np.int64)
    gap[:-1] = sz[1:] - sz[:-1]
    gap[-1] = 0

    is_surface = (~next_same_col) | (gap > height_cells)

    surf_ix = sx[is_surface]
    surf_iy = sy[is_surface]
    surf_iz = sz[is_surface]

    x = (surf_ix.astype(np.float32) + 0.5) * voxel_size
    y = (surf_iy.astype(np.float32) + 0.5) * voxel_size
    z = (surf_iz.astype(np.float32) + 1.0) * voxel_size
    return np.column_stack([x, y, z]).astype(np.float32)


class MLSPlanner(Module):
    config: MLSPlannerConfig

    global_map: In[PointCloud2]
    start_pose: In[Odometry]
    goal_pose: In[Odometry]
    path: Out[Path]
    surface_map: Out[PointCloud2]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_start: Odometry | None = None

    async def handle_global_map(self, msg: PointCloud2) -> None:
        points, _ = msg.as_numpy()
        if points is None or len(points) == 0:
            return
        surface_points = _extract_surfaces(points, self.config.voxel_size, self.config.robot_height)
        logger.info("Surfaces extracted", count=len(surface_points))
        self.surface_map.publish(
            PointCloud2.from_numpy(
                surface_points,
                frame_id=self.config.world_frame,
                timestamp=time.time(),
            )
        )

    async def handle_start_pose(self, msg: Odometry) -> None:
        self._latest_start = msg

    async def handle_goal_pose(self, msg: Odometry) -> None:
        if self._latest_start is None:
            logger.warning("MLSPlanner received goal before start; skipping")
            return
        logger.info(
            "MLSPlanner goal received (not yet implemented)",
            start=(self._latest_start.x, self._latest_start.y, self._latest_start.z),
            goal=(msg.x, msg.y, msg.z),
        )
