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

"""Hand-crafted synthetic scenarios for black-box planner evaluation.

Each scenario is a self-contained (map, start, goal, expectation) bundle.
Maps are PointCloud2 obstacle clouds, start/goal are Odometry poses, and
``expect_path`` records whether a planner *should* be able to find a path
(used by the evaluator to score pass/fail).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.navigation.nav_stack.evaluator.mesh_loader import load_voxelized_mesh

WORLD_FRAME = "map"
MESH_PATH = "/home/andrew/Downloads/model.glb"


@dataclass
class PlannerScenario:
    name: str
    global_map: PointCloud2
    start_pose: Odometry
    goal_pose: Odometry
    expect_path: bool


def _odom(x: float, y: float, z: float = 0.0, frame: str = WORLD_FRAME) -> Odometry:
    pose = Pose(position=[x, y, z], orientation=[0.0, 0.0, 0.0, 1.0])
    return Odometry(frame_id=frame, child_frame_id="body", pose=pose)


def _cloud(points: np.ndarray, frame: str = WORLD_FRAME) -> PointCloud2:
    if points.size == 0:
        points = np.zeros((0, 3), dtype=np.float32)
    return PointCloud2.from_numpy(points.astype(np.float32), frame_id=frame)


def _wall(
    x0: float, y0: float, x1: float, y1: float, *, spacing: float = 0.1, height: float = 0.5
) -> np.ndarray:
    """Sample a vertical-wall obstacle as a line of points from (x0,y0) to (x1,y1)."""
    length = float(np.hypot(x1 - x0, y1 - y0))
    n = max(2, int(np.ceil(length / spacing)))
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    zs = np.linspace(0.0, height, max(2, int(np.ceil(height / spacing))))
    grid_xs, grid_zs = np.meshgrid(xs, zs)
    grid_ys, _ = np.meshgrid(ys, zs)
    return np.column_stack([grid_xs.ravel(), grid_ys.ravel(), grid_zs.ravel()])


def _floor(
    x_min: float = -2.0,
    x_max: float = 8.0,
    y_min: float = -3.0,
    y_max: float = 3.0,
    spacing: float = 0.25,
) -> np.ndarray:
    """Flat ground plane sampled as points at z=0."""
    xs = np.arange(x_min, x_max + spacing, spacing)
    ys = np.arange(y_min, y_max + spacing, spacing)
    grid_xs, grid_ys = np.meshgrid(xs, ys)
    pts = np.column_stack([grid_xs.ravel(), grid_ys.ravel(), np.zeros(grid_xs.size)])
    return pts


def _empty_map() -> PointCloud2:
    return _cloud(_floor())


def _map_with_walls(*walls: np.ndarray) -> PointCloud2:
    return _cloud(np.vstack([_floor(), *walls]))


def _mesh_scene(name: str, mesh_path: str, expect_path: bool = True) -> PlannerScenario:
    points = load_voxelized_mesh(mesh_path)
    return PlannerScenario(
        name=name,
        global_map=_cloud(points),
        start_pose=_odom(-5.0, 0.0),
        goal_pose=_odom(5.0, 0.0),
        expect_path=expect_path,
    )


def default_scenarios() -> list[PlannerScenario]:
    return [_mesh_scene("loaded_mesh", MESH_PATH)]
