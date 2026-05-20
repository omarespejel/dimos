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

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Any

import numpy as np
from reactivex import interval
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.nav_msgs.Path import Path
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


@dataclass
class Scene:
    """Hand-crafted test world for the planner."""

    voxels: np.ndarray  # (N, 3) float32 world-frame coordinates of occupied voxel centers
    voxel_size: float
    start_position: tuple[float, float, float]
    start_orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    goal_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal_orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    name: str = "scene"


def _cell_centers(low: float, high: float, voxel_size: float) -> np.ndarray:
    """World-frame voxel-center positions for cells whose centers lie in [low, high].

    Generated via integer cell indices to avoid floating-point drift in
    ``np.arange(step=voxel_size)``, which would otherwise mis-bucket points
    at certain x and y values and produce missing stripes downstream.
    """
    i_min = math.floor(low / voxel_size)
    i_max = math.floor(high / voxel_size)
    return (np.arange(i_min, i_max + 1) + 0.5) * voxel_size


def _flat_floor(
    voxel_size: float,
    extent: tuple[float, float, float, float],
    z: float = 0.0,
    holes: list[tuple[float, float, float, float]] | None = None,
) -> np.ndarray:
    """Single-layer floor at height ``z`` over ``extent=(xmin, xmax, ymin, ymax)``,
    with rectangular ``holes`` cut out (e.g. footprints of objects sitting on it)."""
    xmin, xmax, ymin, ymax = extent
    xs = _cell_centers(xmin, xmax, voxel_size)
    ys = _cell_centers(ymin, ymax, voxel_size)
    z_center = (math.floor(z / voxel_size) + 0.5) * voxel_size
    fx, fy = np.meshgrid(xs, ys, indexing="ij")
    mask = np.ones(fx.shape, dtype=bool)
    for hx_min, hx_max, hy_min, hy_max in holes or []:
        mask &= ~((fx >= hx_min) & (fx <= hx_max) & (fy >= hy_min) & (fy <= hy_max))
    return np.stack([fx[mask], fy[mask], np.full(int(mask.sum()), z_center)], axis=1)


def _box_shell(
    voxel_size: float,
    bounds: tuple[float, float, float, float, float, float],
    include_bottom: bool = False,
) -> np.ndarray:
    """Hollow axis-aligned box: top face + 4 side walls. No interior.

    ``bounds=(xmin, xmax, ymin, ymax, zmin, zmax)``. ``include_bottom`` defaults
    False since boxes sitting on a floor occlude their bottom face from lidar.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    xs = _cell_centers(xmin, xmax, voxel_size)
    ys = _cell_centers(ymin, ymax, voxel_size)
    zs = _cell_centers(zmin, zmax, voxel_size)

    def _grid(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        ga, gb, gc = np.meshgrid(a, b, c, indexing="ij")
        return np.stack([ga.ravel(), gb.ravel(), gc.ravel()], axis=1)

    faces = [
        _grid(xs, ys, zs[-1:]),  # top
        _grid(xs[:1], ys, zs),  # -x wall
        _grid(xs[-1:], ys, zs),  # +x wall
        _grid(xs, ys[:1], zs),  # -y wall
        _grid(xs, ys[-1:], zs),  # +y wall
    ]
    if include_bottom:
        faces.append(_grid(xs, ys, zs[:1]))
    return np.concatenate(faces, axis=0)


def default_scene(voxel_size: float = 0.1) -> Scene:
    """Lidar-realistic shell scene: floor + tall central box + ramp + bridge.

    Robot starts at (-3, 0) and the goal is at (3, 0). A 2m square, 1m-tall
    box sits at the origin — too tall to climb. To its -y side, a 5-step ramp
    stretches from the floor's -y edge to the box; each step is 1 voxel
    (0.1m) tall, traversable. To its +y side, a 2m-wide bridge spans from
    the box to the floor's +y edge; its underside is at z=0.8m, leaving only
    0.7m of clearance under it — less than the robot's 0.75m height, so the
    walker should filter out the floor underneath the bridge as unreachable.

    Expected path from (-3,0) to (3,0): around the central box, using step 1
    of the ramp at the -y end as a low bridge to cross the obstacle strip
    (step 1 is the only step with a 1-voxel delta to the floor).
    """
    # Central tall box. zmax=0.95 → top voxel at z_voxel=9 (cleanly aligned).
    big_box = (-1.0, 1.0, -1.0, 1.0, 0.0, 0.95)

    # Five 1-voxel-tall steps along the -y side of the box, each 0.8m deep in y.
    # zmax = (k + 0.5) * voxel_size keeps floor-of-zmax/voxel_size = k.
    # Step 5 is split: left half (x in [-1, 0]) stays flat at z_voxel=5, and
    # the right half (x in [0, 1]) is sliced into 5 sub-steps climbing from
    # z_voxel=5 to z_voxel=9 (= box top), so the robot can reach the box top.
    step_x = (-1.0, 1.0)
    step_5_y = (-1.8, -1.0)
    steps = [
        (*step_x, -5.0, -4.2, 0.0, 0.15),  # step 1: top voxel z_voxel=1
        (*step_x, -4.2, -3.4, 0.0, 0.25),  # step 2: z_voxel=2
        (*step_x, -3.4, -2.6, 0.0, 0.35),  # step 3
        (*step_x, -2.6, -1.8, 0.0, 0.45),  # step 4
        # Step 5 left half (flat at z_voxel=5).
        (-1.0, 0.0, *step_5_y, 0.0, 0.55),
        # Step 5 right half: 5 sub-steps climbing in -x from z=5 to z=9.
        (0.8, 1.0, *step_5_y, 0.0, 0.55),  # sub A: z=5 (entry at +x edge)
        (0.6, 0.8, *step_5_y, 0.0, 0.65),  # sub B: z=6
        (0.4, 0.6, *step_5_y, 0.0, 0.75),  # sub C: z=7
        (0.2, 0.4, *step_5_y, 0.0, 0.85),  # sub D: z=8
        (0.0, 0.2, *step_5_y, 0.0, 0.95),  # sub E: z=9 (= box top)
    ]

    # Bridge on +y side of box. Top voxel at z_voxel=9 (matches box top); 2
    # voxels thick (underside at z_voxel=8, z=0.8m).
    bridge = (-1.0, 1.0, 1.0, 5.0, 0.85, 0.95)

    # Floor holes: the strip from ramp through central box (no floor visible).
    # No hole under the bridge — lidar sees the floor through the gap on its
    # sides, and the column walker will filter it as unreachable.
    floor = _flat_floor(
        voxel_size,
        extent=(-5.0, 5.0, -5.0, 5.0),
        holes=[(-1.0, 1.0, -5.0, 1.0)],
    )
    box_voxels = _box_shell(voxel_size, big_box)
    step_voxels = [_box_shell(voxel_size, s) for s in steps]
    # include_bottom=True: lidar would see the bridge's underside from below,
    # so emit voxels there. Without this, interior columns under the bridge
    # only have the top voxel and the column walker computes too generous a
    # gap (8 voxels) to the floor and emits a phantom-reachable floor surface.
    bridge_voxels = _box_shell(voxel_size, bridge, include_bottom=True)
    voxels = np.concatenate([floor, box_voxels, *step_voxels, bridge_voxels], axis=0).astype(
        np.float32
    )

    return Scene(
        voxels=voxels,
        voxel_size=voxel_size,
        start_position=(-3.0, 0.0, 0.5),
        # Goal at the +y end of the bridge: forces the planner to climb the
        # ramp + sub-staircase, traverse the box top, and walk the bridge.
        goal_position=(0.0, 4.5, 1.4),
        name="default_floor_box_ramp_bridge",
    )


class EvaluatorConfig(ModuleConfig):
    world_frame: str = "world"
    body_frame: str = "body"
    publish_period: float = 5.0  # s — republish all messages this often


class Evaluator(Module):
    """Publishes a synthetic scene and evaluates the planner's returned path.

    Outputs the three inputs a planner expects (global_map, odometry, goal);
    subscribes to the planner's path output and logs basic metrics.
    """

    config: EvaluatorConfig

    global_map: Out[PointCloud2]
    odometry: Out[Odometry]
    goal: Out[PoseStamped]
    path: In[Path]

    def __init__(self, scene: Scene | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._scene: Scene = scene if scene is not None else default_scene()
        self._lock = threading.Lock()

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.path.subscribe(self._on_path)))
        self.register_disposable(interval(self.config.publish_period).subscribe(self._publish_all))
        logger.info("Evaluator started with scene=%s", self._scene.name)

    @rpc
    def stop(self) -> None:
        super().stop()

    def _publish_all(self, _: Any) -> None:
        self._publish_map()
        self._publish_odom()
        self._publish_goal()

    def _publish_map(self) -> None:
        cloud = PointCloud2.from_numpy(
            points=self._scene.voxels,
            frame_id=self.config.world_frame,
            timestamp=time.time(),
        )
        self.global_map.publish(cloud)

    def _publish_odom(self) -> None:
        x, y, z = self._scene.start_position
        qx, qy, qz, qw = self._scene.start_orientation
        odom = Odometry(
            ts=time.time(),
            frame_id=self.config.world_frame,
            child_frame_id=self.config.body_frame,
            pose=Pose(position=Vector3(x, y, z), orientation=Quaternion(qx, qy, qz, qw)),
        )
        self.odometry.publish(odom)

    def _publish_goal(self) -> None:
        x, y, z = self._scene.goal_position
        qx, qy, qz, qw = self._scene.goal_orientation
        goal = PoseStamped(
            ts=time.time(),
            frame_id=self.config.world_frame,
            position=Vector3(x, y, z),
            orientation=Quaternion(qx, qy, qz, qw),
        )
        self.goal.publish(goal)
        logger.info("Evaluator published goal at %s", self._scene.goal_position)

    def _on_path(self, path: Path) -> None:
        n = len(path.poses)
        if n == 0:
            logger.warning("Evaluator received empty path")
            return
        total_xy = 0.0
        total_z = 0.0
        for a, b in zip(path.poses, path.poses[1:], strict=False):
            dx = b.position.x - a.position.x
            dy = b.position.y - a.position.y
            dz = b.position.z - a.position.z
            total_xy += (dx * dx + dy * dy) ** 0.5
            total_z += abs(dz)
        logger.info(
            "Evaluator received path: %d poses, xy_len=%.2fm, z_traveled=%.2fm",
            n,
            total_xy,
            total_z,
        )
