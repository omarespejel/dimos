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

import threading
import time
from typing import Any

from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.nav_msgs.Path import Path
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.navigation.nav_stack.modules.mls_planner.planner import (
    MLS,
    points_to_mls,
    robot_height_in_voxels,
    surface_centers,
)
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class MlsPlannerConfig(ModuleConfig):
    world_frame: str = "world"
    voxel_size: float = 0.1
    robot_height: float = 0.75  # m


class MlsPlanner(Module):
    """3D multi-level surface planner.

    Stub: emits a 2-pose straight-line path from latest odometry to the goal.
    The real surface-graph A* will replace _plan() in a follow-up.
    """

    config: MlsPlannerConfig

    global_map: In[PointCloud2]
    odometry: In[Odometry]
    goal: In[PoseStamped]
    path: Out[Path]
    surfaces: Out[PointCloud2]  # debug: extracted MLS surface centers

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._latest_odom: Odometry | None = None
        self._latest_mls: MLS | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.odometry.subscribe(self._on_odom)))
        self.register_disposable(Disposable(self.global_map.subscribe(self._on_map)))
        self.register_disposable(Disposable(self.goal.subscribe(self._on_goal)))

    @rpc
    def stop(self) -> None:
        super().stop()

    def _on_odom(self, msg: Odometry) -> None:
        with self._lock:
            self._latest_odom = msg

    def _on_map(self, msg: PointCloud2) -> None:
        points = msg.points_f32()
        rh_voxels = robot_height_in_voxels(self.config.robot_height, self.config.voxel_size)
        mls = points_to_mls(points, self.config.voxel_size, rh_voxels)
        with self._lock:
            self._latest_mls = mls
        self._publish_surfaces(mls)

    def _on_goal(self, goal: PoseStamped) -> None:
        with self._lock:
            odom = self._latest_odom
        if odom is None:
            return
        path = self._plan(odom, goal)
        self.path.publish(path)

    def _publish_surfaces(self, mls: MLS) -> None:
        centers = surface_centers(mls, self.config.voxel_size)
        cloud = PointCloud2.from_numpy(
            points=centers,
            frame_id=self.config.world_frame,
            timestamp=time.time(),
        )
        self.surfaces.publish(cloud)
        logger.info("MlsPlanner extracted %d surfaces across %d columns", len(centers), len(mls))

    def _plan(self, odom: Odometry, goal: PoseStamped) -> Path:
        start_pose = PoseStamped(
            ts=time.time(),
            frame_id=self.config.world_frame,
            position=[odom.x, odom.y, odom.z],
            orientation=[
                odom.orientation.x,
                odom.orientation.y,
                odom.orientation.z,
                odom.orientation.w,
            ],
        )
        return Path(
            ts=time.time(),
            frame_id=self.config.world_frame,
            poses=[start_pose, goal],
        )
