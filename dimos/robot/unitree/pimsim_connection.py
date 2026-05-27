# Copyright 2025-2026 Dimensional Inc.
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

"""Go2 connection backed by pimsim's BabylonSceneViewerModule.

Parallel to :mod:`dimos.robot.unitree.dimsim_connection`. The Go2's
real control surface is Unitree's onboard SPORT_MOD FSM
(StandUp/BalanceStand/FreeWalk/…), accessed over WebRTC; in a sim
backend those FSM operations have no analogue, so we stub them as
no-op ``True`` and let the simulator do its thing. The actual sim
lifecycle (BabylonSceneViewerModule + optional headless browser) is
wired by the blueprint composition — this class is just the
``Go2ConnectionProtocol`` adapter that republishes ``/odom`` as TF.
"""

from __future__ import annotations

from collections.abc import Callable
import functools
import time
from typing import Any

from reactivex import Observable, Subject

from dimos.core.global_config import GlobalConfig
from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.protocol.tf.tf import LCMTF
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_WIDTH = 640
_HEIGHT = 288
_FOV_DEG = 46


class PimSimConnection:
    """``Go2ConnectionProtocol`` impl over pimsim's LCM topics."""

    camera_info_static: CameraInfo = CameraInfo.from_fov(
        fov_deg=_FOV_DEG,
        width=_WIDTH,
        height=_HEIGHT,
        axis="horizontal",
        frame_id="camera_optical",
    )

    def __init__(self, global_config: GlobalConfig) -> None:
        del global_config  # reserved for future readiness/headless config
        self._odom_transport: LCMTransport[PoseStamped] = LCMTransport("/odom", PoseStamped)
        self._unsubscribe_odom: Callable[[], None] | None = None
        self._tf = LCMTF()

    def start(self) -> None:
        self._odom_transport.start()
        self._unsubscribe_odom = self._odom_transport.subscribe(self._handle_odom)
        # ModuleCoordinator starts every module in parallel via
        # safe_thread_map. With BabylonSceneViewerModule + uvicorn +
        # SceneLidarModule (native subprocess) + VoxelGridMapper (CUDA)
        # all racing for the GIL, the TF service's LCM handler thread
        # can take longer than its 5 s "did it start" sanity check
        # allows. Retry a few times before giving up — total wait stays
        # bounded.
        last_err: Exception | None = None
        for attempt in range(6):
            try:
                self._tf.start()
                return
            except RuntimeError as exc:
                if "LCM handler thread failed to start" not in str(exc):
                    raise
                last_err = exc
                logger.warning(
                    "PimSimConnection: LCM tf start race (attempt %d/6), retrying", attempt + 1
                )
                time.sleep(0.5)
        assert last_err is not None
        raise last_err

    def stop(self) -> None:
        self._tf.stop()
        if self._unsubscribe_odom is not None:
            self._unsubscribe_odom()
        self._odom_transport.stop()

    # ``Go2ConnectionProtocol`` requires these stream methods. The actual
    # data comes from BabylonSceneViewerModule's own LCM publishers; the
    # connection class doesn't see camera/lidar bytes directly.
    @functools.cache
    def lidar_stream(self) -> Observable[PointCloud2]:
        return Subject()

    @functools.cache
    def odom_stream(self) -> Observable[PoseStamped]:
        return Subject()

    @functools.cache
    def video_stream(self) -> Observable[Image]:
        return Subject()

    # SPORT_MOD FSM ops — no-op stubs. Go2's real firmware owns these;
    # the simulator does its own kinematics directly from /cmd_vel.
    def move(self, twist: Twist, duration: float = 0.0) -> bool:
        del twist, duration
        return True

    def standup(self) -> bool:
        return True

    def liedown(self) -> bool:
        return True

    def balance_stand(self) -> bool:
        return True

    def set_obstacle_avoidance(self, enabled: bool = True) -> None:
        del enabled

    def enable_rage_mode(self) -> bool:
        return True

    def publish_request(self, topic: str, data: dict[str, Any]) -> dict[Any, Any]:
        del topic, data
        return {"status": "ok", "message": "pimsim stub"}

    def _handle_odom(self, msg: PoseStamped) -> None:
        self._tf.publish(*_odom_to_tf(msg))


def _odom_to_tf(odom: PoseStamped) -> list[Transform]:
    camera_link = Transform(
        translation=Vector3(0.3, 0.0, 0.0),
        rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
        frame_id="base_link",
        child_frame_id="camera_link",
        ts=odom.ts,
    )
    camera_optical = Transform(
        translation=Vector3(0.0, 0.0, 0.0),
        rotation=Quaternion(-0.5, 0.5, -0.5, 0.5),
        frame_id="camera_link",
        child_frame_id="camera_optical",
        ts=odom.ts,
    )
    return [
        Transform.from_pose("base_link", odom),
        camera_link,
        camera_optical,
    ]


__all__ = ["PimSimConnection"]
