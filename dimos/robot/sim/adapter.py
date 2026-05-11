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

"""DimSimAdapter — sits between DimSimBridge and consumers that expect
standard nav-stack types.

DimSimBridge publishes the raw types DimSim emits: ``odom_raw: PoseStamped``.
nav_stack consumers want ``Odometry`` (with velocity) plus a ``CameraInfo``
that the simulator never publishes itself. This adapter:

- subscribes to ``odom_raw`` and republishes ``Odometry`` with a velocity
  estimate derived from position deltas;
- publishes the TF chain (world→base_link from odom, base_link→sensor static);
- synthesizes ``CameraInfo`` at a configurable rate from the camera FOV.

JPEG-encoded image streams are decoded by ``JpegLcmTransport`` directly at
the consumer side — no adapter logic needed there.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

from reactivex.disposable import Disposable

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# DimSim camera defaults
_CAM_W = 640
_CAM_H = 288


def _make_camera_info(fov_deg: int) -> CameraInfo:
    """Build CameraInfo for DimSim's virtual camera."""
    fx = (_CAM_W / 2) / math.tan(math.radians(fov_deg / 2))
    fy = fx
    cx, cy = _CAM_W / 2.0, _CAM_H / 2.0

    return CameraInfo(
        frame_id="camera_optical",
        height=_CAM_H,
        width=_CAM_W,
        distortion_model="plumb_bob",
        D=[0.0, 0.0, 0.0, 0.0, 0.0],
        K=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        P=[fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
    )


class DimSimAdapterConfig(ModuleConfig):
    """Configuration for the DimSim → nav-stack type adapter."""

    camera_fov: int = 46
    camera_offset_x: float = 0.3  # camera 30cm forward of base_link
    caminfo_rate_hz: float = 1.0  # how often to republish camera_info


class DimSimAdapter(Module):
    """Convert DimSim's raw outputs into nav-stack-compatible types.

    Inputs (autoconnected to DimSimBridge):
        odom (In[PoseStamped]): raw pose snapshot from DimSim physics.

    Outputs:
        odometry (Out[Odometry]): pose + velocity (computed from deltas),
            frames world→base_link.
        camera_info (Out[CameraInfo]): synthesized at ``caminfo_rate_hz``.

    Also publishes TF: world→base_link (from odom) and base_link→sensor
    (static, at ``camera_offset_x``).
    """

    config: DimSimAdapterConfig

    odom: In[PoseStamped]

    odometry: Out[Odometry]
    camera_info: Out[CameraInfo]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._running = threading.Event()
        self._caminfo_thread: threading.Thread | None = None

        self._prev_odom_time: float | None = None
        self._prev_x = 0.0
        self._prev_y = 0.0
        self._prev_yaw = 0.0

    @rpc
    def start(self) -> None:
        super().start()

        # Force LCMTF construction on the start thread — its lazy property
        # otherwise initializes inside a subscriber callback, which fails
        # silently to publish (same fix DimSimBridge needed pre-NativeModule).
        self.tf  # noqa: B018  -- intentional eager init

        self.register_disposable(Disposable(self.odom.subscribe(self._on_odom)))

        self._running.set()
        self.camera_info.publish(_make_camera_info(self.config.camera_fov))
        self._caminfo_thread = threading.Thread(target=self._caminfo_loop, daemon=True)
        self._caminfo_thread.start()
        logger.info("DimSimAdapter started")

    @rpc
    def stop(self) -> None:
        self._running.clear()
        if self._caminfo_thread:
            self._caminfo_thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        super().stop()

    def _on_odom(self, ps: PoseStamped) -> None:
        """Convert PoseStamped → Odometry, publish TF chain."""
        now = time.time()
        x, y, z = ps.x, ps.y, ps.z
        orient = ps.orientation
        qx, qy, qz, qw = orient.x, orient.y, orient.z, orient.w

        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        vx = vy = vyaw = 0.0
        if self._prev_odom_time is not None:
            dt = now - self._prev_odom_time
            if dt > 0.001:
                dx = x - self._prev_x
                dy = y - self._prev_y
                cos_yaw = math.cos(yaw)
                sin_yaw = math.sin(yaw)
                vx = (dx * cos_yaw + dy * sin_yaw) / dt
                vy = (-dx * sin_yaw + dy * cos_yaw) / dt
                dyaw = yaw - self._prev_yaw
                while dyaw > math.pi:
                    dyaw -= 2 * math.pi
                while dyaw < -math.pi:
                    dyaw += 2 * math.pi
                vyaw = dyaw / dt

        self._prev_odom_time = now
        self._prev_x = x
        self._prev_y = y
        self._prev_yaw = yaw

        pose = Pose()
        pose.position = Vector3(x, y, z)
        pose.orientation = Quaternion(qx, qy, qz, qw)
        odom_twist = Twist()
        odom_twist.linear = Vector3(vx, vy, 0.0)
        odom_twist.angular = Vector3(0.0, 0.0, vyaw)
        self.odometry.publish(
            Odometry(
                ts=ps.ts,
                frame_id="world",
                child_frame_id="base_link",
                pose=pose,
                twist=odom_twist,
            )
        )

        self.tf.publish(
            Transform(
                ts=ps.ts, parent_frame_id="world", child_frame_id="base_link",
                translation=Vector3(x, y, z),
                rotation=Quaternion(qx, qy, qz, qw),
            )
        )
        self.tf.publish(
            Transform(
                ts=ps.ts, parent_frame_id="base_link", child_frame_id="sensor",
                translation=Vector3(self.config.camera_offset_x, 0.0, 0.0),
                rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
            )
        )

    def _caminfo_loop(self) -> None:
        period = 1.0 / max(self.config.caminfo_rate_hz, 0.01)
        while self._running.is_set():
            self.camera_info.publish(_make_camera_info(self.config.camera_fov))
            time.sleep(period)


sim_adapter = DimSimAdapter.blueprint

__all__ = ["DimSimAdapter", "DimSimAdapterConfig", "sim_adapter"]
