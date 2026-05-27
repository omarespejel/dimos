#!/usr/bin/env python3
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

from pathlib import Path
import time
from typing import Any

from reactivex.disposable import Disposable

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.stream import In
from dimos.hardware.sensors.lidar.fastlio2.module import FastLio2
from dimos.memory2.module import Recorder, RecorderConfig
from dimos.memory2.stream import Stream
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.robot.unitree.go2.connection import GO2Connection
from dimos.robot.unitree.keyboard_teleop import KeyboardTeleop
from dimos.utils.logging_config import setup_logger

# dimos --no-obstacle-avoidance --robot-ip 192.168.124.177 run unitree-go2-record
logger = setup_logger()

# FastLIO ports stamped by the C++ binary with hardware clock; must be
# overridden to time.time() so they align with color_image (also time.time()).
_FASTLIO_PORTS = frozenset({"lidar", "odometry"})


class Go2Mid360MemoryConfig(RecorderConfig):
    db_path: str | Path = "recording_go2_mid360.db"
    default_frame_id: str = "base_link"


class Go2Mid360Memory(Recorder):
    """Records Go2 camera, native Go2 lidar, Mid-360 lidar, FastLio2 odometry, and Go2 leg odometry."""

    config: Go2Mid360MemoryConfig

    color_image: In[Image]
    lidar: In[PointCloud2]
    odom: In[PoseStamped]
    fastlio_lidar: In[PointCloud2]
    fastlio_odometry: In[Odometry]

    # def _port_to_stream(self, name: str, input_topic: In[Any], stream: Stream[Any]) -> None:
    #     if name not in _FASTLIO_PORTS:
    #         super()._port_to_stream(name, input_topic, stream)
    #         return

    #     # Force time.time() so FastLIO hardware timestamps match image timestamps.
    #     default_frame_id = self.config.default_frame_id
    #     tf_tolerance = self.config.tf_tolerance

    #     def on_msg(msg: Any) -> None:
    #         ts = time.time()
    #         msg_ts = getattr(msg, "ts", None) or ts
    #         frame_id = (
    #             getattr(msg, "child_frame_id", None)
    #             or getattr(msg, "frame_id", None)
    #             or default_frame_id
    #         )
    #         if frame_id == "world":
    #             frame_id = default_frame_id
    #         transform = self.tf.get(
    #             "world", frame_id, time_point=msg_ts, time_tolerance=tf_tolerance
    #         )
    #         pose = transform.to_pose() if transform is not None else None
    #         if not pose:
    #             logger.warning(
    #                 "[%s] No tf available for frame '%s' at time %s (msg ts: %s), storing without pose",
    #                 name,
    #                 frame_id,
    #                 msg_ts,
    #                 getattr(msg, "ts", None),
    #             )
    #         stream.append(msg, ts=ts, pose=pose)

    #     self.register_disposable(Disposable(input_topic.subscribe(on_msg)))

    def _port_to_stream(self, name: str, input_topic: In[Any], stream: Stream[Any]) -> None:
        """Append each message from *input_topic* to *stream*, attaching world pose via tf.

        Stamped messages use their own ``.frame_id`` and ``.ts``; unstamped
        messages (or ones whose frame isn't in the tf graph, e.g. a payload
        already in world coords) fall back to ``config.default_frame_id`` —
        so every observation gets a robot-pose anchor when tf is publishing.

        Registers the subscription as a disposable on this module.
        """

        default_frame_id = self.config.default_frame_id
        tf_tolerance = self.config.tf_tolerance

        def on_msg(msg: Any) -> None:
            # Force system time for all messages
            ts = time.time()
            frame_id = (
                getattr(msg, "child_frame_id", None)
                or getattr(msg, "frame_id", None)
                or default_frame_id
            )
            transform = self.tf.get("world", frame_id, time_point=ts, time_tolerance=tf_tolerance)
            pose = transform.to_pose() if transform is not None else None

            stream.append(msg, ts=ts, pose=pose)

        self.register_disposable(Disposable(input_topic.subscribe(on_msg)))


unitree_go2_record = autoconnect(
    GO2Connection.blueprint(),
    KeyboardTeleop.blueprint(),
    # unitree_go2_basic.remappings(
    #     [
    #         # (GO2Connection, "lidar", "go2_lidar"),
    #     ]
    # ),
    MovementManager.blueprint(),
    FastLio2.blueprint(
        # host_ip=os.getenv("LIDAR_HOST_IP", "192.168.123.164"),
        # lidar_ip=os.getenv("LIDAR_IP", "192.168.123.120"),
        # mount=G1.internal_odom_offsets["mid360_link"],
        # map_freq=1.0,
        # config="default.yaml",
    ).remappings(
        [
            # (FastLio2, "global_map", "global_map_fastlio"),
            (FastLio2, "lidar", "fastlio_lidar"),
            (FastLio2, "odometry", "fastlio_odometry"),
        ]
    ),
    Go2Mid360Memory.blueprint(),
).global_config(n_workers=10, robot_model="unitree_go2")

# @ivan if "why not using _FASTLIO_PORTS", b/c there's a timestamp issue:
# Error in LCM handling: UNIQUE constraint failed: odom.ts
# Traceback (most recent call last):
#   File "/Users/jeffhykin/repos/dimos/dimos/protocol/service/lcmservice.py", line 135, in _lcm_loop
#     self.l.handle_timeout(_LCM_LOOP_TIMEOUT)
#   File "/Users/jeffhykin/repos/dimos/dimos/protocol/pubsub/impl/lcmpubsub.py", line 122, in <lambda>
#     lcm_subscription = self.l.subscribe(topic_str, lambda _, msg: callback(msg, topic))
#                                                                   ^^^^^^^^^^^^^^^^^^^^
#   File "/Users/jeffhykin/repos/dimos/dimos/protocol/pubsub/encoders.py", line 82, in wrapper_cb
#     callback(decoded_message, topic)
#   File "/Users/jeffhykin/repos/dimos/dimos/core/transport.py", line 141, in <lambda>
#     return self.lcm.subscribe(self.topic, lambda msg, topic: callback(msg))  # type: ignore[arg-type]
#                                                              ^^^^^^^^^^^^^
#   File "/Users/jeffhykin/repos/dimos/dimos/memory2/module.py", line 320, in on_msg
#     stream.append(msg, ts=ts, pose=pose)
#   File "/Users/jeffhykin/repos/dimos/dimos/memory2/stream.py", line 525, in append
#     return self._source.append(obs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/jeffhykin/repos/dimos/dimos/memory2/backend.py", line 109, in append
#     row_id = self.metadata_store.insert(obs)
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/jeffhykin/repos/dimos/dimos/memory2/observationstore/sqlite.py", line 368, in insert
#     cur = self._conn.execute(
#           ^^^^^^^^^^^^^^^^^^^
# sqlite3.IntegrityError: UNIQUE constraint failed: odom.ts
