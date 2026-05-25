#!/usr/bin/env python3
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

"""Camera bridge subprocess for X2Connection.

Why this exists: when X2Connection's rclpy subscriber for the head RGB camera
runs inside the dimos worker (a multiprocessing.forkserver child), the FastDDS
reader silently drops messages after the first one or two. A standalone rclpy
subscriber with identical QoS receives the same topic reliably at 10 Hz.

Running this script as a subprocess gives us a fresh Python interpreter and
fresh DDS state, sidestepping whatever forkserver/FastDDS interaction is
breaking the in-worker subscription.

Wire protocol on stdout:
    repeated frames of  [u32 little-endian length] [length bytes JPEG]

The parent reads the stream, decodes, rotates, and publishes to dimos.
"""

from __future__ import annotations

import signal
import struct
import sys


def main() -> None:
    import rclpy
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from sensor_msgs.msg import CompressedImage

    topic = (
        sys.argv[1]
        if len(sys.argv) > 1
        else ("/aima/hal/sensor/rgbd_head_front/rgb_image/compressed")
    )

    rclpy.init()
    node = rclpy.create_node("dimos_x2_camera_bridge")
    qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        durability=QoSDurabilityPolicy.VOLATILE,
    )

    stdout = sys.stdout.buffer

    def on_frame(msg) -> None:  # type: ignore[no-untyped-def]
        payload = bytes(msg.data)
        try:
            stdout.write(struct.pack("<I", len(payload)))
            stdout.write(payload)
            stdout.flush()
        except (BrokenPipeError, OSError):
            # Parent closed; exit the spin loop.
            rclpy.try_shutdown()

    node.create_subscription(CompressedImage, topic, on_frame, qos)

    def _shutdown(*_: object) -> None:
        rclpy.try_shutdown()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        rclpy.spin(node)
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
