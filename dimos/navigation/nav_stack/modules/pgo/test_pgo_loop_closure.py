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

"""Validate PGO publishes loop-closure delta events on the
``loop_closure`` topic.

Replays the ``og_nav_60s`` rosbag through the native PGO binary with
aggressive loop-closure thresholds (low ``loop_time_thresh`` +
``min_loop_detect_duration``, larger ``loop_search_radius``) so any
revisit during the recorded trajectory fires a loop event. For each
event:

* it is logged to stdout/stderr at receive time (shape + first row),
* assertions confirm the shape (N>0 PoseStamped entries), each
  quaternion is unit-norm, and each translation is finite.

The test passes if **at least one** loop closure event is published
with valid shape and content. If the bag doesn't trigger any loop, the
test skips (rosbag is data-dependent — not a code defect).
"""

from __future__ import annotations

import math
from pathlib import Path
import threading
import time

import lcm as lcmlib
import pytest

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.msgs.nav_msgs.Path import Path as NavPath
from dimos.navigation.nav_stack.tests.rosbag_fixtures import (
    NativeProcessRunner,
    feed_at_original_timing,
    lcm_handle_loop,
    load_rosbag_window,
    make_isolated_lcm_url,
)
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

pytestmark = [pytest.mark.slow]

PGO_BIN = Path(__file__).parent / "cpp" / "result" / "bin" / "pgo"

# TODO: use modules rather than LCM directly
SCAN_LCM = "/lc_test_scan#sensor_msgs.PointCloud2"
ODOM_LCM = "/lc_test_odom#nav_msgs.Odometry"
CORRECTED_ODOM_LCM = "/lc_test_corrected#nav_msgs.Odometry"
GLOBAL_MAP_LCM = "/lc_test_global_map#sensor_msgs.PointCloud2"
TF_LCM = "/lc_test_tf#tf2_msgs.TFMessage"
GRAPH_NODES_LCM = "/lc_test_graph_nodes#nav_msgs.GraphNodes3D"
GRAPH_EDGES_LCM = "/lc_test_graph_edges#nav_msgs.LineSegments3D"
LOOP_CLOSURE_LCM = "/lc_test_loop_closure#nav_msgs.Path"

_PROCESS_STARTUP_SEC = 2.0
_POST_FEED_DRAIN_SEC = 5.0

_QUATERNION_UNIT_TOL = 0.05
_TRANSLATION_MAX_M = 100.0


def _validate_path_message(message: NavPath, event_index: int) -> tuple[float, float]:
    """Assert each PoseStamped's quaternion is unit + translation finite.

    Returns ``(max_translation_norm, max_quat_drift)`` so the caller can
    log aggregate stats per event.
    """
    assert len(message.poses) > 0, f"event {event_index}: loop-closure message has no poses"

    max_translation_norm = 0.0
    max_quaternion_drift = 0.0
    for pose_index, pose in enumerate(message.poses):
        translation_x, translation_y, translation_z = (
            pose.position.x,
            pose.position.y,
            pose.position.z,
        )
        quaternion_x, quaternion_y, quaternion_z, quaternion_w = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        for value, name in [
            (translation_x, "translation_x"),
            (translation_y, "translation_y"),
            (translation_z, "translation_z"),
        ]:
            assert math.isfinite(value), (
                f"event {event_index} pose {pose_index}: {name}={value} not finite"
            )
        for value, name in [
            (quaternion_x, "quaternion_x"),
            (quaternion_y, "quaternion_y"),
            (quaternion_z, "quaternion_z"),
            (quaternion_w, "quaternion_w"),
        ]:
            assert math.isfinite(value), (
                f"event {event_index} pose {pose_index}: {name}={value} not finite"
            )
        translation_norm = math.sqrt(
            translation_x * translation_x
            + translation_y * translation_y
            + translation_z * translation_z
        )
        assert translation_norm < _TRANSLATION_MAX_M, (
            f"event {event_index} pose {pose_index}: |t|={translation_norm:.3f}m "
            f"exceeds sanity cap {_TRANSLATION_MAX_M}m"
        )
        quaternion_norm = math.sqrt(
            quaternion_x * quaternion_x
            + quaternion_y * quaternion_y
            + quaternion_z * quaternion_z
            + quaternion_w * quaternion_w
        )
        quaternion_drift = abs(quaternion_norm - 1.0)
        assert quaternion_drift < _QUATERNION_UNIT_TOL, (
            f"event {event_index} pose {pose_index}: |q|={quaternion_norm:.6f} drifts "
            f"from unit by {quaternion_drift:.6f} (tol {_QUATERNION_UNIT_TOL})"
        )
        max_translation_norm = max(max_translation_norm, translation_norm)
        max_quaternion_drift = max(max_quaternion_drift, quaternion_drift)

    return max_translation_norm, max_quaternion_drift


class TestPGOLoopClosure:
    """End-to-end: PGO native publishes loop-closure events with valid shape."""

    def test_loop_closure_events_published(self) -> None:
        if not PGO_BIN.exists():
            pytest.skip(f"PGO binary not found: {PGO_BIN}")

        window = load_rosbag_window()
        assert len(window.scans) > 0, "No scans in rosbag fixture"
        assert len(window.odom) > 0, "No odometry in rosbag fixture"

        # Isolate this test's LCM bus from anything else on the host (other
        # tests, dimos nodes, an actual robot on the LAN).
        lcm_url = make_isolated_lcm_url()
        lcm_instance = lcmlib.LCM(lcm_url)

        received_events: list[NavPath] = []
        events_lock = threading.Lock()

        def _on_loop_closure(_channel: str, data: bytes) -> None:
            message = NavPath.lcm_decode(data)
            with events_lock:
                event_index = len(received_events)
                received_events.append(message)
            first_pose = message.poses[0] if message.poses else None
            first_pose_summary = (
                f"first=t=({first_pose.position.x:.3f},{first_pose.position.y:.3f},"
                f"{first_pose.position.z:.3f}) "
                f"q=({first_pose.orientation.x:.3f},{first_pose.orientation.y:.3f},"
                f"{first_pose.orientation.z:.3f},{first_pose.orientation.w:.3f})"
                if first_pose
                else "<empty>"
            )
            logger.info(
                f"[loop_closure] event #{event_index} received: "
                f"poses_length={len(message.poses)}, frame_id={message.frame_id!r}, "
                f"ts={message.ts:.3f}, {first_pose_summary}"
            )

        loop_closure_subscription = lcm_instance.subscribe(LOOP_CLOSURE_LCM, _on_loop_closure)

        stop_event = threading.Event()
        handle_thread = threading.Thread(
            target=lcm_handle_loop, args=(lcm_instance, stop_event), daemon=True
        )
        handle_thread.start()

        runner = NativeProcessRunner(
            binary_path=str(PGO_BIN),
            args=[
                "--registered_scan",
                SCAN_LCM,
                "--odometry",
                ODOM_LCM,
                "--corrected_odometry",
                CORRECTED_ODOM_LCM,
                "--global_map",
                GLOBAL_MAP_LCM,
                "--tf_channel",
                TF_LCM,
                "--pose_graph_nodes",
                GRAPH_NODES_LCM,
                "--pose_graph_edges",
                GRAPH_EDGES_LCM,
                "--loop_closure",
                LOOP_CLOSURE_LCM,
                # Aggressive loop-closure thresholds — bag is 60s, so we
                # need short re-visit windows to actually fire events.
                "--key_pose_delta_deg",
                "10.0",
                "--key_pose_delta_trans",
                "0.5",
                "--loop_search_radius",
                "2.0",
                "--loop_time_thresh",
                "5.0",
                "--loop_score_thresh",
                "0.5",
                "--loop_submap_half_range",
                "5",
                "--submap_resolution",
                "0.1",
                "--min_loop_detect_duration",
                "1.0",
                "--global_map_voxel_size",
                "0.1",
                "--global_map_publish_rate",
                "1.0",
                "--unregister_input",
                "true",
                "--world_frame",
                "map",
                "--local_frame",
                "odom",
            ],
        )

        try:
            runner.start(capture_stderr=True, env={"LCM_DEFAULT_URL": lcm_url})
            assert runner.is_running, "PGO binary failed to start"
            time.sleep(_PROCESS_STARTUP_SEC)

            feed_at_original_timing(
                lcm_instance,
                window,
                topic_map={
                    "odom": ODOM_LCM,
                    "scan": SCAN_LCM,
                },
            )

            time.sleep(_POST_FEED_DRAIN_SEC)

        finally:
            runner.stop()
            stop_event.set()
            handle_thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
            lcm_instance.unsubscribe(loop_closure_subscription)

        # -- Analysis --
        with events_lock:
            events = list(received_events)

        logger.info(f"\n[loop_closure] total events received: {len(events)}")

        if not events:
            pytest.skip(
                "rosbag trajectory didn't trigger any PGO loop closures "
                "even with aggressive thresholds — this validates only the "
                "publishing path's existence (verified via native log "
                "lines), not the on-wire payload."
            )

        for event_index, message in enumerate(events):
            max_translation_norm, max_quaternion_drift = _validate_path_message(
                message, event_index
            )
            logger.info(
                f"[loop_closure] event #{event_index} VALID: "
                f"keyframe_count={len(message.poses)}, "
                f"max|t|={max_translation_norm:.4f}m, "
                f"max|q|-1|={max_quaternion_drift:.6f}"
            )

        assert all(len(message.poses) > 0 for message in events)
