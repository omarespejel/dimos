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

"""
MovementManager: click-to-goal relay + teleop/nav velocity mux.

NOTE: this should be majorly updated/reworked when mustafa's trajectory controller lands
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Literal

from dimos_lcm.std_msgs import Bool  # type: ignore[import-untyped]
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# without this you can (basically) click into infinity in rerun (not good for the planner)
MAX_CLICK_HORIZONTAL_M = 500.0
MAX_CLICK_VERTICAL_M = 50.0
DEFAULT_MAX_TELEOP_LINEAR_M_S = 1.0
DEFAULT_MAX_TELEOP_ANGULAR_RAD_S = 2.0


def _canonical_frame_id(frame_id: str) -> str:
    """Normalize a coordinate frame identifier for comparison."""
    return frame_id.strip("/")


def _twist_is_finite(twist: Twist) -> bool:
    """Return whether every Twist component is finite."""
    return all(
        math.isfinite(value)
        for vector in (twist.linear, twist.angular)
        for value in vector.as_tuple
    )


def _twist_within_limits(twist: Twist, linear_limit: float, angular_limit: float) -> bool:
    """Return whether every Twist component is finite and within its limit."""
    if (
        not math.isfinite(linear_limit)
        or not math.isfinite(angular_limit)
        or linear_limit < 0.0
        or angular_limit < 0.0
    ):
        return False
    return _twist_is_finite(twist) and all(
        abs(value) <= limit
        for vector, limit in (
            (twist.linear, linear_limit),
            (twist.angular, angular_limit),
        )
        for value in vector.as_tuple
    )


class MovementManagerConfig(ModuleConfig):
    tele_cooldown_sec: float = 1.0
    tele_cmd_vel_scaling: Twist = Twist(Vector3(1, 1, 1), Vector3(1, 1, 1))
    # Default keyboard teleop peaks at 1 m/s linear and 1.6 rad/s angular
    # under boost. Other control envelopes must opt in explicitly.
    max_teleop_linear_speed: float = DEFAULT_MAX_TELEOP_LINEAR_M_S
    max_teleop_angular_speed: float = DEFAULT_MAX_TELEOP_ANGULAR_RAD_S
    planning_frame_id: str = "map"
    # mixed preserves the teleop cooldown mux; manual_only rejects all planner velocity.
    control_mode: Literal["mixed", "manual_only"] = "mixed"
    # Keep an explicit viewer STOP active until the operator publishes a valid new goal.
    latch_teleop_stop: bool = False


class MovementManager(Module):
    """Combine tele_cmd_vel (keyboard controls) and nav_cmd_vel in a sane way, output cmd_vel"""

    config: MovementManagerConfig

    clicked_point: In[PointStamped]
    nav_cmd_vel: In[Twist]
    tele_cmd_vel: In[Twist]
    teleop_stop: In[Bool]

    goal: Out[PointStamped]
    way_point: Out[PointStamped]
    cmd_vel: Out[Twist]
    stop_movement: Out[Bool]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.RLock()
        self._teleop_active = False
        self._last_teleop_time = 0.0
        self._operator_stop_latched = False
        self._operator_stop_ts: float | None = None
        self._last_goal: tuple[float, float, float, float] | None = None
        self._transition_generation = 0
        self._stop_transition_active = False
        self._stopping = False
        self._planning_frame_id = _canonical_frame_id(self.config.planning_frame_id)
        if not self._planning_frame_id:
            raise ValueError("planning_frame_id must not be empty")

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.clicked_point.subscribe(self._on_click)))
        self.register_disposable(Disposable(self.nav_cmd_vel.subscribe(self._on_nav)))
        self.register_disposable(Disposable(self.tele_cmd_vel.subscribe(self._on_teleop)))
        self.register_disposable(Disposable(self.teleop_stop.subscribe(self._on_teleop_stop)))

    @rpc
    def stop(self) -> None:
        first_stop = False
        try:
            with self._lock:
                if self._stopping:
                    return
                # This gate is terminal for the module instance. Set it only
                # after any in-flight callback releases the lock, and before
                # the final zero is published or subscriptions are torn down.
                self._stopping = True
                first_stop = True
                self._teleop_active = False
                self._operator_stop_latched = False
                self._operator_stop_ts = None
                self._last_goal = None
                self._transition_generation += 1
                self._stop_transition_active = False
                self.cmd_vel.publish(Twist.zero())
        finally:
            if first_stop:
                super().stop()

    def _on_click(self, msg: PointStamped) -> None:
        if not all(math.isfinite(v) for v in (msg.x, msg.y, msg.z)):
            logger.warning("Ignored invalid click", x=msg.x, y=msg.y, z=msg.z)
            return
        if (
            abs(msg.x) > MAX_CLICK_HORIZONTAL_M
            or abs(msg.y) > MAX_CLICK_HORIZONTAL_M
            or abs(msg.z) > MAX_CLICK_VERTICAL_M
        ):
            logger.warning("Ignored out-of-range click", x=msg.x, y=msg.y, z=msg.z)
            return
        if not math.isfinite(msg.ts):
            logger.warning("Ignored click with invalid timestamp", ts=msg.ts)
            return

        # The viewer uses zero when timestamp_ms is absent. Treat receipt of
        # such a click as a new operator action while retaining explicit
        # timestamps for stale/replay protection.
        click_ts = time.time() if msg.ts == 0.0 else msg.ts
        click_frame_id = _canonical_frame_id(msg.frame_id)
        if click_frame_id != self._planning_frame_id:
            logger.warning(
                "Ignored click from a mismatched coordinate frame",
                expected=self._planning_frame_id,
                received=msg.frame_id,
            )
            return
        if msg.frame_id != self._planning_frame_id:
            msg = PointStamped(
                # Use the resolved receipt timestamp so PointStamped does not
                # independently normalize an explicit zero a second time.
                ts=click_ts,
                frame_id=self._planning_frame_id,
                x=msg.x,
                y=msg.y,
                z=msg.z,
            )

        with self._lock:
            if self._stopping:
                return
            if self._stop_transition_active:
                logger.warning("Ignored replacement goal during STOP transition")
                return
            if self._operator_stop_latched:
                baseline_ts = self._operator_stop_ts
                if self._last_goal is not None:
                    baseline_ts = (
                        self._last_goal[0]
                        if baseline_ts is None
                        else max(baseline_ts, self._last_goal[0])
                    )
                if baseline_ts is not None and click_ts <= baseline_ts:
                    logger.warning(
                        "Ignored stale or replayed replacement goal",
                        received_ts=click_ts,
                        last_ts=baseline_ts,
                    )
                    return

            self._transition_generation += 1
            transition_generation = self._transition_generation
            logger.debug("Goal", x=round(msg.x, 1), y=round(msg.y, 1), z=round(msg.z, 1))
            self.way_point.publish(msg)
            if transition_generation != self._transition_generation:
                return
            self.goal.publish(msg)
            if transition_generation != self._transition_generation:
                return

            self._last_goal = (click_ts, msg.x, msg.y, msg.z)

            # The same lock serializes replacement-goal publication with STOP.
            # Keep the latch set during publish so reentrant planner traffic is
            # rejected, then release it only after the goal has reached subscribers.
            if self._operator_stop_latched:
                self._operator_stop_latched = False
                self._operator_stop_ts = None
                self._teleop_active = False

    def _cancel_goal(self) -> None:
        self.stop_movement.publish(Bool(data=True))
        # NOTE: this NaN goal is more of a safety fallback.
        # It can be REALLY bad if a robot is supposed to stop moving but wont
        # we should probably think a more robust/strict requirement on planners
        cancel = PointStamped(
            ts=time.time(),
            frame_id=self._planning_frame_id,
            x=float("nan"),
            y=float("nan"),
            z=float("nan"),
        )
        self.way_point.publish(cancel)
        self.goal.publish(cancel)
        logger.debug("Navigation cancelled — waiting for new goal")

    def _on_nav(self, msg: Twist) -> None:
        with self._lock:
            if self._stopping:
                return
            if self.config.control_mode == "manual_only" or self._operator_stop_latched:
                return
            # Navigation owns a different speed envelope than keyboard teleop,
            # but no planner should be able to forward NaN or infinity to cmd_vel.
            if not _twist_is_finite(msg):
                logger.warning("Ignored non-finite navigation command")
                return
            if self._teleop_active:
                # check if cooldown has expired
                elapsed = time.monotonic() - self._last_teleop_time
                if elapsed < self.config.tele_cooldown_sec:
                    return
                self._teleop_active = False
            self.cmd_vel.publish(msg)

    def _on_teleop(self, msg: Twist) -> None:
        linear_limit = self.config.max_teleop_linear_speed
        angular_limit = self.config.max_teleop_angular_speed
        input_is_valid = _twist_within_limits(msg, linear_limit, angular_limit)
        scale = self.config.tele_cmd_vel_scaling
        scaled = Twist(
            linear=Vector3(
                msg.linear.x * scale.linear.x,
                msg.linear.y * scale.linear.y,
                msg.linear.z * scale.linear.z,
            ),
            angular=Vector3(
                msg.angular.x * scale.angular.x,
                msg.angular.y * scale.angular.y,
                msg.angular.z * scale.angular.z,
            ),
        )
        scaled_is_valid = _twist_within_limits(scaled, linear_limit, angular_limit)

        with self._lock:
            if self._stopping:
                return
            if not input_is_valid or not scaled_is_valid:
                logger.warning(
                    "Rejected invalid or out-of-range teleop command",
                    max_linear=linear_limit,
                    max_angular=angular_limit,
                )
                self.cmd_vel.publish(Twist.zero())
                return
            transition_generation = self._transition_generation
            self._teleop_active = True
            self._last_teleop_time = time.monotonic()

            self._cancel_goal()
            if transition_generation != self._transition_generation:
                return

            self.cmd_vel.publish(scaled)

    def _on_teleop_stop(self, msg: Bool) -> None:
        if not self.config.latch_teleop_stop or not msg.data:
            return

        with self._lock:
            if self._stopping:
                return
            self._transition_generation += 1
            self._operator_stop_latched = True
            # dimos-viewer click timestamps and time.time() both use Unix epoch
            # seconds, so this is a comparable stale-message boundary.
            self._operator_stop_ts = time.time()
            self._teleop_active = True
            self._last_teleop_time = time.monotonic()
            self._stop_transition_active = True
            # Serialize the full STOP transition with replacement-goal
            # publication. The viewer also publishes a zero Twist for consumers
            # that do not use MovementManager.
            try:
                self._cancel_goal()
            finally:
                try:
                    self.cmd_vel.publish(Twist.zero())
                finally:
                    self._stop_transition_active = False
