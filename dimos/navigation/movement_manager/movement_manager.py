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

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

OperatorEvent = tuple[Literal["click", "teleop", "stop"], PointStamped | Twist | None, int]

# without this you can (basically) click into infinity in rerun (not good for the planner)
MAX_CLICK_HORIZONTAL_M = 500.0
MAX_CLICK_VERTICAL_M = 50.0


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


def _twist_within_limits(
    twist: Twist, linear_limit: float | None, angular_limit: float | None
) -> bool:
    """Return whether every Twist component is finite and within configured limits."""
    if not _twist_is_finite(twist):
        return False
    for vector, limit in ((twist.linear, linear_limit), (twist.angular, angular_limit)):
        if limit is None:
            continue
        if not math.isfinite(limit) or limit < 0.0:
            return False
        if any(abs(value) > limit for value in vector.as_tuple):
            return False
    return True


class MovementManagerConfig(ModuleConfig):
    tele_cooldown_sec: float = 1.0
    tele_cmd_vel_scaling: Twist = Twist(Vector3(1, 1, 1), Vector3(1, 1, 1))
    # None preserves the historical finite-command behavior. Supervised teleop
    # deployments should set limits for their robot and operating environment.
    max_teleop_linear_speed: float | None = None
    max_teleop_angular_speed: float | None = None
    planning_frame_id: str = "map"
    # mixed preserves the teleop cooldown mux; manual_only rejects all planner velocity.
    control_mode: Literal["mixed", "manual_only"] = "mixed"
    # Keep an explicit viewer STOP active for the rest of the module lifetime.
    # Safe rearming requires an ordered control protocol rather than timestamps
    # carried on independent streams.
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
        self._transition_generation = 0
        self._stop_transition_active = False
        self._output_transition_active = False
        self._output_transition_owner_ident: int | None = None
        self._pending_operator_event: OperatorEvent | None = None
        self._pending_safe_zero = False
        self._pending_nav: Twist | None = None
        self._stopping = False
        self._lifecycle_stop_pending = False
        self._stop_error: BaseException | None = None
        self._stop_complete = threading.Event()
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
        should_drain = False
        wait_for_stop = False
        current_ident = threading.get_ident()
        with self._lock:
            if self._stopping:
                if self._stop_complete.is_set():
                    return
                # The dispatcher cannot wait for itself while inside a
                # synchronous output subscriber. It will finalize STOP as soon
                # as that broadcast unwinds.
                if self._output_transition_owner_ident == current_ident:
                    return
                wait_for_stop = True
            else:
                self._stopping = True
                self._stop_error = None
                self._stop_complete.clear()
                self._teleop_active = False
                self._operator_stop_latched = False
                self._transition_generation += 1
                self._stop_transition_active = False
                self._clear_pending_locked()
                self._lifecycle_stop_pending = True
                if not self._output_transition_active:
                    self._begin_output_transition_locked()
                    should_drain = True
                elif self._output_transition_owner_ident != current_ident:
                    wait_for_stop = True

        if should_drain:
            self._drain_pending_events()

        if wait_for_stop and not self._stop_complete.wait(timeout=DEFAULT_THREAD_JOIN_TIMEOUT):
            raise TimeoutError("MovementManager stop did not complete")
        if wait_for_stop:
            with self._lock:
                stop_error = self._stop_error
            if stop_error is not None:
                raise RuntimeError("MovementManager stop failed") from stop_error

    def _on_click(self, msg: PointStamped) -> None:
        # PointStamped is mutable and may be reused by its publisher. Snapshot
        # first, then validate and queue that same private value.
        click = PointStamped(
            ts=msg.ts,
            frame_id=msg.frame_id,
            x=msg.x,
            y=msg.y,
            z=msg.z,
        )
        if not all(math.isfinite(v) for v in (click.x, click.y, click.z)):
            logger.warning("Ignored invalid click", x=click.x, y=click.y, z=click.z)
            return
        if (
            abs(click.x) > MAX_CLICK_HORIZONTAL_M
            or abs(click.y) > MAX_CLICK_HORIZONTAL_M
            or abs(click.z) > MAX_CLICK_VERTICAL_M
        ):
            logger.warning("Ignored out-of-range click", x=click.x, y=click.y, z=click.z)
            return
        if not math.isfinite(click.ts) or click.ts < 0.0:
            logger.warning("Ignored click with invalid timestamp", ts=click.ts)
            return

        # Timestamps are downstream metadata only; they do not rearm the
        # terminal operator STOP latch.
        click_frame_id = _canonical_frame_id(click.frame_id)
        if click_frame_id != self._planning_frame_id:
            logger.warning(
                "Ignored click from a mismatched coordinate frame",
                expected=self._planning_frame_id,
                received=click.frame_id,
            )
            return
        click.frame_id = self._planning_frame_id
        self._queue_operator_event("click", click)

    def _on_nav(self, msg: Twist) -> None:
        # Twist is mutable and may be reused by its publisher. Snapshot first,
        # then validate and queue that same private value.
        nav = Twist(msg)
        # Navigation owns a different speed envelope than keyboard teleop, but
        # no planner should be able to forward NaN or infinity to cmd_vel.
        if not _twist_is_finite(nav):
            logger.warning("Stopped on non-finite navigation command")
            self._queue_safe_zero()
            return
        self._queue_nav(nav)

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

        if not input_is_valid or not scaled_is_valid:
            logger.warning(
                "Rejected invalid or out-of-range teleop command",
                max_linear=linear_limit,
                max_angular=angular_limit,
            )
            self._queue_safe_zero()
            return
        self._queue_operator_event("teleop", scaled)

    def _on_teleop_stop(self, msg: Bool) -> None:
        if not self.config.latch_teleop_stop or not msg.data:
            return

        self._queue_operator_event("stop", None)

    def _queue_operator_event(
        self,
        kind: Literal["click", "teleop", "stop"],
        payload: PointStamped | Twist | None,
    ) -> None:
        should_drain = False
        with self._lock:
            if self._stopping:
                return
            if kind == "stop":
                if self._stop_transition_active:
                    return
                self._operator_stop_latched = True
                self._teleop_active = True
                self._last_teleop_time = time.monotonic()
            elif self._operator_stop_latched:
                logger.warning("Ignored operator command while operator STOP is latched")
                return

            # Accepted operator actions supersede an in-flight multi-output
            # transition immediately. The replacement itself is not published
            # until the current synchronous Out.publish() broadcast unwinds.
            self._transition_generation += 1
            generation = self._transition_generation
            self._pending_operator_event = (kind, payload, generation)
            self._pending_safe_zero = False
            self._pending_nav = None
            if not self._output_transition_active:
                self._begin_output_transition_locked()
                should_drain = True

        if should_drain:
            self._drain_pending_events()

    def _queue_safe_zero(self) -> None:
        should_drain = False
        with self._lock:
            if self._stopping or self._operator_stop_latched:
                return
            self._pending_safe_zero = True
            self._pending_nav = None
            if not self._output_transition_active:
                self._begin_output_transition_locked()
                should_drain = True

        if should_drain:
            self._drain_pending_events()

    def _queue_nav(self, msg: Twist) -> None:
        should_drain = False
        with self._lock:
            if (
                self._stopping
                or self._operator_stop_latched
                or self.config.control_mode == "manual_only"
            ):
                return
            self._pending_nav = msg
            if not self._output_transition_active:
                self._begin_output_transition_locked()
                should_drain = True

        if should_drain:
            self._drain_pending_events()

    def _drain_pending_events(self) -> None:
        """Serialize outputs and defer reentrant inputs until a broadcast unwinds."""
        pending_error: BaseException | None = None

        while True:
            lifecycle_stop = False
            with self._lock:
                if self._stopping:
                    self._clear_pending_locked()
                    if self._lifecycle_stop_pending:
                        self._lifecycle_stop_pending = False
                        lifecycle_stop = True
                    else:
                        self._finish_output_transition_locked()
                        break
                    event = None
                    safe_zero = False
                    nav = None
                else:
                    event = self._pending_operator_event
                    self._pending_operator_event = None
                    safe_zero = False
                    nav = None
                    if event is None:
                        if self._pending_safe_zero:
                            safe_zero = True
                            self._pending_safe_zero = False
                        elif self._pending_nav is not None:
                            nav = self._pending_nav
                            self._pending_nav = None
                        else:
                            self._finish_output_transition_locked()
                            break

            if lifecycle_stop:
                try:
                    self._process_lifecycle_stop()
                except BaseException as exc:
                    if pending_error is None:
                        pending_error = exc
                break

            try:
                if event is not None:
                    self._process_operator_event(event)
                elif safe_zero:
                    self.cmd_vel.publish(Twist.zero())
                else:
                    assert nav is not None
                    self._process_nav(nav)
            except BaseException as exc:
                if pending_error is None:
                    pending_error = exc
                with self._lock:
                    lifecycle_pending = self._stopping and self._lifecycle_stop_pending
                    pending = self._pending_operator_event
                    if not lifecycle_pending and (pending is None or pending[0] != "stop"):
                        self._clear_pending_locked()
                        self._finish_output_transition_locked()
                        break

        if pending_error is not None:
            raise pending_error

    def _process_lifecycle_stop(self) -> None:
        """Publish terminal zero, then close the module from the dispatcher."""
        stop_error: BaseException | None = None
        try:
            # Out.publish() invokes local subscribers synchronously. The
            # dispatcher reaches this only after any older broadcast unwinds,
            # and never holds the state lock while publishing.
            self.cmd_vel.publish(Twist.zero())
        except BaseException as exc:
            stop_error = exc

        try:
            super().stop()
        except BaseException as exc:
            if stop_error is None:
                stop_error = exc
        finally:
            with self._lock:
                self._stop_error = stop_error
                self._finish_output_transition_locked()
                self._stop_complete.set()

        if stop_error is not None:
            raise stop_error

    def _process_operator_event(self, event: OperatorEvent) -> None:
        kind, payload, generation = event
        if kind == "click":
            assert isinstance(payload, PointStamped)
            self._process_click(payload, generation)
        elif kind == "teleop":
            assert isinstance(payload, Twist)
            self._process_teleop(payload, generation)
        else:
            self._process_operator_stop(generation)

    def _process_click(self, msg: PointStamped, generation: int) -> None:
        if not self._transition_is_current(generation):
            return
        logger.debug("Goal", x=round(msg.x, 1), y=round(msg.y, 1), z=round(msg.z, 1))
        self.way_point.publish(msg)
        if not self._transition_is_current(generation):
            return
        self.goal.publish(msg)

    def _process_teleop(self, msg: Twist, generation: int) -> None:
        with self._lock:
            if not self._transition_is_current_locked(generation):
                return
            self._teleop_active = True
            self._last_teleop_time = time.monotonic()

        if not self._publish_cancel_for_transition(generation):
            return
        if self._transition_is_current(generation):
            self.cmd_vel.publish(msg)

    def _process_operator_stop(self, generation: int) -> None:
        with self._lock:
            if not self._transition_is_current_locked(generation, allow_latched=True):
                return
            self._stop_transition_active = True

        try:
            self._publish_cancel_for_transition(generation, allow_latched=True)
        finally:
            try:
                # STOP is terminal and zero is last even if cancellation fails.
                self.cmd_vel.publish(Twist.zero())
            finally:
                with self._lock:
                    self._stop_transition_active = False

    def _process_nav(self, msg: Twist) -> None:
        with self._lock:
            if (
                self._stopping
                or self._operator_stop_latched
                or self.config.control_mode == "manual_only"
            ):
                return
            if self._teleop_active:
                elapsed = time.monotonic() - self._last_teleop_time
                if elapsed < self.config.tele_cooldown_sec:
                    return
                self._teleop_active = False

        self.cmd_vel.publish(msg)

    def _publish_cancel_for_transition(
        self, generation: int, *, allow_latched: bool = False
    ) -> bool:
        cancel = PointStamped(
            ts=time.time(),
            frame_id=self._planning_frame_id,
            x=float("nan"),
            y=float("nan"),
            z=float("nan"),
        )
        self.stop_movement.publish(Bool(data=True))
        if not self._transition_is_current(generation, allow_latched=allow_latched):
            return False
        self.way_point.publish(cancel)
        if not self._transition_is_current(generation, allow_latched=allow_latched):
            return False
        self.goal.publish(cancel)
        if not self._transition_is_current(generation, allow_latched=allow_latched):
            return False
        logger.debug("Navigation cancelled — waiting for new goal")
        return True

    def _transition_is_current(self, generation: int, *, allow_latched: bool = False) -> bool:
        with self._lock:
            return self._transition_is_current_locked(generation, allow_latched=allow_latched)

    def _transition_is_current_locked(
        self, generation: int, *, allow_latched: bool = False
    ) -> bool:
        return (
            not self._stopping
            and generation == self._transition_generation
            and (allow_latched or not self._operator_stop_latched)
        )

    def _clear_pending_locked(self) -> None:
        self._pending_operator_event = None
        self._pending_safe_zero = False
        self._pending_nav = None

    def _begin_output_transition_locked(self) -> None:
        self._output_transition_active = True
        self._output_transition_owner_ident = threading.get_ident()

    def _finish_output_transition_locked(self) -> None:
        self._output_transition_active = False
        self._output_transition_owner_ident = None
