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

from collections.abc import Callable, Generator
from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any

from dimos_lcm.std_msgs import Bool  # type: ignore[import-untyped]
import pytest

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.stream import Stream, Transport
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.navigation.movement_manager.movement_manager import (
    MovementManager,
)


@dataclass
class Captured:
    """Captures messages published by a MovementManager via real subscribers."""

    cmd_vel: list[Twist] = field(default_factory=list)
    stop_movement: list[Bool] = field(default_factory=list)
    goal: list[PointStamped] = field(default_factory=list)
    way_point: list[PointStamped] = field(default_factory=list)


class _DirectTransport(Transport):  # type: ignore[type-arg]
    """Synchronous transport for exercising subscriptions registered by start()."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[Any], Any]] = []

    def broadcast(self, _selfstream: Any, value: Any) -> None:
        for callback in list(self._subscribers):
            callback(value)

    def subscribe(
        self, callback: Callable[[Any], Any], _selfstream: Stream[Any] | None = None
    ) -> Callable[[], None]:
        self._subscribers.append(callback)

        def unsubscribe() -> None:
            self._subscribers.remove(callback)

        return unsubscribe

    def start(self) -> None: ...

    def stop(self) -> None:
        self._subscribers.clear()


def _attach(module: MovementManager) -> tuple[Captured, list[Callable[[], None]]]:
    """Subscribe to every Out port; return (captured, unsubscribers)."""
    captured = Captured()
    unsubs = [
        module.cmd_vel.subscribe(captured.cmd_vel.append),
        module.stop_movement.subscribe(captured.stop_movement.append),
        module.goal.subscribe(captured.goal.append),
        module.way_point.subscribe(captured.way_point.append),
    ]
    return captured, unsubs


@pytest.fixture()
def manager_and_captured() -> Generator[tuple[MovementManager, Captured], None, None]:
    module = MovementManager(tele_cooldown_sec=0.1)
    captured, unsubs = _attach(module)
    try:
        yield module, captured
    finally:
        for unsub in unsubs:
            unsub()
        module._close_module()


@pytest.fixture()
def started_manager_and_captured() -> Generator[tuple[MovementManager, Captured], None, None]:
    module = MovementManager(latch_teleop_stop=True, tele_cooldown_sec=0.0)
    for input_stream in module.inputs.values():
        input_stream.transport = _DirectTransport()
    captured, unsubs = _attach(module)
    try:
        module.start()
        yield module, captured
    finally:
        for unsub in unsubs:
            unsub()
        module.stop()


def _twist(lx: float = 0.0) -> Twist:
    return Twist(linear=Vector3(lx, 0, 0), angular=Vector3(0, 0, 0))


def _click(
    x: float = 1.0,
    y: float = 2.0,
    z: float = 0.0,
    *,
    ts: float | None = None,
    frame_id: str = "map",
) -> PointStamped:
    return PointStamped(ts=time.time() if ts is None else ts, frame_id=frame_id, x=x, y=y, z=z)


def test_teleop_suppresses_nav_and_cancels_goal(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    """Teleop arriving should suppress nav, publish stop_movement, and cancel the goal with NaN."""
    manager, captured = manager_and_captured
    manager.config.tele_cooldown_sec = 10.0
    manager._on_teleop(_twist(lx=0.3))

    cmd_count_after_teleop = len(captured.cmd_vel)
    manager._on_nav(_twist(lx=0.9))
    # Nav was suppressed: no new cmd_vel
    assert len(captured.cmd_vel) == cmd_count_after_teleop

    # stop_movement fired
    assert len(captured.stop_movement) == 1

    # Goal cancelled with NaN
    assert len(captured.goal) == 1
    assert math.isnan(captured.goal[0].x)


def test_nav_resumes_after_cooldown(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    """After the cooldown expires, nav commands pass through again."""
    manager, captured = manager_and_captured
    manager.config.tele_cooldown_sec = 0.05
    manager._on_teleop(_twist(lx=0.3))
    time.sleep(DEFAULT_THREAD_JOIN_TIMEOUT)
    cmd_count_before = len(captured.cmd_vel)

    manager._on_nav(_twist(lx=0.9))
    assert len(captured.cmd_vel) == cmd_count_before + 1


def test_manual_only_mode_never_forwards_navigation(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = manager_and_captured
    manager.config.control_mode = "manual_only"

    manager._on_nav(_twist(lx=0.9))

    assert captured.cmd_vel == []


def test_manual_only_mode_still_forwards_teleop(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = manager_and_captured
    manager.config.control_mode = "manual_only"

    manager._on_teleop(_twist(lx=0.3))

    assert captured.cmd_vel == [_twist(lx=0.3)]


def test_idle_zero_teleop_does_not_latch_operator_stop(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = manager_and_captured
    manager.config.latch_teleop_stop = True
    manager.config.tele_cooldown_sec = 0.0

    manager._on_teleop(_twist())
    manager._on_nav(_twist(lx=0.9))

    assert captured.cmd_vel == [_twist(), _twist(lx=0.9)]


def test_explicit_stop_is_opt_in(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = manager_and_captured
    manager.config.tele_cooldown_sec = 0.0

    manager._on_teleop_stop(Bool(data=True))
    manager._on_nav(_twist(lx=0.9))

    assert captured.cmd_vel == [_twist(lx=0.9)]


def test_latched_teleop_stop_requires_new_valid_goal(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = manager_and_captured
    manager.config.latch_teleop_stop = True
    manager.config.tele_cooldown_sec = 0.0

    manager._on_teleop_stop(Bool(data=True))
    manager._on_nav(_twist(lx=0.7))
    manager._on_click(_click(x=float("nan")))
    manager._on_nav(_twist(lx=0.8))
    manager._on_click(_click(x=600.0))
    manager._on_nav(_twist(lx=0.85))
    manager._on_click(_click())
    manager._on_nav(_twist(lx=0.9))

    assert captured.cmd_vel == [_twist(), _twist(lx=0.9)]


@pytest.mark.parametrize(
    "replacement",
    [
        _click(x=3.0, ts=99.0),
        _click(x=1.0, y=2.0, ts=100.0),
        _click(x=3.0, ts=100.0),
        _click(x=3.0, ts=float("nan")),
    ],
)
def test_latched_stop_rejects_stale_replayed_or_invalid_goal(
    manager_and_captured: tuple[MovementManager, Captured],
    replacement: PointStamped,
) -> None:
    manager, captured = manager_and_captured
    manager.config.latch_teleop_stop = True
    manager.config.tele_cooldown_sec = 0.0
    manager._on_click(_click(ts=100.0))
    manager._on_teleop_stop(Bool(data=True))
    goal_count = len(captured.goal)
    way_point_count = len(captured.way_point)

    manager._on_click(replacement)
    manager._on_nav(_twist(lx=0.9))

    assert len(captured.goal) == goal_count
    assert len(captured.way_point) == way_point_count
    assert captured.cmd_vel == [_twist()]
    assert manager._operator_stop_latched


def test_latched_stop_accepts_missing_timestamp(
    manager_and_captured: tuple[MovementManager, Captured],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, captured = manager_and_captured
    manager.config.latch_teleop_stop = True
    manager.config.tele_cooldown_sec = 0.0
    timestamps = iter((100.0, 100.5, 101.0))
    monkeypatch.setattr(time, "time", lambda: next(timestamps))

    manager._on_click(_click(ts=0.0, frame_id="/map"))
    manager._on_teleop_stop(Bool(data=True))
    manager._on_click(_click(ts=0.0, frame_id="map"))
    manager._on_nav(_twist(lx=0.9))

    assert captured.cmd_vel == [_twist(), _twist(lx=0.9)]


def test_latched_stop_rejects_goal_from_mismatched_coordinate_frame(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = manager_and_captured
    manager.config.latch_teleop_stop = True
    manager.config.tele_cooldown_sec = 0.0

    manager._on_click(_click(ts=100.0, frame_id="map"))
    manager._on_teleop_stop(Bool(data=True))
    goal_count = len(captured.goal)
    way_point_count = len(captured.way_point)
    manager._on_click(_click(x=3.0, ts=101.0, frame_id="world/lidar"))
    manager._on_nav(_twist(lx=0.9))

    assert len(captured.goal) == goal_count
    assert len(captured.way_point) == way_point_count
    assert captured.cmd_vel == [_twist()]
    assert manager._operator_stop_latched


def test_reentrant_stop_wins_over_replacement_goal(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = manager_and_captured
    manager.config.latch_teleop_stop = True
    manager._on_click(_click(ts=100.0))
    manager._on_teleop_stop(Bool(data=True))
    stop_sent = False

    def stop_during_goal(_msg: PointStamped) -> None:
        nonlocal stop_sent
        if not stop_sent:
            stop_sent = True
            manager._on_teleop_stop(Bool(data=True))

    unsubscribe = manager.way_point.subscribe(stop_during_goal)
    try:
        manager._on_click(_click(x=3.0, ts=101.0))
        manager._on_nav(_twist(lx=0.9))
    finally:
        unsubscribe()

    assert math.isnan(captured.goal[-1].x)
    assert captured.cmd_vel == [_twist(), _twist()]
    assert manager._operator_stop_latched


def test_reentrant_goal_during_stop_is_rejected(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = manager_and_captured
    manager.config.latch_teleop_stop = True
    manager._on_click(_click(ts=100.0))

    def goal_during_stop(_msg: Bool) -> None:
        manager._on_click(_click(x=3.0, ts=101.0))

    unsubscribe = manager.stop_movement.subscribe(goal_during_stop)
    try:
        manager._on_teleop_stop(Bool(data=True))
    finally:
        unsubscribe()

    assert math.isnan(captured.goal[-1].x)
    assert all(goal.x != 3.0 for goal in captured.goal)
    assert captured.cmd_vel == [_twist()]
    assert manager._operator_stop_latched


def test_reentrant_stop_wins_over_teleop_command(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = manager_and_captured
    manager.config.latch_teleop_stop = True
    stop_sent = False

    def stop_during_teleop_cancel(_msg: Bool) -> None:
        nonlocal stop_sent
        if not stop_sent:
            stop_sent = True
            manager._on_teleop_stop(Bool(data=True))

    unsubscribe = manager.stop_movement.subscribe(stop_during_teleop_cancel)
    try:
        manager._on_teleop(_twist(lx=0.3))
    finally:
        unsubscribe()

    assert captured.cmd_vel == [_twist()]
    assert manager._operator_stop_latched


def test_stop_publishes_zero_when_goal_cancellation_fails(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = manager_and_captured
    manager.config.latch_teleop_stop = True

    def fail_stop_subscriber(_msg: Bool) -> None:
        raise RuntimeError("planner subscriber failed")

    unsubscribe = manager.stop_movement.subscribe(fail_stop_subscriber)
    try:
        with pytest.raises(RuntimeError, match="planner subscriber failed"):
            manager._on_teleop_stop(Bool(data=True))
    finally:
        unsubscribe()

    assert captured.cmd_vel == [_twist()]
    assert manager._operator_stop_latched


def test_started_manager_latches_explicit_stop(
    started_manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = started_manager_and_captured

    manager.teleop_stop.transport.publish(Bool(data=True))
    manager.nav_cmd_vel.transport.publish(_twist(lx=0.9))

    assert captured.cmd_vel == [_twist()]


def test_stop_clears_operator_stop_latch(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, _captured = manager_and_captured
    manager.config.latch_teleop_stop = True
    manager._on_teleop_stop(Bool(data=True))

    manager.stop()

    assert not manager._operator_stop_latched


def test_stop_zero_is_final_and_suppresses_waiting_callback(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = manager_and_captured
    manager._on_teleop(_twist(lx=0.3))
    zero_publish_entered = threading.Event()
    release_zero_publish = threading.Event()
    waiting_callback_started = threading.Event()
    waiting_callback_returned = threading.Event()

    def block_final_zero(twist: Twist) -> None:
        if twist.is_zero():
            zero_publish_entered.set()
            assert release_zero_publish.wait(timeout=2.0)

    unsubscribe = manager.cmd_vel.subscribe(block_final_zero)
    stop_thread = threading.Thread(target=manager.stop)

    def publish_waiting_teleop() -> None:
        waiting_callback_started.set()
        manager._on_teleop(_twist(lx=0.7))
        waiting_callback_returned.set()

    waiting_callback = threading.Thread(target=publish_waiting_teleop)
    try:
        stop_thread.start()
        assert zero_publish_entered.wait(timeout=2.0)
        waiting_callback.start()
        assert waiting_callback_started.wait(timeout=2.0)
        assert not waiting_callback_returned.is_set()
        release_zero_publish.set()
        stop_thread.join(timeout=2.0)
        waiting_callback.join(timeout=2.0)
    finally:
        release_zero_publish.set()
        stop_thread.join(timeout=2.0)
        waiting_callback.join(timeout=2.0)
        unsubscribe()

    manager._on_nav(_twist(lx=0.9))
    manager._on_click(_click(x=3.0))
    manager._on_teleop_stop(Bool(data=True))

    assert not stop_thread.is_alive()
    assert not waiting_callback.is_alive()
    assert captured.cmd_vel == [_twist(lx=0.3), _twist()]
    assert captured.cmd_vel[-1].is_zero()


def test_cancel_goal_uses_configured_planning_frame() -> None:
    manager = MovementManager(planning_frame_id="/world")
    captured, unsubs = _attach(manager)
    try:
        manager._cancel_goal()
    finally:
        for unsub in unsubs:
            unsub()
        manager._close_module()

    assert captured.goal[-1].frame_id == "world"
    assert captured.way_point[-1].frame_id == "world"


@pytest.mark.parametrize(
    ("linear", "angular"),
    [
        (Vector3(float("nan"), 0, 0), Vector3()),
        (Vector3(0, float("nan"), 0), Vector3()),
        (Vector3(0, 0, float("nan")), Vector3()),
        (Vector3(), Vector3(float("inf"), 0, 0)),
        (Vector3(), Vector3(0, float("inf"), 0)),
        (Vector3(), Vector3(0, 0, float("inf"))),
        (Vector3(1.01, 0, 0), Vector3()),
        (Vector3(0, -1.01, 0), Vector3()),
        (Vector3(0, 0, 1.01), Vector3()),
        (Vector3(), Vector3(2.01, 0, 0)),
        (Vector3(), Vector3(0, -2.01, 0)),
        (Vector3(), Vector3(0, 0, 2.01)),
    ],
)
def test_invalid_or_out_of_range_teleop_is_rejected_before_state_change(
    manager_and_captured: tuple[MovementManager, Captured],
    linear: Vector3,
    angular: Vector3,
) -> None:
    manager, captured = manager_and_captured

    manager._on_teleop(Twist(linear, angular))

    assert captured.cmd_vel == [_twist()]
    assert captured.stop_movement == []
    assert captured.goal == []
    assert not manager._teleop_active
    assert manager._last_teleop_time == 0.0


def test_teleop_rejects_scaled_output_above_configured_limit(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = manager_and_captured
    manager.config.tele_cmd_vel_scaling = Twist(Vector3(2, 1, 1), Vector3(1, 1, 1))

    manager._on_teleop(_twist(lx=0.6))

    assert captured.cmd_vel == [_twist()]
    assert not manager._teleop_active
    assert manager._last_teleop_time == 0.0


def test_valid_click_publishes_goal(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    """A valid click should publish to both goal and way_point."""
    manager, captured = manager_and_captured
    click = _click(x=5.0, y=3.0, z=0.1)
    manager._on_click(click)
    assert captured.goal == [click]
    assert captured.way_point == [click]


def test_invalid_clicks_rejected(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    """NaN, Inf, and out-of-range clicks should not publish."""
    manager, captured = manager_and_captured
    for bad_click in [
        _click(x=float("nan")),
        _click(x=float("inf")),
        _click(x=600.0),
    ]:
        manager._on_click(bad_click)
    assert captured.goal == []


def test_tele_cmd_vel_scaling(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    """tele_cmd_vel_scaling multiplies each teleop twist component independently."""
    manager, captured = manager_and_captured
    scaling = Twist(Vector3(0.5, 2.0, 0.0), Vector3(1.0, 1.0, 0.25))
    manager.config.tele_cmd_vel_scaling = scaling
    manager.config.max_teleop_linear_speed = 2.0
    manager.config.tele_cooldown_sec = 10.0

    manager._on_teleop(Twist(Vector3(1, 1, 1), Vector3(1, 1, 1)))

    assert len(captured.cmd_vel) == 1
    published = captured.cmd_vel[0]
    assert published.linear.x == pytest.approx(0.5)
    assert published.linear.y == pytest.approx(2.0)
    assert published.linear.z == pytest.approx(0.0)
    assert published.angular.z == pytest.approx(0.25)
