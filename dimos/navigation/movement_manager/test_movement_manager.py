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
from typing import Any, TypeVar

from dimos_lcm.std_msgs import Bool  # type: ignore[import-untyped]
import pytest

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.stream import Out, Stream, Transport
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.navigation.movement_manager import movement_manager as movement_manager_module
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


T = TypeVar("T")


class _DirectTransport(Transport[T]):
    """Synchronous transport for exercising subscriptions registered by start()."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[T], Any]] = []

    def broadcast(self, _selfstream: Out[T] | None, value: T) -> None:
        for callback in list(self._subscribers):
            callback(value)

    def subscribe(
        self, callback: Callable[[T], Any], _selfstream: Stream[T] | None = None
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
    click_ts = time.time() if ts is None else ts
    return PointStamped(
        ts=click_ts,
        frame_id=frame_id,
        x=x,
        y=y,
        z=z,
    )


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


@pytest.mark.parametrize(
    ("linear", "angular"),
    [
        (Vector3(float("nan"), 0, 0), Vector3()),
        (Vector3(0, float("inf"), 0), Vector3()),
        (Vector3(0, 0, float("-inf")), Vector3()),
        (Vector3(), Vector3(float("nan"), 0, 0)),
        (Vector3(), Vector3(0, float("inf"), 0)),
        (Vector3(), Vector3(0, 0, float("-inf"))),
    ],
)
def test_non_finite_navigation_is_rejected(
    manager_and_captured: tuple[MovementManager, Captured],
    linear: Vector3,
    angular: Vector3,
) -> None:
    manager, captured = manager_and_captured

    manager._on_nav(Twist(linear, angular))

    assert captured.cmd_vel == [_twist()]


def test_finite_navigation_keeps_planner_owned_speed_envelope(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = manager_and_captured
    planner_twist = _twist(lx=10.0)

    manager._on_nav(planner_twist)

    assert captured.cmd_vel == [planner_twist]


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


def test_latched_teleop_stop_is_terminal_for_module_lifetime(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = manager_and_captured
    manager.config.latch_teleop_stop = True
    manager.config.tele_cooldown_sec = 0.0

    manager._on_teleop_stop(Bool(data=True))
    cmd_count = len(captured.cmd_vel)
    goal_count = len(captured.goal)
    way_point_count = len(captured.way_point)
    stop_count = len(captured.stop_movement)

    manager._on_nav(_twist(lx=0.7))
    manager._on_click(_click(x=float("nan")))
    manager._on_click(_click(x=600.0))
    manager._on_click(_click())
    manager._on_teleop(_twist(lx=0.3))
    manager._on_nav(_twist(lx=0.9))

    assert len(captured.cmd_vel) == cmd_count
    assert len(captured.goal) == goal_count
    assert len(captured.way_point) == way_point_count
    assert len(captured.stop_movement) == stop_count
    assert captured.cmd_vel[-1].is_zero()
    assert manager._operator_stop_latched


@pytest.mark.parametrize("timestamp", [12.0, 1_700_000_001.0, 1_900_000_000.0])
def test_terminal_stop_rejects_click_in_every_timestamp_domain(
    manager_and_captured: tuple[MovementManager, Captured],
    timestamp: float,
) -> None:
    manager, captured = manager_and_captured
    manager.config.latch_teleop_stop = True

    manager._on_teleop_stop(Bool(data=True))
    goal_count = len(captured.goal)
    way_point_count = len(captured.way_point)
    manager._on_click(_click(x=4.0, ts=timestamp))

    assert len(captured.goal) == goal_count
    assert len(captured.way_point) == way_point_count
    assert captured.cmd_vel[-1].is_zero()
    assert manager._operator_stop_latched


def test_accepted_teleop_invalidates_reentrant_outer_goal(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = manager_and_captured
    late_way_points: list[PointStamped] = []
    teleop_sent = False

    def teleop_during_way_point(msg: PointStamped) -> None:
        nonlocal teleop_sent
        if not teleop_sent and not math.isnan(msg.x):
            teleop_sent = True
            manager._on_teleop(_twist(lx=0.3))

    trigger_unsub = manager.way_point.subscribe(teleop_during_way_point)
    late_unsub = manager.way_point.subscribe(late_way_points.append)
    try:
        manager._on_click(_click(x=4.0))
    finally:
        late_unsub()
        trigger_unsub()

    assert all(goal.x != 4.0 for goal in captured.goal)
    assert math.isnan(captured.goal[-1].x)
    assert late_way_points[0].x == 4.0
    assert math.isnan(late_way_points[-1].x)
    assert captured.cmd_vel == [_twist(lx=0.3)]


def test_reentrant_stop_has_priority_over_queued_operator_and_nav_events(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = manager_and_captured
    manager.config.latch_teleop_stop = True
    queued = False

    def queue_everything(_msg: PointStamped) -> None:
        nonlocal queued
        if queued:
            return
        queued = True
        manager._on_click(_click(x=2.0))
        manager._on_teleop(_twist(lx=0.3))
        manager._on_nav(_twist(lx=0.9))
        manager._on_teleop_stop(Bool(data=True))

    unsubscribe = manager.way_point.subscribe(queue_everything)
    try:
        manager._on_click(_click(x=1.0))
    finally:
        unsubscribe()

    assert all(goal.x not in (1.0, 2.0) for goal in captured.goal)
    assert math.isnan(captured.goal[-1].x)
    assert captured.cmd_vel == [_twist()]
    assert manager._operator_stop_latched


def test_reentrant_stop_zero_is_final_for_late_cmd_vel_subscriber(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = manager_and_captured
    manager.config.latch_teleop_stop = True
    late_cmd_vel: list[Twist] = []
    stop_sent = False

    def stop_during_nonzero(msg: Twist) -> None:
        nonlocal stop_sent
        if not stop_sent and not msg.is_zero():
            stop_sent = True
            manager._on_teleop_stop(Bool(data=True))

    trigger_unsub = manager.cmd_vel.subscribe(stop_during_nonzero)
    late_unsub = manager.cmd_vel.subscribe(late_cmd_vel.append)
    try:
        manager._on_teleop(_twist(lx=0.3))
    finally:
        late_unsub()
        trigger_unsub()

    assert captured.cmd_vel == [_twist(lx=0.3), _twist()]
    assert late_cmd_vel == [_twist(lx=0.3), _twist()]
    assert captured.cmd_vel[-1].is_zero()
    assert late_cmd_vel[-1].is_zero()
    assert manager._operator_stop_latched


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


def test_stop_zero_is_final_without_holding_state_lock_during_publish(
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
        assert waiting_callback_returned.wait(timeout=2.0)
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
    manager.config.latch_teleop_stop = True
    manager._on_teleop_stop(Bool(data=True))

    assert manager._stopping
    assert not stop_thread.is_alive()
    assert not waiting_callback.is_alive()
    assert captured.cmd_vel == [_twist(lx=0.3), _twist()]
    assert captured.cmd_vel[-1].is_zero()


@pytest.mark.parametrize("source", ["teleop", "nav"])
def test_lifecycle_stop_waits_for_inflight_velocity_then_zero_is_final(
    manager_and_captured: tuple[MovementManager, Captured],
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    manager, captured = manager_and_captured
    manager.config.tele_cooldown_sec = 0.0
    publish_entered = threading.Event()
    release_publish = threading.Event()
    stop_returned = threading.Event()
    worker_errors: list[BaseException] = []
    original_publish = manager.cmd_vel.publish

    def gated_publish(msg: Twist) -> None:
        if not msg.is_zero() and not publish_entered.is_set():
            publish_entered.set()
            assert release_publish.wait(timeout=2.0)
        original_publish(msg)

    monkeypatch.setattr(manager.cmd_vel, "publish", gated_publish)

    def publish_velocity() -> None:
        try:
            if source == "teleop":
                manager._on_teleop(_twist(lx=0.3))
            else:
                manager._on_nav(_twist(lx=0.9))
        except BaseException as exc:
            worker_errors.append(exc)

    velocity_thread = threading.Thread(target=publish_velocity)

    def stop_manager() -> None:
        try:
            manager.stop()
        except BaseException as exc:
            worker_errors.append(exc)
        finally:
            stop_returned.set()

    stop_thread = threading.Thread(target=stop_manager)
    try:
        velocity_thread.start()
        assert publish_entered.wait(timeout=2.0)
        stop_thread.start()
        assert manager._stopping
        assert not stop_returned.wait(timeout=0.05)
        assert captured.cmd_vel == []
        release_publish.set()
        velocity_thread.join(timeout=2.0)
        stop_thread.join(timeout=2.0)
    finally:
        release_publish.set()
        velocity_thread.join(timeout=2.0)
        stop_thread.join(timeout=2.0)

    expected_nonzero = _twist(lx=0.3 if source == "teleop" else 0.9)
    assert worker_errors == []
    assert not velocity_thread.is_alive()
    assert not stop_thread.is_alive()
    assert stop_returned.is_set()
    assert captured.cmd_vel == [expected_nonzero, _twist()]
    assert captured.cmd_vel[-1].is_zero()
    assert manager._stop_complete.is_set()


def test_lifecycle_stop_waits_for_inflight_goal_before_returning(
    manager_and_captured: tuple[MovementManager, Captured],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, captured = manager_and_captured
    goal_publish_entered = threading.Event()
    release_goal_publish = threading.Event()
    stop_returned = threading.Event()
    worker_errors: list[BaseException] = []
    output_order: list[str] = []
    original_publish = manager.goal.publish

    def gated_publish(msg: PointStamped) -> None:
        if not math.isnan(msg.x) and not goal_publish_entered.is_set():
            goal_publish_entered.set()
            assert release_goal_publish.wait(timeout=2.0)
        original_publish(msg)
        output_order.append("goal")

    monkeypatch.setattr(manager.goal, "publish", gated_publish)

    def publish_goal() -> None:
        try:
            manager._on_click(_click(x=4.0))
        except BaseException as exc:
            worker_errors.append(exc)

    goal_thread = threading.Thread(target=publish_goal)

    def stop_manager() -> None:
        try:
            manager.stop()
        except BaseException as exc:
            worker_errors.append(exc)
        finally:
            output_order.append("stop returned")
            stop_returned.set()

    zero_unsub = manager.cmd_vel.subscribe(
        lambda msg: output_order.append("zero") if msg.is_zero() else None
    )
    stop_thread = threading.Thread(target=stop_manager)
    try:
        goal_thread.start()
        assert goal_publish_entered.wait(timeout=2.0)
        stop_thread.start()
        assert manager._stopping
        assert not stop_returned.wait(timeout=0.05)
        release_goal_publish.set()
        goal_thread.join(timeout=2.0)
        stop_thread.join(timeout=2.0)
    finally:
        release_goal_publish.set()
        goal_thread.join(timeout=2.0)
        stop_thread.join(timeout=2.0)
        zero_unsub()

    assert worker_errors == []
    assert not goal_thread.is_alive()
    assert not stop_thread.is_alive()
    assert stop_returned.is_set()
    assert [goal.x for goal in captured.goal] == [4.0]
    assert captured.cmd_vel == [_twist()]
    assert output_order == ["goal", "zero", "stop returned"]
    assert manager._stop_complete.is_set()

    manager._on_click(_click(x=5.0))
    assert [goal.x for goal in captured.goal] == [4.0]


def test_deferred_navigation_uses_validated_message_snapshot(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = manager_and_captured
    manager.config.tele_cooldown_sec = 0.0
    publish_entered = threading.Event()
    release_publish = threading.Event()

    def block_first_velocity(msg: Twist) -> None:
        if msg.linear.x == 0.3:
            publish_entered.set()
            assert release_publish.wait(timeout=2.0)

    unsubscribe = manager.cmd_vel.subscribe(block_first_velocity)
    teleop_thread = threading.Thread(target=manager._on_teleop, args=(_twist(lx=0.3),))
    nav = _twist(lx=0.9)
    try:
        teleop_thread.start()
        assert publish_entered.wait(timeout=2.0)
        manager._on_nav(nav)
        nav.linear.x = float("nan")
        release_publish.set()
        teleop_thread.join(timeout=2.0)
    finally:
        release_publish.set()
        teleop_thread.join(timeout=2.0)
        unsubscribe()

    assert not teleop_thread.is_alive()
    assert captured.cmd_vel == [_twist(lx=0.3), _twist(lx=0.9)]


def test_deferred_click_uses_validated_message_snapshot(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = manager_and_captured
    publish_entered = threading.Event()
    release_publish = threading.Event()

    def block_velocity(msg: Twist) -> None:
        if not msg.is_zero():
            publish_entered.set()
            assert release_publish.wait(timeout=2.0)

    unsubscribe = manager.cmd_vel.subscribe(block_velocity)
    teleop_thread = threading.Thread(target=manager._on_teleop, args=(_twist(lx=0.3),))
    click = _click(x=4.0)
    try:
        teleop_thread.start()
        assert publish_entered.wait(timeout=2.0)
        manager._on_click(click)
        click.x = float("nan")
        click.frame_id = "wrong"
        release_publish.set()
        teleop_thread.join(timeout=2.0)
    finally:
        release_publish.set()
        teleop_thread.join(timeout=2.0)
        unsubscribe()

    assert not teleop_thread.is_alive()
    assert [(goal.x, goal.frame_id) for goal in captured.goal if math.isfinite(goal.x)] == [
        (4.0, "map")
    ]
    assert [
        (point.x, point.frame_id) for point in captured.way_point if math.isfinite(point.x)
    ] == [(4.0, "map")]


def test_navigation_snapshots_before_validation(
    manager_and_captured: tuple[MovementManager, Captured],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, captured = manager_and_captured
    validation_entered = threading.Event()
    release_validation = threading.Event()
    original_validation = movement_manager_module._twist_is_finite

    def gated_validation(msg: Twist) -> bool:
        validation_entered.set()
        assert release_validation.wait(timeout=2.0)
        return original_validation(msg)

    monkeypatch.setattr(movement_manager_module, "_twist_is_finite", gated_validation)
    nav = _twist(lx=0.9)
    nav_thread = threading.Thread(target=manager._on_nav, args=(nav,))
    try:
        nav_thread.start()
        assert validation_entered.wait(timeout=2.0)
        nav.linear.x = float("nan")
        release_validation.set()
        nav_thread.join(timeout=2.0)
    finally:
        release_validation.set()
        nav_thread.join(timeout=2.0)

    assert not nav_thread.is_alive()
    assert captured.cmd_vel == [_twist(lx=0.9)]


def test_click_snapshots_before_validation(
    manager_and_captured: tuple[MovementManager, Captured],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, captured = manager_and_captured
    validation_entered = threading.Event()
    release_validation = threading.Event()
    original_isfinite = math.isfinite
    first_validation = True

    def gated_isfinite(value: float) -> bool:
        nonlocal first_validation
        if first_validation:
            first_validation = False
            validation_entered.set()
            assert release_validation.wait(timeout=2.0)
        return original_isfinite(value)

    monkeypatch.setattr(math, "isfinite", gated_isfinite)
    click = _click(x=4.0)
    click_thread = threading.Thread(target=manager._on_click, args=(click,))
    try:
        click_thread.start()
        assert validation_entered.wait(timeout=2.0)
        click.x = float("nan")
        click.frame_id = "wrong"
        release_validation.set()
        click_thread.join(timeout=2.0)
    finally:
        release_validation.set()
        click_thread.join(timeout=2.0)

    assert not click_thread.is_alive()
    assert [(goal.x, goal.frame_id) for goal in captured.goal] == [(4.0, "map")]


def test_lifecycle_stop_finalizes_when_inflight_velocity_publish_fails(
    manager_and_captured: tuple[MovementManager, Captured],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, captured = manager_and_captured
    publish_entered = threading.Event()
    release_publish = threading.Event()
    stop_returned = threading.Event()
    velocity_errors: list[BaseException] = []
    stop_errors: list[BaseException] = []
    original_publish = manager.cmd_vel.publish

    def failing_publish(msg: Twist) -> None:
        if not msg.is_zero():
            publish_entered.set()
            assert release_publish.wait(timeout=2.0)
            raise RuntimeError("velocity publish failed")
        original_publish(msg)

    monkeypatch.setattr(manager.cmd_vel, "publish", failing_publish)

    def publish_velocity() -> None:
        try:
            manager._on_teleop(_twist(lx=0.3))
        except BaseException as exc:
            velocity_errors.append(exc)

    velocity_thread = threading.Thread(target=publish_velocity)

    def stop_manager() -> None:
        try:
            manager.stop()
        except BaseException as exc:
            stop_errors.append(exc)
        finally:
            stop_returned.set()

    stop_thread = threading.Thread(target=stop_manager)
    try:
        velocity_thread.start()
        assert publish_entered.wait(timeout=2.0)
        stop_thread.start()
        assert not stop_returned.wait(timeout=0.05)
        release_publish.set()
        velocity_thread.join(timeout=2.0)
        stop_thread.join(timeout=2.0)
    finally:
        release_publish.set()
        velocity_thread.join(timeout=2.0)
        stop_thread.join(timeout=2.0)

    assert len(velocity_errors) == 1
    assert isinstance(velocity_errors[0], RuntimeError)
    assert stop_errors == []
    assert stop_returned.is_set()
    assert captured.cmd_vel == [_twist()]
    assert manager._stop_complete.is_set()


def test_reentrant_lifecycle_stop_is_finalized_by_dispatcher(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = manager_and_captured
    late_cmd_vel: list[Twist] = []
    nested_stop_returned = threading.Event()
    stop_sent = False

    def stop_during_nonzero(msg: Twist) -> None:
        nonlocal stop_sent
        if not stop_sent and not msg.is_zero():
            stop_sent = True
            manager.stop()
            nested_stop_returned.set()

    trigger_unsub = manager.cmd_vel.subscribe(stop_during_nonzero)
    late_unsub = manager.cmd_vel.subscribe(late_cmd_vel.append)
    try:
        manager._on_teleop(_twist(lx=0.3))
    finally:
        late_unsub()
        trigger_unsub()

    assert nested_stop_returned.is_set()
    assert manager._stop_complete.is_set()
    assert captured.cmd_vel == [_twist(lx=0.3), _twist()]
    assert late_cmd_vel == [_twist(lx=0.3), _twist()]
    assert captured.cmd_vel[-1].is_zero()
    assert late_cmd_vel[-1].is_zero()


def test_lifecycle_stop_completes_cleanup_when_zero_subscriber_fails(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = manager_and_captured

    def fail_zero(_msg: Twist) -> None:
        raise RuntimeError("velocity subscriber failed")

    unsubscribe = manager.cmd_vel.subscribe(fail_zero)
    try:
        with pytest.raises(RuntimeError, match="velocity subscriber failed"):
            manager.stop()
    finally:
        unsubscribe()

    assert captured.cmd_vel == [_twist()]
    assert manager._stop_complete.is_set()
    manager.stop()


def test_reentrant_stop_can_latch_while_cmd_vel_subscriber_is_blocked(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = manager_and_captured
    manager.config.latch_teleop_stop = True
    nonzero_publish_entered = threading.Event()
    release_nonzero_publish = threading.Event()
    stop_returned = threading.Event()

    def block_nonzero(twist: Twist) -> None:
        if not twist.is_zero():
            nonzero_publish_entered.set()
            assert release_nonzero_publish.wait(timeout=2.0)

    unsubscribe = manager.cmd_vel.subscribe(block_nonzero)
    teleop_thread = threading.Thread(target=manager._on_teleop, args=(_twist(lx=0.3),))

    def publish_stop() -> None:
        manager._on_teleop_stop(Bool(data=True))
        stop_returned.set()

    stop_thread = threading.Thread(target=publish_stop)
    try:
        teleop_thread.start()
        assert nonzero_publish_entered.wait(timeout=2.0)
        stop_thread.start()
        assert stop_returned.wait(timeout=2.0)
        assert manager._operator_stop_latched
        release_nonzero_publish.set()
        teleop_thread.join(timeout=2.0)
        stop_thread.join(timeout=2.0)
    finally:
        release_nonzero_publish.set()
        teleop_thread.join(timeout=2.0)
        stop_thread.join(timeout=2.0)
        unsubscribe()

    assert not teleop_thread.is_alive()
    assert not stop_thread.is_alive()
    assert captured.cmd_vel == [_twist(lx=0.3), _twist()]
    assert captured.cmd_vel[-1].is_zero()


def test_concurrent_stop_waits_for_first_stop_to_complete(
    manager_and_captured: tuple[MovementManager, Captured],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _captured = manager_and_captured
    close_entered = threading.Event()
    release_close = threading.Event()
    second_returned = threading.Event()
    original_close = manager._close_module

    def blocking_close() -> None:
        close_entered.set()
        assert release_close.wait(timeout=2.0)
        original_close()

    monkeypatch.setattr(manager, "_close_module", blocking_close)
    first = threading.Thread(target=manager.stop)

    def stop_again() -> None:
        manager.stop()
        second_returned.set()

    second = threading.Thread(target=stop_again)
    try:
        first.start()
        assert close_entered.wait(timeout=2.0)
        second.start()
        assert not second_returned.wait(timeout=0.05)
        release_close.set()
        first.join(timeout=2.0)
        second.join(timeout=2.0)
    finally:
        release_close.set()
        first.join(timeout=2.0)
        second.join(timeout=2.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert second_returned.is_set()


def test_teleop_cancellation_uses_configured_planning_frame() -> None:
    manager = MovementManager(planning_frame_id="/world")
    captured, unsubs = _attach(manager)
    try:
        manager._on_teleop(_twist(lx=0.3))
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
    manager.config.max_teleop_linear_speed = 1.0
    manager.config.max_teleop_angular_speed = 2.0

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
    manager.config.max_teleop_linear_speed = 1.0
    manager.config.max_teleop_angular_speed = 2.0
    manager.config.tele_cmd_vel_scaling = Twist(Vector3(2, 1, 1), Vector3(1, 1, 1))

    manager._on_teleop(_twist(lx=0.6))

    assert captured.cmd_vel == [_twist()]
    assert not manager._teleop_active
    assert manager._last_teleop_time == 0.0


def test_default_teleop_limits_preserve_finite_existing_commands(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    manager, captured = manager_and_captured
    command = _twist(lx=1.5)

    manager._on_teleop(command)

    assert captured.cmd_vel == [command]


def test_valid_click_publishes_goal(
    manager_and_captured: tuple[MovementManager, Captured],
) -> None:
    """A valid click should publish to both goal and way_point."""
    manager, captured = manager_and_captured
    click = _click(x=5.0, y=3.0, z=0.1)
    manager._on_click(click)
    assert [(goal.x, goal.y, goal.z, goal.ts, goal.frame_id) for goal in captured.goal] == [
        (click.x, click.y, click.z, click.ts, click.frame_id)
    ]
    assert [
        (point.x, point.y, point.z, point.ts, point.frame_id) for point in captured.way_point
    ] == [(click.x, click.y, click.z, click.ts, click.frame_id)]


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
