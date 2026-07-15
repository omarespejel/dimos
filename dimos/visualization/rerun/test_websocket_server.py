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

"""Tests for RerunWebSocketServer."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
import json
import threading
import time
from typing import Any

from dimos_lcm.std_msgs import Bool  # type: ignore[import-untyped]
import pytest
import websockets.asyncio.client as ws_client
import websockets.asyncio.server as ws_server

from dimos.core.global_config import global_config
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.Twist import Twist
import dimos.visualization.rerun.websocket_server as websocket_server_module
from dimos.visualization.rerun.websocket_server import RerunWebSocketServer

_TEST_PORT = 13031


class MockViewerPublisher:
    """Simulates dimos-viewer sending JSON events over WebSocket."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._ws: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def __enter__(self) -> MockViewerPublisher:
        self._loop = asyncio.new_event_loop()
        self._ws = self._loop.run_until_complete(self._connect())
        return self

    def __exit__(self, *_: Any) -> None:
        if self._ws is not None and self._loop is not None:
            self._loop.run_until_complete(self._ws.close())
        if self._loop is not None:
            self._loop.close()

    async def _connect(self) -> Any:
        return await ws_client.connect(self._url)

    def send_click(
        self, x: float, y: float, z: float, entity_path: str = "", timestamp_ms: int = 0
    ) -> None:
        self._send(
            {
                "type": "click",
                "x": x,
                "y": y,
                "z": z,
                "entity_path": entity_path,
                "timestamp_ms": timestamp_ms,
            }
        )

    def send_twist(
        self,
        linear_x: float,
        linear_y: float,
        linear_z: float,
        angular_x: float,
        angular_y: float,
        angular_z: float,
    ) -> None:
        self._send(
            {
                "type": "twist",
                "linear_x": linear_x,
                "linear_y": linear_y,
                "linear_z": linear_z,
                "angular_x": angular_x,
                "angular_y": angular_y,
                "angular_z": angular_z,
            }
        )

    def send_stop(self) -> None:
        self._send({"type": "stop"})

    def flush(self, delay: float = 0.1) -> None:
        time.sleep(delay)

    def _send(self, msg: dict[str, Any]) -> None:
        assert self._loop is not None and self._ws is not None
        self._loop.run_until_complete(self._ws.send(json.dumps(msg)))


@pytest.fixture()
def server(wait_for_server: Any) -> Generator[RerunWebSocketServer, None, None]:
    original_port = global_config.rerun_websocket_server_port
    global_config.update(rerun_websocket_server_port=_TEST_PORT)
    module: RerunWebSocketServer | None = None
    try:
        module = RerunWebSocketServer()
        module.start()
        wait_for_server(_TEST_PORT)
        yield module  # type: ignore[misc]
    finally:
        try:
            if module is not None:
                module.stop()
        finally:
            global_config.update(rerun_websocket_server_port=original_port)


@pytest.fixture()
def publisher(server: RerunWebSocketServer) -> Generator[MockViewerPublisher, None, None]:
    with MockViewerPublisher(f"ws://127.0.0.1:{_TEST_PORT}/ws") as publisher:
        yield publisher  # type: ignore[misc]


def test_click_publishes_point_stamped(
    server: RerunWebSocketServer, publisher: MockViewerPublisher
) -> None:
    """Click coordinates use the planning frame, never the picked entity path."""
    received: list[PointStamped] = []
    done = threading.Event()

    def capture_point(point: PointStamped) -> None:
        received.append(point)
        done.set()

    unsub = server.clicked_point.subscribe(capture_point)

    publisher.send_click(1.5, 2.5, 0.0, "/robot/base", timestamp_ms=5000)
    publisher.flush()
    done.wait(timeout=2.0)
    unsub()

    assert len(received) == 1
    point = received[0]
    assert point.x == pytest.approx(1.5)
    assert point.y == pytest.approx(2.5)
    assert point.z == pytest.approx(0.0)
    assert point.frame_id == "map"
    assert point.ts == pytest.approx(5.0)


def test_twist_publishes_on_tele_cmd_vel(
    server: RerunWebSocketServer, publisher: MockViewerPublisher
) -> None:
    """Twist event arrives as Twist on tele_cmd_vel."""
    received: list[Twist] = []
    done = threading.Event()

    def capture_twist(twist: Twist) -> None:
        received.append(twist)
        done.set()

    unsub = server.tele_cmd_vel.subscribe(capture_twist)

    publisher.send_twist(0.5, 0.0, 0.0, 0.0, 0.0, 0.8)
    publisher.flush()
    done.wait(timeout=2.0)
    unsub()

    assert len(received) == 1
    assert received[0].linear.x == pytest.approx(0.5)
    assert received[0].angular.z == pytest.approx(0.8)


def test_stop_publishes_explicit_signal_and_zero_twist(
    server: RerunWebSocketServer, publisher: MockViewerPublisher
) -> None:
    """Stop event preserves its semantic and publishes zero velocity."""
    received_twists: list[Any] = []
    received_stops: list[Any] = []
    done = threading.Event()

    def mark_done() -> None:
        if received_twists and received_stops:
            done.set()

    def capture_twist(twist: Twist) -> None:
        received_twists.append(twist)
        mark_done()

    def capture_stop(stop: Bool) -> None:
        received_stops.append(stop)
        mark_done()

    unsub_twist = server.tele_cmd_vel.subscribe(capture_twist)
    unsub_stop = server.teleop_stop.subscribe(capture_stop)

    publisher.send_stop()
    done.wait(timeout=2.0)
    unsub_twist()
    unsub_stop()

    assert len(received_twists) == 1
    assert received_twists[0].is_zero()
    assert len(received_stops) == 1
    assert received_stops[0].data


def test_stop_publishes_zero_when_semantic_subscriber_fails(
    server: RerunWebSocketServer,
) -> None:
    received_twists: list[Twist] = []

    def fail_stop_subscriber(_stop: Bool) -> None:
        raise RuntimeError("semantic stop subscriber failed")

    unsub_twist = server.tele_cmd_vel.subscribe(received_twists.append)
    unsub_stop = server.teleop_stop.subscribe(fail_stop_subscriber)
    try:
        with pytest.raises(RuntimeError, match="semantic stop subscriber failed"):
            server._dispatch(json.dumps({"type": "stop"}))
    finally:
        unsub_twist()
        unsub_stop()

    assert len(received_twists) == 1
    assert received_twists[0].is_zero()


def test_controlling_client_disconnect_publishes_stop_and_zero(
    server: RerunWebSocketServer,
) -> None:
    received_twists: list[Twist] = []
    received_stops: list[Bool] = []
    moving = threading.Event()
    stopped = threading.Event()

    def capture_twist(twist: Twist) -> None:
        received_twists.append(twist)
        if twist.is_zero():
            stopped.set()
        else:
            moving.set()

    unsub_twist = server.tele_cmd_vel.subscribe(capture_twist)
    unsub_stop = server.teleop_stop.subscribe(received_stops.append)
    try:
        with MockViewerPublisher(f"ws://127.0.0.1:{_TEST_PORT}/ws") as client:
            client.send_twist(0.5, 0.0, 0.0, 0.0, 0.0, 0.0)
            assert moving.wait(timeout=2.0)
        assert stopped.wait(timeout=2.0)
    finally:
        unsub_twist()
        unsub_stop()

    assert len(received_stops) == 1
    assert received_stops[0].data
    assert len(received_twists) == 2
    assert received_twists[0].linear.x == pytest.approx(0.5)
    assert received_twists[1].is_zero()


def test_invalid_json_does_not_crash(server: RerunWebSocketServer) -> None:
    """Malformed JSON is silently dropped; server stays alive for the next message."""

    async def _send_bad() -> None:
        async with ws_client.connect(f"ws://127.0.0.1:{_TEST_PORT}/ws") as ws:
            await ws.send("this is not json {{")
            await asyncio.sleep(0.1)
            await ws.send(json.dumps({"type": "heartbeat", "timestamp_ms": 0}))
            await asyncio.sleep(0.1)

    asyncio.run(_send_bad())


def test_mixed_message_sequence(
    server: RerunWebSocketServer, publisher: MockViewerPublisher
) -> None:
    """Realistic session: heartbeat, click, twist, stop — only the click produces a point."""
    received: list[PointStamped] = []
    done = threading.Event()

    def capture_point(point: PointStamped) -> None:
        received.append(point)
        done.set()

    unsub = server.clicked_point.subscribe(capture_point)

    publisher.send_click(7.0, 8.0, 9.0, "/map", timestamp_ms=1100)
    publisher.send_twist(0.3, 0.0, 0.0, 0.0, 0.0, 0.2)
    publisher.send_stop()
    publisher.flush()
    done.wait(timeout=2.0)
    unsub()

    assert len(received) == 1
    assert received[0].x == pytest.approx(7.0)
    assert received[0].y == pytest.approx(8.0)
    assert received[0].z == pytest.approx(9.0)


def test_start_propagates_serve_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    module = RerunWebSocketServer()

    def fail_serve(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("bind failed")

    monkeypatch.setattr(ws_server, "serve", fail_serve)
    try:
        with pytest.raises(OSError, match="bind failed"):
            module.start()
        assert not module._server_ready.is_set()
        assert module._serve_teardown_complete.is_set()
        assert module._loop is None
        assert module._loop_thread is None
    finally:
        module.stop()


def test_start_waits_for_serve_teardown_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    module = RerunWebSocketServer()
    server_context_exited = threading.Event()

    class SlowServerContext:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, *_args: Any) -> None:
            await asyncio.sleep(0.02)
            server_context_exited.set()

    def slow_serve(*_args: Any, **_kwargs: Any) -> SlowServerContext:
        return SlowServerContext()

    def never_ready(timeout: float | None = None) -> bool:
        if timeout:
            time.sleep(timeout)
        return False

    monkeypatch.setattr(ws_server, "serve", slow_serve)
    monkeypatch.setattr(websocket_server_module, "DEFAULT_THREAD_JOIN_TIMEOUT", 0.1)
    monkeypatch.setattr(module._server_ready, "wait", never_ready)
    try:
        with pytest.raises(TimeoutError, match="did not become ready"):
            module.start()
        assert server_context_exited.is_set()
        assert module._serve_teardown_complete.is_set()
        assert not module._server_ready.is_set()
        assert module._loop is None
        assert module._loop_thread is None
    finally:
        module.stop()


def test_stop_reports_incomplete_serve_teardown(monkeypatch: pytest.MonkeyPatch) -> None:
    module = RerunWebSocketServer()
    cancel_serve_and_wait = module._cancel_serve_and_wait
    monkeypatch.setattr(module, "_cancel_serve_and_wait", lambda: False)
    try:
        with pytest.raises(TimeoutError, match="teardown did not complete"):
            module.stop()
        assert module._loop is not None
        assert module._loop_thread is not None
        assert module._loop_thread.is_alive()
    finally:
        monkeypatch.setattr(module, "_cancel_serve_and_wait", cancel_serve_and_wait)
        module.stop()


def test_ready_server_stop_preserves_loop_until_context_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = RerunWebSocketServer()
    context_exit_entered = threading.Event()
    exit_gate: asyncio.Event | None = None

    class SlowExitServerContext:
        async def __aenter__(self) -> None:
            nonlocal exit_gate
            exit_gate = asyncio.Event()

        async def __aexit__(self, *_args: Any) -> None:
            assert exit_gate is not None
            context_exit_entered.set()
            await exit_gate.wait()

    def slow_exit_serve(*_args: Any, **_kwargs: Any) -> SlowExitServerContext:
        return SlowExitServerContext()

    monkeypatch.setattr(ws_server, "serve", slow_exit_serve)
    monkeypatch.setattr(websocket_server_module, "DEFAULT_THREAD_JOIN_TIMEOUT", 0.05)
    try:
        module.start()
        with pytest.raises(TimeoutError, match="teardown did not complete"):
            module.stop()

        assert context_exit_entered.is_set()
        assert module._loop is not None
        assert not module._loop.is_closed()
        assert module._loop_thread is not None
        assert module._loop_thread.is_alive()

        assert exit_gate is not None
        module._loop.call_soon_threadsafe(exit_gate.set)
        assert module._serve_teardown_complete.wait(timeout=2.0)
        module.stop()
        assert module._loop is None
        assert module._loop_thread is None
    finally:
        if exit_gate is not None and module._loop is not None:
            module._loop.call_soon_threadsafe(exit_gate.set)
            module._serve_teardown_complete.wait(timeout=2.0)
        module.stop()
