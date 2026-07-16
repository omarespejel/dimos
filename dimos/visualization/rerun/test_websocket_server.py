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
from concurrent.futures import Future
import json
import socket
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
def publisher(server: RerunWebSocketServer) -> Generator[MockViewerPublisher, None, None]:
    with MockViewerPublisher(f"ws://127.0.0.1:{server.port}/ws") as publisher:
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


def test_click_uses_configured_normalized_frame() -> None:
    """Viewer picks use the configured planning frame, not the picked entity."""
    module = RerunWebSocketServer(click_frame_id="/world/")
    received: list[PointStamped] = []
    unsubscribe = module.clicked_point.subscribe(received.append)
    try:
        module._dispatch(
            json.dumps(
                {
                    "type": "click",
                    "x": 1.0,
                    "y": 2.0,
                    "z": 3.0,
                    "entity_path": "/robot/base",
                    "timestamp_ms": 1234,
                }
            )
        )
    finally:
        unsubscribe()
        module.stop()

    assert len(received) == 1
    assert received[0].frame_id == "world"
    # The viewer timestamp is retained as source metadata. STOP ordering does
    # not depend on timestamps carried by independently delivered streams.
    assert received[0].ts == pytest.approx(1.234)


def test_click_frame_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="click_frame_id must not be empty"):
        RerunWebSocketServer(click_frame_id="///")


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
        with MockViewerPublisher(f"ws://127.0.0.1:{server.port}/ws") as client:
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
        async with ws_client.connect(f"ws://127.0.0.1:{server.port}/ws") as ws:
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


def test_start_fails_immediately_when_serve_exits_before_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = RerunWebSocketServer()

    async def exit_before_ready() -> None:
        return None

    monkeypatch.setattr(module, "_serve", exit_before_ready)
    try:
        with pytest.raises(RuntimeError, match="exited before becoming ready"):
            module.start()
        assert module._serve_teardown_complete.is_set()
        assert module._module_finalize_complete.is_set()
        assert module._loop is None
        assert module._loop_thread is None
    finally:
        module.stop()


def test_start_timeout_defers_cleanup_until_context_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = RerunWebSocketServer()
    loop = module._loop
    assert loop is not None
    context_exit_entered = threading.Event()
    exit_gate: asyncio.Event | None = None
    original_ready_wait = module._server_ready.wait

    class SlowServerContext:
        async def __aenter__(self) -> None:
            nonlocal exit_gate
            exit_gate = asyncio.Event()

        async def __aexit__(self, *_args: Any) -> None:
            assert exit_gate is not None
            context_exit_entered.set()
            await exit_gate.wait()

    def slow_serve(*_args: Any, **_kwargs: Any) -> SlowServerContext:
        return SlowServerContext()

    def never_ready(timeout: float | None = None) -> bool:
        assert original_ready_wait(timeout=2.0)
        return False

    monkeypatch.setattr(ws_server, "serve", slow_serve)
    monkeypatch.setattr(websocket_server_module, "DEFAULT_THREAD_JOIN_TIMEOUT", 0.0)
    monkeypatch.setattr(module._server_ready, "wait", never_ready)
    try:
        with pytest.raises(TimeoutError, match="did not become ready"):
            module.start()

        assert context_exit_entered.wait(timeout=2.0)
        assert not module._serve_teardown_complete.is_set()
        assert not module._module_finalize_complete.is_set()
        assert exit_gate is not None
        loop.call_soon_threadsafe(exit_gate.set)

        assert module._module_finalize_complete.wait(timeout=2.0)
        assert module._serve_teardown_complete.is_set()
        assert module._loop is None
        assert module._loop_thread is None
    finally:
        if exit_gate is not None and not module._serve_teardown_complete.is_set():
            loop.call_soon_threadsafe(exit_gate.set)
            module._module_finalize_complete.wait(timeout=2.0)
        module.stop()


def test_same_loop_start_fails_before_scheduling() -> None:
    async def run() -> None:
        running_loop = asyncio.get_running_loop()
        module = RerunWebSocketServer()
        assert module._loop is running_loop
        assert module._loop_thread is None

        with pytest.raises(RuntimeError, match="cannot synchronously start"):
            module.start()

        assert module._serve_future is None
        assert module._serve_teardown_complete.is_set()
        assert await asyncio.to_thread(module._module_finalize_complete.wait, 2.0)
        assert running_loop.is_running()
        assert not running_loop.is_closed()
        module.stop()

    asyncio.run(run())


def test_off_loop_stop_waits_for_start_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ServerContext:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(ws_server, "serve", lambda *_args, **_kwargs: ServerContext())
    module = RerunWebSocketServer()
    original_submit = asyncio.run_coroutine_threadsafe
    submission_entered = threading.Event()
    release_submission = threading.Event()
    start_done = threading.Event()
    stop_started = threading.Event()
    stop_done = threading.Event()
    errors: list[BaseException] = []

    def blocked_submit(coroutine: Any, loop: asyncio.AbstractEventLoop) -> Any:
        submission_entered.set()
        if not release_submission.wait(timeout=2.0):
            raise TimeoutError("test did not release server submission")
        return original_submit(coroutine, loop)

    def start_module() -> None:
        try:
            module.start()
        except BaseException as error:
            errors.append(error)
        finally:
            start_done.set()

    def stop_module() -> None:
        stop_started.set()
        try:
            module.stop()
        except BaseException as error:
            errors.append(error)
        finally:
            stop_done.set()

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", blocked_submit)
    starter = threading.Thread(target=start_module)
    stopper = threading.Thread(target=stop_module)
    try:
        starter.start()
        assert submission_entered.wait(timeout=2.0)
        stopper.start()
        assert stop_started.wait(timeout=2.0)
        assert not stop_done.wait(timeout=0.1)

        release_submission.set()
        assert start_done.wait(timeout=2.0)
        assert stop_done.wait(timeout=2.0)
        starter.join(timeout=2.0)
        stopper.join(timeout=2.0)

        assert not starter.is_alive()
        assert not stopper.is_alive()
        assert errors == []
        assert module._serve_teardown_complete.is_set()
        assert module._module_finalize_complete.is_set()
    finally:
        release_submission.set()
        starter.join(timeout=2.0)
        if stopper.ident is not None:
            stopper.join(timeout=2.0)
        module.stop()


def test_same_loop_stop_interrupts_start_before_future_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ServerContext:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(ws_server, "serve", lambda *_args, **_kwargs: ServerContext())
    module = RerunWebSocketServer()
    loop = module._loop
    assert loop is not None
    original_submit = asyncio.run_coroutine_threadsafe
    submission_entered = threading.Event()
    release_submission = threading.Event()
    same_loop_stop_done = threading.Event()
    start_done = threading.Event()
    start_errors: list[BaseException] = []
    same_loop_errors: list[BaseException] = []

    def blocked_submit(coroutine: Any, target_loop: asyncio.AbstractEventLoop) -> Any:
        submission_entered.set()
        if not release_submission.wait(timeout=2.0):
            raise TimeoutError("test did not release server submission")
        return original_submit(coroutine, target_loop)

    def start_module() -> None:
        try:
            module.start()
        except BaseException as error:
            start_errors.append(error)
        finally:
            start_done.set()

    def same_loop_stop() -> None:
        try:
            module.stop()
        except BaseException as error:
            same_loop_errors.append(error)
        finally:
            same_loop_stop_done.set()

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", blocked_submit)
    starter = threading.Thread(target=start_module)
    try:
        starter.start()
        assert submission_entered.wait(timeout=2.0)
        loop.call_soon_threadsafe(same_loop_stop)
        assert same_loop_stop_done.wait(timeout=2.0)
        assert module._stop_requested.is_set()
        assert not module._module_finalize_complete.is_set()

        release_submission.set()
        assert start_done.wait(timeout=2.0)
        starter.join(timeout=2.0)

        assert not starter.is_alive()
        assert same_loop_errors == []
        assert len(start_errors) == 1
        assert isinstance(start_errors[0], RuntimeError)
        assert "stop requested during startup" in str(start_errors[0])
        assert module._serve_teardown_complete.is_set()
        assert module._module_finalize_complete.is_set()
        assert module._loop is None
        assert module._loop_thread is None
    finally:
        release_submission.set()
        starter.join(timeout=2.0)
        module.stop()


def test_same_loop_stop_after_readiness_check_is_not_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ServerContext:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(ws_server, "serve", lambda *_args, **_kwargs: ServerContext())
    module = RerunWebSocketServer()
    loop = module._loop
    assert loop is not None
    final_readiness_check = threading.Event()
    release_readiness_check = threading.Event()
    readiness_wait_returned = threading.Event()
    same_loop_stop_done = threading.Event()
    start_done = threading.Event()
    start_errors: list[BaseException] = []
    same_loop_errors: list[BaseException] = []

    class ObservedServerReady:
        def __init__(self) -> None:
            self._event = threading.Event()

        def clear(self) -> None:
            self._event.clear()

        def is_set(self) -> bool:
            return self._event.is_set()

        def set(self) -> None:
            self._event.set()

        def wait(self, timeout: float | None = None) -> bool:
            ready = self._event.wait(timeout)
            if ready:
                readiness_wait_returned.set()
            return ready

    class GatedStopRequest:
        def __init__(self) -> None:
            self._event = threading.Event()
            self._lock = threading.Lock()
            self._gated_after_readiness = False

        def is_set(self) -> bool:
            with self._lock:
                gate_this_check = (
                    readiness_wait_returned.is_set() and not self._gated_after_readiness
                )
                if gate_this_check:
                    self._gated_after_readiness = True
                value = self._event.is_set()
            if gate_this_check:
                final_readiness_check.set()
                if not release_readiness_check.wait(timeout=2.0):
                    raise TimeoutError("test did not release final readiness check")
            return value

        def set(self) -> None:
            self._event.set()

    monkeypatch.setattr(module, "_server_ready", ObservedServerReady())
    monkeypatch.setattr(module, "_stop_requested", GatedStopRequest())

    def start_module() -> None:
        try:
            module.start()
        except BaseException as error:
            start_errors.append(error)
        finally:
            start_done.set()

    def same_loop_stop() -> None:
        try:
            module.stop()
        except BaseException as error:
            same_loop_errors.append(error)
        finally:
            same_loop_stop_done.set()

    starter = threading.Thread(target=start_module)
    try:
        starter.start()
        assert final_readiness_check.wait(timeout=2.0)
        loop.call_soon_threadsafe(same_loop_stop)
        assert same_loop_stop_done.wait(timeout=2.0)

        release_readiness_check.set()
        assert start_done.wait(timeout=2.0)
        starter.join(timeout=2.0)

        assert not starter.is_alive()
        assert same_loop_errors == []
        assert len(start_errors) == 1
        assert isinstance(start_errors[0], RuntimeError)
        assert "stop requested during startup" in str(start_errors[0])
        assert module._serve_teardown_complete.is_set()
        assert module._module_finalize_complete.is_set()
    finally:
        release_readiness_check.set()
        starter.join(timeout=2.0)
        module.stop()


def test_start_timeout_aborts_unstarted_submission_before_live_loop_resumes(
    monkeypatch: pytest.MonkeyPatch,
    unused_tcp_port: int,
) -> None:
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever)
    loop_thread.start()
    blocker_entered = threading.Event()
    release_blocker = threading.Event()
    submitted = threading.Event()
    context_entered = threading.Event()
    start_done = threading.Event()
    start_errors: list[BaseException] = []

    async def construct_on_borrowed_loop() -> RerunWebSocketServer:
        return RerunWebSocketServer()

    module = asyncio.run_coroutine_threadsafe(construct_on_borrowed_loop(), loop).result(
        timeout=2.0
    )
    original_submit = asyncio.run_coroutine_threadsafe
    original_thread_join_timeout = vars(websocket_server_module)["DEFAULT_THREAD_JOIN_TIMEOUT"]

    class BindingContext:
        def __init__(self) -> None:
            self.listener: socket.socket | None = None

        async def __aenter__(self) -> object:
            listener = socket.socket()
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", unused_tcp_port))
            listener.listen()
            self.listener = listener
            context_entered.set()
            return object()

        async def __aexit__(self, *_args: Any) -> None:
            assert self.listener is not None
            self.listener.close()

    def block_loop() -> None:
        blocker_entered.set()
        assert release_blocker.wait(timeout=2.0)

    def observed_submit(coroutine: Any, target_loop: asyncio.AbstractEventLoop) -> Any:
        future = original_submit(coroutine, target_loop)
        submitted.set()
        return future

    def start_module() -> None:
        try:
            module.start()
        except BaseException as error:
            start_errors.append(error)
        finally:
            start_done.set()

    async def drain_loop() -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(ws_server, "serve", lambda *_args, **_kwargs: BindingContext())
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", observed_submit)
    monkeypatch.setattr(websocket_server_module, "DEFAULT_THREAD_JOIN_TIMEOUT", 0.0)
    loop.call_soon_threadsafe(block_loop)
    assert blocker_entered.wait(timeout=2.0)
    starter = threading.Thread(target=start_module)
    try:
        starter.start()
        assert submitted.wait(timeout=2.0)
        assert start_done.wait(timeout=2.0)
        starter.join(timeout=2.0)

        assert not starter.is_alive()
        assert len(start_errors) == 1
        assert isinstance(start_errors[0], TimeoutError)
        assert "did not become ready" in str(start_errors[0])
        assert not module._serve_started.is_set()
        assert not context_entered.is_set()

        drain_future = original_submit(drain_loop(), loop)
        release_blocker.set()
        drain_future.result(timeout=2.0)

        assert module._serve_teardown_complete.wait(timeout=2.0)
        assert module._module_finalize_complete.wait(timeout=2.0)
        assert not context_entered.is_set()
        assert not module._server_ready.is_set()
        assert module._ws_server is None
        assert loop.is_running()
        module.stop()

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", unused_tcp_port))
    finally:
        release_blocker.set()
        monkeypatch.setattr(
            websocket_server_module,
            "DEFAULT_THREAD_JOIN_TIMEOUT",
            original_thread_join_timeout,
        )
        if not module._module_finalize_complete.is_set():
            module.stop()
        starter.join(timeout=2.0)
        if loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=2.0)
        if not loop.is_closed():
            loop.close()


def test_start_force_finalizes_when_borrowed_loop_closes_before_serve_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever)
    loop_thread.start()
    blocker_entered = threading.Event()
    release_blocker = threading.Event()
    submitted = threading.Event()
    start_errors: list[BaseException] = []

    async def construct_on_borrowed_loop() -> RerunWebSocketServer:
        return RerunWebSocketServer()

    module = asyncio.run_coroutine_threadsafe(construct_on_borrowed_loop(), loop).result(
        timeout=2.0
    )
    original_submit = asyncio.run_coroutine_threadsafe

    def block_loop() -> None:
        blocker_entered.set()
        assert release_blocker.wait(timeout=2.0)

    def observed_submit(coroutine: Any, target_loop: asyncio.AbstractEventLoop) -> Any:
        future = original_submit(coroutine, target_loop)
        submitted.set()
        return future

    def start_module() -> None:
        try:
            module.start()
        except BaseException as error:
            start_errors.append(error)

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", observed_submit)
    monkeypatch.setattr(websocket_server_module, "DEFAULT_THREAD_JOIN_TIMEOUT", 0.5)
    loop.call_soon_threadsafe(block_loop)
    assert blocker_entered.wait(timeout=2.0)
    starter = threading.Thread(target=start_module)
    try:
        starter.start()
        assert submitted.wait(timeout=2.0)
        # The submission callback runs before loop.stop(), creating the Task,
        # but the Task's first coroutine step is queued after stop takes effect.
        loop.call_soon_threadsafe(loop.stop)
        release_blocker.set()
        loop_thread.join(timeout=2.0)
        assert not loop_thread.is_alive()
        loop.close()
        starter.join(timeout=2.0)

        assert not starter.is_alive()
        assert len(start_errors) == 1
        assert isinstance(start_errors[0], RuntimeError)
        assert "resources were force-closed" in str(start_errors[0])
        assert not module._serve_started.is_set()
        assert module._serve_task is None
        assert module._serve_coroutine is None
        assert module._serve_teardown_complete.is_set()
        assert module._serve_finalization_ready.is_set()
        assert module._module_finalize_complete.is_set()
        assert module._loop is None
        assert module._loop_thread is None
    finally:
        release_blocker.set()
        if loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=2.0)
        if not loop.is_closed():
            loop.close()


def test_deferred_same_loop_stop_preserves_teardown_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExplodingExitServerContext:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, *_args: Any) -> None:
            raise RuntimeError("teardown exploded")

    monkeypatch.setattr(
        ws_server,
        "serve",
        lambda *_args, **_kwargs: ExplodingExitServerContext(),
    )

    async def run() -> None:
        running_loop = asyncio.get_running_loop()
        module = RerunWebSocketServer()
        await asyncio.to_thread(module.start)
        future = module._serve_future
        assert future is not None

        module.stop()
        assert await asyncio.to_thread(module._module_finalize_complete.wait, 2.0)
        assert future.done()
        assert isinstance(future.exception(), RuntimeError)
        assert str(future.exception()) == "teardown exploded"
        assert not module._serve_teardown_complete.is_set()

        with pytest.raises(RuntimeError, match="teardown exploded"):
            await asyncio.to_thread(module.stop)
        assert running_loop.is_running()
        assert not running_loop.is_closed()

    asyncio.run(run())


def test_concurrent_off_loop_stops_share_one_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ServerContext:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(ws_server, "serve", lambda *_args, **_kwargs: ServerContext())
    module = RerunWebSocketServer()
    module.start()
    delegate = module._serve_future
    assert delegate is not None
    first_result_entered = threading.Event()
    release_result = threading.Event()
    second_started = threading.Event()
    first_done = threading.Event()
    second_done = threading.Event()
    errors: list[BaseException] = []

    class GatedFuture:
        def __init__(self, serve_future: Future[None]) -> None:
            self._serve_future = serve_future
            self.cancel_calls = 0

        def done(self) -> bool:
            if first_result_entered.is_set() and not release_result.is_set():
                return False
            return self._serve_future.done()

        def cancel(self) -> bool:
            self.cancel_calls += 1
            return self._serve_future.cancel()

        def result(self, timeout: float | None = None) -> None:
            first_result_entered.set()
            module._server_ready.clear()
            if not release_result.wait(timeout=2.0):
                raise TimeoutError("test did not release serve result")
            self._serve_future.result(timeout=timeout)

    gated_future = GatedFuture(delegate)
    monkeypatch.setattr(module, "_serve_future", gated_future)

    def stop_module(done: threading.Event, *, mark_started: bool = False) -> None:
        if mark_started:
            second_started.set()
        try:
            module.stop()
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    first = threading.Thread(target=stop_module, args=(first_done,))
    second = threading.Thread(
        target=stop_module,
        args=(second_done,),
        kwargs={"mark_started": True},
    )
    try:
        first.start()
        assert first_result_entered.wait(timeout=2.0)
        second.start()
        assert second_started.wait(timeout=2.0)
        assert not second_done.wait(timeout=0.1)
        assert gated_future.cancel_calls == 0

        release_result.set()
        assert first_done.wait(timeout=2.0)
        assert second_done.wait(timeout=2.0)
        first.join(timeout=2.0)
        second.join(timeout=2.0)

        assert not first.is_alive()
        assert not second.is_alive()
        assert errors == []
        assert gated_future.cancel_calls == 0
        assert module._serve_teardown_complete.is_set()
        assert module._module_finalize_complete.is_set()
    finally:
        release_result.set()
        first.join(timeout=2.0)
        if second.ident is not None:
            second.join(timeout=2.0)
        module.stop()


def test_same_loop_stop_is_nonblocking_and_preserves_borrowed_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(ws_server, "serve", lambda *_args, **_kwargs: SlowExitServerContext())

    async def run() -> None:
        running_loop = asyncio.get_running_loop()
        module = RerunWebSocketServer()
        assert module._loop is running_loop
        assert module._loop_thread is None

        await asyncio.to_thread(module.start)
        try:
            # This call runs on the borrowed module loop. The blocked async
            # context proves stop returned without waiting for teardown.
            module.stop()
            assert not module._module_finalize_complete.is_set()
            finalizer = module._deferred_finalizer
            assert finalizer is not None
            assert finalizer.daemon
            module.stop()
            assert module._deferred_finalizer is finalizer

            assert await asyncio.to_thread(context_exit_entered.wait, 2.0)

            assert exit_gate is not None
            exit_gate.set()
            assert await asyncio.to_thread(module._module_finalize_complete.wait, 2.0)
            assert running_loop.is_running()
            assert not running_loop.is_closed()
            assert module._loop is running_loop
            assert module._loop_thread is None
            module.stop()
        finally:
            if exit_gate is not None:
                exit_gate.set()
            if not module._module_finalize_complete.is_set():
                await asyncio.to_thread(module.stop)

    asyncio.run(run())


def test_ready_server_stop_timeout_defers_cleanup_until_context_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = RerunWebSocketServer()
    loop = module._loop
    assert loop is not None
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
        loop.call_soon_threadsafe(exit_gate.set)
        assert module._module_finalize_complete.wait(timeout=2.0)
        assert module._serve_teardown_complete.is_set()
        module.stop()
        assert module._loop is None
        assert module._loop_thread is None
    finally:
        if exit_gate is not None and not module._serve_teardown_complete.is_set():
            loop.call_soon_threadsafe(exit_gate.set)
            module._module_finalize_complete.wait(timeout=2.0)
        module.stop()


def test_stop_recovers_when_signal_scheduling_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ServerContext:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(ws_server, "serve", lambda *_args, **_kwargs: ServerContext())
    module = RerunWebSocketServer()
    module.start()
    loop = module._loop
    stop_event = module._stop_event
    assert loop is not None
    assert stop_event is not None
    original_call_soon_threadsafe = loop.call_soon_threadsafe
    signal_failed = False

    def close_during_signal(callback: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal signal_failed
        if not signal_failed and getattr(callback, "__self__", None) is stop_event:
            signal_failed = True
            raise RuntimeError("Event loop is closed")
        return original_call_soon_threadsafe(callback, *args, **kwargs)

    monkeypatch.setattr(loop, "call_soon_threadsafe", close_during_signal)
    try:
        module.stop()

        assert signal_failed
        assert module._serve_teardown_complete.is_set()
        assert module._module_finalize_complete.is_set()
        assert module._loop is None
        assert module._loop_thread is None
    finally:
        module.stop()


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_closed_borrowed_loop_force_stops_controller_and_releases_listener(
    unused_tcp_port: int,
) -> None:
    original_port = global_config.rerun_websocket_server_port
    global_config.update(rerun_websocket_server_port=unused_tcp_port)
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever)
    loop_thread.start()
    module: RerunWebSocketServer | None = None
    finalizer: threading.Thread | None = None
    client: MockViewerPublisher | None = None
    received_twists: list[Twist] = []
    received_stops: list[Bool] = []
    moving = threading.Event()
    unsubscribe_twist: Any = None
    unsubscribe_stop: Any = None

    async def construct_on_borrowed_loop() -> RerunWebSocketServer:
        return RerunWebSocketServer()

    try:
        module = asyncio.run_coroutine_threadsafe(construct_on_borrowed_loop(), loop).result(
            timeout=2.0
        )
        assert module._loop is loop
        assert module._loop_thread is None
        module.start()
        assert module._ws_server is not None

        def capture_twist(twist: Twist) -> None:
            received_twists.append(twist)
            if not twist.is_zero():
                moving.set()

        def capture_stop_and_fail(stop: Bool) -> None:
            received_stops.append(stop)
            raise RuntimeError("semantic stop subscriber failed")

        unsubscribe_twist = module.tele_cmd_vel.subscribe(capture_twist)
        unsubscribe_stop = module.teleop_stop.subscribe(capture_stop_and_fail)
        client = MockViewerPublisher(f"ws://127.0.0.1:{unused_tcp_port}/ws")
        client.__enter__()
        client.send_twist(0.7, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert moving.wait(timeout=2.0)
        assert module._controlling_client is not None
        assert module._ws_server is not None
        server_connection = next(iter(module._ws_server.handlers))
        server_transport = server_connection.transport

        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=2.0)
        assert not loop_thread.is_alive()
        loop.close()

        module._schedule_deferred_finalizer()
        finalizer = module._deferred_finalizer
        assert finalizer is not None
        assert finalizer.is_alive()

        with pytest.raises(RuntimeError, match="resources were force-closed"):
            module.stop()

        assert not module._serve_teardown_complete.is_set()
        assert module._serve_finalization_ready.is_set()
        assert module._module_finalize_complete.is_set()
        assert module._loop is None
        assert module._loop_thread is None
        assert module._controlling_client is None
        assert server_transport.is_closing()
        assert getattr(server_transport, "_sock", None) is None
        assert len(received_stops) == 1
        assert received_stops[0].data
        assert len(received_twists) == 2
        assert received_twists[0].linear.x == pytest.approx(0.7)
        assert received_twists[1].is_zero()
        finalizer.join(timeout=2.0)
        assert not finalizer.is_alive()

        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", unused_tcp_port))
    finally:
        if loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=2.0)
        if not loop.is_closed():
            loop.close()
        if finalizer is not None:
            finalizer.join(timeout=2.0)
        try:
            if client is not None:
                client.__exit__(None, None, None)
        finally:
            if unsubscribe_twist is not None:
                unsubscribe_twist()
            if unsubscribe_stop is not None:
                unsubscribe_stop()
            global_config.update(rerun_websocket_server_port=original_port)
