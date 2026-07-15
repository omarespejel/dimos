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

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from concurrent.futures import Future
import json
import logging
import threading
import time
from typing import Any, Literal, TypedDict, Union

from dimos_lcm.std_msgs import Bool  # type: ignore[import-untyped]
from pydantic import field_validator
import websockets
import websockets.asyncio.server as ws_server

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class ClickMsg(TypedDict):
    type: Literal["click"]
    x: float
    y: float
    z: float
    entity_path: str
    timestamp_ms: int


class TwistMsg(TypedDict):
    type: Literal["twist"]
    linear_x: float
    linear_y: float
    linear_z: float
    angular_x: float
    angular_y: float
    angular_z: float


class StopMsg(TypedDict):
    type: Literal["stop"]


class HeartbeatMsg(TypedDict):
    type: Literal["heartbeat"]
    timestamp_ms: int


ViewerMsg = Union[ClickMsg, TwistMsg, StopMsg, HeartbeatMsg]


class Config(ModuleConfig):
    """Configuration for viewer input received over WebSocket."""

    click_frame_id: str = "map"

    @field_validator("click_frame_id")
    @classmethod
    def normalize_click_frame_id(cls, value: str) -> str:
        normalized = value.strip("/")
        if not normalized:
            raise ValueError("click_frame_id must not be empty")
        return normalized


def _handshake_noise_filter(record: logging.LogRecord) -> bool:
    """Drop noisy "opening handshake failed" records from port scanners etc."""
    msg = record.getMessage()
    return not ("opening handshake failed" in msg or "did not receive a valid HTTP request" in msg)


class RerunWebSocketServer(Module):
    """This handles outputs from dimos-viewer (like keyboard controls)"""

    config: Config

    clicked_point: Out[PointStamped]
    tele_cmd_vel: Out[Twist]
    teleop_stop: Out[Bool]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stop_event: asyncio.Event | None = None
        self._serve_future: Future[None] | None = None
        self._serve_coroutine: Coroutine[Any, Any, None] | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._serve_error: BaseException | None = None
        self._serve_error_reported = False
        self._serve_started = threading.Event()
        self._serve_context_entered = threading.Event()
        self._server_ready = threading.Event()
        self._serve_teardown_complete = threading.Event()
        self._serve_teardown_complete.set()
        self._serve_finalization_ready = threading.Event()
        self._serve_finalization_ready.set()
        self._lifecycle_lock = threading.RLock()
        self._stop_requested = threading.Event()
        self._force_cleanup_started = threading.Event()
        self._ws_server: ws_server.Server | None = None
        self._module_finalize_lock = threading.RLock()
        self._module_finalize_started = False
        self._module_finalize_complete = threading.Event()
        self._module_finalize_error: BaseException | None = None
        self._deferred_finalizer_lock = threading.Lock()
        self._deferred_finalizer: threading.Thread | None = None
        # Set on the first WebSocket client connection. Tests use this to verify
        # an external client (e.g. dimos-viewer --connect) actually connected.
        self.client_connected = threading.Event()
        self._controlling_client: Any | None = None

    @property
    def host(self) -> str:
        return self.config.g.rerun_host or self.config.g.listen_host

    @property
    def port(self) -> int:
        return self.config.g.rerun_websocket_server_port

    @rpc
    def start(self) -> None:
        if self._is_on_module_loop():
            if self._lifecycle_lock.acquire(blocking=False):
                try:
                    if self._serve_future is None and not self._module_finalize_complete.is_set():
                        self._schedule_deferred_finalizer()
                finally:
                    self._lifecycle_lock.release()
            raise RuntimeError("WebSocket server cannot synchronously start on its event loop")

        with self._lifecycle_lock:
            if self._module_finalize_complete.is_set():
                raise RuntimeError("WebSocket server module is already stopped")
            if self._serve_future is not None:
                raise RuntimeError("WebSocket server is already started")
            if self._stop_requested.is_set():
                self._schedule_deferred_finalizer()
                raise RuntimeError("WebSocket server stop was requested before startup")

            try:
                super().start()
                loop = self._loop
                if loop is None or loop.is_closed() or not loop.is_running():
                    raise RuntimeError("WebSocket server event loop is not running")

                self._server_ready.clear()
                self._serve_teardown_complete.clear()
                self._serve_finalization_ready.clear()
                self._serve_started.clear()
                self._serve_context_entered.clear()
                self._force_cleanup_started.clear()
                serve = self._run_server()
                self._serve_coroutine = serve
                try:
                    serve_future = asyncio.run_coroutine_threadsafe(serve, loop)
                except BaseException:
                    self._serve_coroutine = None
                    serve.close()
                    self._serve_teardown_complete.set()
                    self._serve_finalization_ready.set()
                    raise
                self._serve_future = serve_future

                if self._stop_requested.is_set():
                    raise RuntimeError("WebSocket server stop requested during startup")

                deadline = time.monotonic() + DEFAULT_THREAD_JOIN_TIMEOUT
                while True:
                    if self._stop_requested.is_set():
                        raise RuntimeError("WebSocket server stop requested during startup")
                    if self._server_ready.wait(
                        timeout=min(0.05, max(0.0, deadline - time.monotonic()))
                    ):
                        if self._stop_requested.is_set():
                            raise RuntimeError("WebSocket server stop requested during startup")
                        break
                    if serve_future.done():
                        serve_future.result()
                        raise RuntimeError("WebSocket server exited before becoming ready")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("WebSocket server did not become ready")
            except BaseException:
                # Startup failure is terminal for this module instance. A
                # run_coroutine_threadsafe submission may still be queued on a
                # live but stalled borrowed loop; its first step must observe
                # this request before it can bind a listener.
                self._stop_requested.set()
                cleanup_complete = self._cancel_serve_and_wait()
                self._serve_error_reported = True
                if not cleanup_complete:
                    loop = self._loop
                    if loop is not None and loop.is_closed():
                        self._finalize_after_loop_closed()
                    self._schedule_deferred_finalizer()
                    logger.error(
                        "WebSocket server teardown is still running after startup error; "
                        "cleanup was deferred"
                    )
                else:
                    try:
                        self._finalize_module_once()
                    except BaseException:
                        logger.exception("Failed to tear down WebSocket server after startup error")
                raise

        # A same-loop stop can set the request after the readiness check but
        # before start() releases the lifecycle lock. Recheck after release;
        # later same-loop stops can acquire the lock and tear down directly.
        if self._stop_requested.is_set():
            startup_error = RuntimeError("WebSocket server stop requested during startup")
            try:
                self.stop()
            except BaseException as error:
                raise startup_error from error
            raise startup_error

    def _is_on_module_loop(self) -> bool:
        loop = self._loop
        if loop is None:
            return False
        try:
            if asyncio.get_running_loop() is loop:
                return True
        except RuntimeError:
            pass
        return self._loop_thread is not None and threading.current_thread() is self._loop_thread

    def _finalize_module_once(self) -> None:
        """Run the base module teardown once, and always from a non-loop thread."""
        with self._module_finalize_lock:
            if not self._module_finalize_started:
                self._module_finalize_started = True
                try:
                    super().stop()
                except BaseException as error:
                    self._module_finalize_error = error
                finally:
                    if (
                        self._module_finalize_error is None
                        and self._serve_error is not None
                        and not self._serve_error_reported
                    ):
                        self._module_finalize_error = self._serve_error
                    self._module_finalize_complete.set()

            if self._module_finalize_error is not None:
                raise self._module_finalize_error

    def _schedule_deferred_finalizer(self) -> None:
        """Finish module teardown after the WebSocket context has unwound."""
        with self._deferred_finalizer_lock:
            if self._module_finalize_complete.is_set() or self._deferred_finalizer is not None:
                return

            def finalize() -> None:
                self._serve_finalization_ready.wait()
                try:
                    with self._lifecycle_lock:
                        self._finalize_module_once()
                except BaseException:
                    logger.exception("Deferred WebSocket server teardown failed")

            self._deferred_finalizer = threading.Thread(
                target=finalize,
                name="rerun-websocket-finalizer",
                daemon=True,
            )
            self._deferred_finalizer.start()

    def _cancel_serve_and_wait(self) -> bool:
        future = self._serve_future
        if future is None:
            return True
        if not future.done():
            # Let the lifecycle wrapper take its first step before cancelling
            # it. Cancelling run_coroutine_threadsafe's Future before its Task
            # starts would skip the wrapper's finally block and strand both
            # teardown gates.
            if not self._serve_started.is_set() and not self._is_on_module_loop():
                self._serve_started.wait(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
            if self._serve_teardown_complete.is_set():
                return True
            task = self._serve_task
            loop = self._loop
            if task is None or loop is None or loop.is_closed() or not loop.is_running():
                return False
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                if loop.is_closed():
                    self._server_ready.clear()
                    self._serve_finalization_ready.set()
                    return False
                raise
        if self._serve_teardown_complete.is_set():
            return True
        if self._is_on_module_loop():
            return False
        return self._serve_teardown_complete.wait(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)

    @staticmethod
    def _close_task_after_loop_loss(task: Any, description: str) -> None:
        """Dispose a suspended task without scheduling on its closed loop."""
        if task is None or task.done():
            return

        # Task.cancel() and coroutine.close() both cancel the Task's current
        # waiter. asyncio schedules the wakeup callback on the owning loop,
        # which is impossible here. Detach callbacks and finish the waiter
        # first; the coroutine's later cancel() is then an idempotent no-op.
        waiter = getattr(task, "_fut_waiter", None)
        if waiter is not None and not waiter.done():
            callbacks = tuple(getattr(waiter, "_callbacks", ()) or ())
            for callback_entry in callbacks:
                callback = (
                    callback_entry[0] if isinstance(callback_entry, tuple) else callback_entry
                )
                waiter.remove_done_callback(callback)
            try:
                waiter.set_result(None)
            except BaseException as error:
                logger.warning(f"Could not finish {description} waiter: {error}")

        task._log_destroy_pending = False  # type: ignore[attr-defined]
        try:
            coroutine = task.get_coro()
            if coroutine is not None:
                coroutine.close()
        except BaseException as error:
            logger.warning(f"Could not close abandoned {description} coroutine: {error}")

    def _force_close_ws_resources(self) -> None:
        """Close listener and client sockets after their event loop is gone."""
        server = self._ws_server
        if server is None:
            return

        underlying_server = getattr(server, "server", None)
        listener_sockets = tuple(getattr(underlying_server, "_sockets", ()) or ())
        if underlying_server is not None:
            try:
                underlying_server.close()
            except BaseException as error:
                logger.warning(f"Could not close WebSocket listener normally: {error}")
        for listener_socket in listener_sockets:
            try:
                listener_socket.close()
            except OSError as error:
                logger.warning(f"Could not force-close WebSocket listener socket: {error}")

        handlers = getattr(server, "handlers", {})
        for connection, handler_task in tuple(handlers.items()):
            transport = getattr(connection, "transport", None)
            raw_socket = getattr(transport, "_sock", None) if transport is not None else None
            if raw_socket is not None and transport is not None:
                try:
                    raw_socket.close()
                except OSError as error:
                    logger.warning(f"Could not force-close WebSocket client socket: {error}")
                finally:
                    # _SelectorTransport.__del__ treats a retained _sock as an
                    # unclosed transport even when that socket was closed
                    # directly. The loop cannot run _call_connection_lost(),
                    # so mirror its terminal transport state without invoking
                    # protocol callbacks on the closed loop.
                    transport._sock = None  # type: ignore[attr-defined]
                    transport._closing = True  # type: ignore[attr-defined]
                    transport._conn_lost = max(  # type: ignore[attr-defined]
                        1, getattr(transport, "_conn_lost", 0)
                    )
            keepalive_task = getattr(connection, "keepalive_task", None)
            self._close_task_after_loop_loss(keepalive_task, "WebSocket keepalive")
            self._close_task_after_loop_loss(handler_task, "WebSocket client handler")

        self._ws_server = None

    def _publish_stop_after_loop_loss(self) -> None:
        """Fail closed once when a controlling viewer cannot disconnect cleanly."""
        if self._controlling_client is None:
            return
        self._controlling_client = None
        try:
            self._publish_stop()
        except BaseException:
            # _publish_stop publishes zero in a finally block. Preserve socket
            # and module cleanup even if a semantic-stop subscriber fails.
            logger.exception("Forced WebSocket cleanup stop subscriber failed")

    def _finalize_after_loop_closed(self) -> None:
        """Force-close sockets when the event loop cannot unwind the server."""
        lifecycle_error = RuntimeError(
            "WebSocket event loop closed before graceful teardown; resources were force-closed"
        )
        self._force_cleanup_started.set()
        # run_coroutine_threadsafe() chains cancellation back onto the event
        # loop. Once that loop is closed, Future.cancel() only emits an
        # unhandled callback error; close the retained task/coroutine below
        # without scheduling work instead.
        self._publish_stop_after_loop_loss()
        self._force_close_ws_resources()
        serve_coroutine = self._serve_coroutine
        serve_task = self._serve_task
        loop = self._loop
        if serve_task is None and serve_coroutine is not None and loop is not None:
            # run_coroutine_threadsafe() may have created the Task while the
            # loop was stopping, without ever executing _run_server's first
            # line. Recover that Task from the loop so it doesn't survive as a
            # destroyed-pending warning.
            serve_task = next(
                (task for task in asyncio.all_tasks(loop) if task.get_coro() is serve_coroutine),
                None,
            )
        self._close_task_after_loop_loss(serve_task, "WebSocket serve")
        self._serve_coroutine = None
        if serve_coroutine is not None and serve_task is None:
            serve_coroutine.close()
        if not self._serve_context_entered.is_set():
            # No async server context was entered, so there was nothing to
            # unwind even though losing the loop is still a lifecycle error.
            self._serve_teardown_complete.set()
        self._serve_task = None
        self._server_ready.clear()
        # A closed loop cannot run the async context's finally block. Release
        # finalizers through a separate gate without claiming teardown ran.
        self._serve_finalization_ready.set()
        self._serve_future = None
        self._stop_event = None
        self._loop = None
        self._loop_thread = None
        self._serve_error = lifecycle_error
        self._serve_error_reported = False
        try:
            self._finalize_module_once()
        except BaseException as error:
            if error is lifecycle_error:
                raise
            raise lifecycle_error from error
        raise lifecycle_error

    @rpc
    def stop(self) -> None:
        if self._is_on_module_loop():
            # An off-loop start may hold the lifecycle lock while submitting
            # _run_server. Record the request without blocking this event loop;
            # start() observes it before reporting readiness.
            self._stop_requested.set()
            if not self._lifecycle_lock.acquire(blocking=False):
                return
            try:
                if self._module_finalize_complete.is_set():
                    self._finalize_module_once()
                    return
                if self._server_ready.is_set() and self._stop_event is not None:
                    self._stop_event.set()
                else:
                    task = self._serve_task
                    if task is not None and not task.done():
                        task.cancel()
                self._schedule_deferred_finalizer()
            finally:
                self._lifecycle_lock.release()
            return

        with self._lifecycle_lock:
            self._stop_requested.set()
            if self._module_finalize_complete.is_set():
                self._finalize_module_once()
                return

            loop = self._loop
            if loop is not None and loop.is_closed() and not self._serve_teardown_complete.is_set():
                self._finalize_after_loop_closed()

            if not self._server_ready.is_set():
                if not self._cancel_serve_and_wait():
                    self._schedule_deferred_finalizer()
                    raise TimeoutError("WebSocket server teardown did not complete during stop")
                self._finalize_module_once()
                return

            stop_event = self._stop_event
            future = self._serve_future
            if loop is None or stop_event is None or future is None:
                if not self._cancel_serve_and_wait():
                    self._schedule_deferred_finalizer()
                    raise TimeoutError("WebSocket server teardown did not complete during stop")
            else:
                try:
                    loop.call_soon_threadsafe(stop_event.set)
                except RuntimeError as error:
                    if loop.is_closed():
                        self._finalize_after_loop_closed()
                    if not self._cancel_serve_and_wait():
                        self._schedule_deferred_finalizer()
                        raise TimeoutError(
                            "WebSocket server teardown did not complete during stop"
                        ) from error
                    self._finalize_module_once()
                    return
                serve_error: BaseException | None = None
                try:
                    future.result(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
                except TimeoutError as error:
                    self._schedule_deferred_finalizer()
                    raise TimeoutError(
                        "WebSocket server teardown did not complete during stop"
                    ) from error
                except BaseException as error:
                    serve_error = error

                if not self._serve_teardown_complete.is_set():
                    self._schedule_deferred_finalizer()
                    raise TimeoutError("WebSocket server teardown did not complete during stop")

                self._finalize_module_once()
                if serve_error is not None:
                    raise serve_error
                return

            self._finalize_module_once()

    async def _run_server(self) -> None:
        """Own lifecycle gates around the replaceable WebSocket serve body."""
        self._serve_task = asyncio.current_task()
        self._serve_coroutine = None
        self._serve_started.set()
        try:
            # Do not enter the server context after start() or same-loop stop()
            # has failed an as-yet-unstarted submission. Cancelling the chained
            # concurrent Future before this Task's first step can skip this
            # wrapper's finally block and strand both lifecycle gates.
            if self._stop_requested.is_set():
                return
            await self._serve()
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except BaseException as error:
            self._serve_error = error
            raise
        finally:
            retained_server = self._ws_server is not None
            if self._serve_teardown_complete.is_set() or self._force_cleanup_started.is_set():
                self._ws_server = None
            self._serve_task = None
            self._server_ready.clear()
            if (
                not self._force_cleanup_started.is_set()
                and not self._serve_context_entered.is_set()
                and not retained_server
            ):
                # No context was entered (for example, bind failed), so there
                # are no WebSocket resources left to unwind.
                self._serve_teardown_complete.set()
            self._serve_finalization_ready.set()

    async def _serve(self) -> None:
        self._stop_event = asyncio.Event()
        ws_logger = logging.getLogger("websockets.server")
        ws_logger.addFilter(_handshake_noise_filter)

        server_context = ws_server.serve(
            self._handle_client,
            host=self.host,
            port=self.port,
            ping_interval=30,
            ping_timeout=30,
            logger=ws_logger,
        )
        server = await server_context.__aenter__()
        self._ws_server = server
        self._serve_context_entered.set()
        try:
            self._server_ready.set()
            await self._stop_event.wait()
        finally:
            if not self._force_cleanup_started.is_set():
                await server_context.__aexit__(None, None, None)
                self._serve_teardown_complete.set()

    async def _handle_client(self, websocket: Any) -> None:
        if hasattr(websocket, "request") and websocket.request.path != "/ws":
            await websocket.close(1008, "Not Found")
            return
        addr = websocket.remote_address
        logger.info(f"RerunWebSocketServer: viewer connected from {addr}")
        self.client_connected.set()
        try:
            async for raw in websocket:
                self._dispatch(raw, source=websocket)
        except websockets.ConnectionClosed:
            pass
        finally:
            if self._controlling_client is websocket:
                self._controlling_client = None
                self._publish_stop()

    def _publish_stop(self) -> None:
        try:
            self.teleop_stop.publish(Bool(data=True))
        finally:
            self.tele_cmd_vel.publish(Twist.zero())

    def _dispatch(self, raw: str | bytes, source: Any | None = None) -> None:
        try:
            msg: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"RerunWebSocketServer: ignoring non-JSON message: {raw!r}")
            return

        if not isinstance(msg, dict):
            return

        msg_type = msg.get("type")

        # dict.get's default is only used when the key is missing — if Rerun
        # sends a 2D-panel click the "z" key is present with value None, and
        # `float(None)` raises.  Coerce explicitly.
        def _num(v: Any) -> float:
            return float(v) if v is not None else 0.0

        if msg_type == "click":
            self.clicked_point.publish(
                PointStamped(
                    x=_num(msg.get("x")),
                    y=_num(msg.get("y")),
                    z=_num(msg.get("z")),
                    ts=_num(msg.get("timestamp_ms")) / 1000.0,
                    # entity_path identifies the picked Rerun entity, not a
                    # coordinate frame. timestamp_ms remains source metadata;
                    # it is not a safety-ordering signal for operator STOP.
                    frame_id=self.config.click_frame_id,
                )
            )

        elif msg_type == "twist":
            self._controlling_client = source
            self.tele_cmd_vel.publish(
                Twist(
                    linear=Vector3(
                        _num(msg.get("linear_x")),
                        _num(msg.get("linear_y")),
                        _num(msg.get("linear_z")),
                    ),
                    angular=Vector3(
                        _num(msg.get("angular_x")),
                        _num(msg.get("angular_y")),
                        _num(msg.get("angular_z")),
                    ),
                )
            )

        elif msg_type == "stop":
            if self._controlling_client is source:
                self._controlling_client = None
            self._publish_stop()
