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

"""MuJoCo acceptance test for the Rerun operator-STOP path."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import threading
import time
from typing import Any

import pytest
import websockets.asyncio.client as ws_client

from dimos.core.global_config import global_config
from dimos.core.transport import PubSubTransport
from dimos.core.transport_factory import make_transport
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.protocol.service.zenohservice import default_session_pool


def _twist_message(linear_x: float) -> dict[str, float | str]:
    return {
        "type": "twist",
        "linear_x": linear_x,
        "linear_y": 0.0,
        "linear_z": 0.0,
        "angular_x": 0.0,
        "angular_y": 0.0,
        "angular_z": 0.0,
    }


async def _connect(port: int, process: subprocess.Popen[Any]) -> Any:
    deadline = time.monotonic() + 120.0
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"DimOS exited before WebSocket startup: {process.returncode}")
        try:
            return await ws_client.connect(f"ws://127.0.0.1:{port}/ws")
        except (ConnectionError, OSError, TimeoutError) as error:
            last_error = error
            await asyncio.sleep(0.25)
    raise TimeoutError(f"Rerun WebSocket did not start on port {port}: {last_error}")


async def _wait_for_websocket(port: int, process: subprocess.Popen[Any]) -> None:
    websocket = await _connect(port, process)
    await websocket.close()


async def _send_twists_and_disconnect(
    port: int,
    process: subprocess.Popen[Any],
    *,
    count: int,
    linear_x: float,
    minimum_count: int = 0,
    until: Callable[[], bool] | None = None,
) -> None:
    websocket = await _connect(port, process)
    try:
        for index in range(count):
            await websocket.send(json.dumps(_twist_message(linear_x)))
            await asyncio.sleep(0.1)
            if until is not None and index + 1 >= minimum_count and until():
                break
    finally:
        await websocket.close()


def _wait_until(predicate: Any, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return bool(predicate())


def _log_tail(path: Path, limit: int = 12_000) -> str:
    try:
        return path.read_text(errors="replace")[-limit:]
    except OSError:
        return "<DimOS log unavailable>"


def _terminate_process_group(process: subprocess.Popen[Any]) -> tuple[int | None, bool]:
    forced = False
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            forced = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10.0)
    return process.returncode, forced


def _port_can_bind(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False


@pytest.mark.skipif_in_ci
@pytest.mark.mujoco
def test_go2_rerun_disconnect_latches_operator_stop(
    tmp_path: Path,
    unused_tcp_port_factory: Callable[[], int],
) -> None:
    """A lost controlling viewer stops motion and rejects later teleop commands."""
    websocket_port = unused_tcp_port_factory()
    grpc_port = unused_tcp_port_factory()
    rerun_web_port = unused_tcp_port_factory()
    websocket_vis_port = unused_tcp_port_factory()

    lock = threading.Lock()
    commands: list[Twist] = []
    odometry: list[PoseStamped] = []

    env = os.environ.copy()
    project_bin = Path(__file__).resolve().parents[2] / ".venv" / "bin"
    env["PATH"] = os.pathsep.join((str(project_bin), env.get("PATH", "")))
    env["PYTEST_VERSION"] = pytest.__version__
    env["RERUN_WEB"] = "false"
    env["XDG_STATE_HOME"] = str(tmp_path / "state")

    dimos_executable = shutil.which("dimos", path=env["PATH"])
    assert dimos_executable is not None

    def on_command(message: Twist) -> None:
        with lock:
            commands.append(message)

    def on_odometry(message: PoseStamped) -> None:
        with lock:
            odometry.append(message)

    cmd_transport: PubSubTransport[Any] = make_transport("/cmd_vel", Twist)
    odom_transport: PubSubTransport[Any] = make_transport("/odom", PoseStamped)
    unsubscribe_cmd = cmd_transport.subscribe(on_command)
    unsubscribe_odom = odom_transport.subscribe(on_odometry)

    log_path = tmp_path / "dimos-go2-operator-stop.log"
    config_path = tmp_path / "empty-config.json"
    command = [
        dimos_executable,
        "--transport",
        global_config.transport,
        "--simulation",
        "mujoco",
        "--viewer",
        "rerun",
        "--rerun-open",
        "none",
        "--no-rerun-web",
        "--rerun-websocket-server-port",
        str(websocket_port),
        "run",
        "unitree-go2",
        "--config",
        str(config_path),
        "-o",
        "movementmanager.control_mode=manual_only",
        "-o",
        "movementmanager.latch_teleop_stop=true",
        "-o",
        "movementmanager.tele_cooldown_sec=0",
        "-o",
        f"websocketvismodule.port={websocket_vis_port}",
        "-o",
        f"rerunbridgemodule.connect_url=rerun+http://127.0.0.1:{grpc_port}/proxy",
        "-o",
        f"rerunbridgemodule.web_port={rerun_web_port}",
    ]

    process: subprocess.Popen[Any] | None = None
    returncode: int | None = None
    forced_shutdown = False
    with log_path.open("w") as log:
        try:
            process = subprocess.Popen(
                command,
                cwd=tmp_path,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            asyncio.run(_wait_for_websocket(websocket_port, process))

            assert _wait_until(lambda: bool(odometry), 120.0), _log_tail(log_path)
            with lock:
                command_start = len(commands)
                odom_start = len(odometry)
                start_pose = odometry[-1]

            def maximum_displacement() -> float:
                with lock:
                    poses = list(odometry[odom_start:])
                return max(
                    (math.hypot(pose.x - start_pose.x, pose.y - start_pose.y) for pose in poses),
                    default=0.0,
                )

            motion_distance_m = 0.02
            asyncio.run(
                _send_twists_and_disconnect(
                    websocket_port,
                    process,
                    count=60,
                    linear_x=0.3,
                    minimum_count=15,
                    until=lambda: maximum_displacement() >= motion_distance_m,
                )
            )

            def command_sequence() -> tuple[int, int] | None:
                with lock:
                    recent = list(commands[command_start:])
                nonzero = next(
                    (i for i, message in enumerate(recent) if not message.is_zero()), None
                )
                if nonzero is None:
                    return None
                zero = next(
                    (
                        i
                        for i, message in enumerate(recent[nonzero + 1 :], nonzero + 1)
                        if message.is_zero()
                    ),
                    None,
                )
                return (nonzero, zero) if zero is not None else None

            assert _wait_until(lambda: command_sequence() is not None, 10.0), _log_tail(log_path)
            sequence = command_sequence()
            assert sequence is not None
            _, stop_offset = sequence

            assert _wait_until(lambda: maximum_displacement() >= motion_distance_m, 2.0), (
                f"maximum simulated displacement was {maximum_displacement():.3f} m\n"
                f"{_log_tail(log_path)}"
            )

            with lock:
                stop_index = command_start + stop_offset

            asyncio.run(
                _send_twists_and_disconnect(
                    websocket_port,
                    process,
                    count=5,
                    linear_x=0.3,
                )
            )
            time.sleep(1.0)

            with lock:
                after_stop = list(commands[stop_index:])
            assert after_stop, _log_tail(log_path)
            assert after_stop[0].is_zero(), _log_tail(log_path)
            assert all(message.is_zero() for message in after_stop), _log_tail(log_path)
        finally:
            if process is not None:
                returncode, forced_shutdown = _terminate_process_group(process)
            unsubscribe_cmd()
            unsubscribe_odom()
            cmd_transport.stop()
            odom_transport.stop()
            if global_config.transport == "zenoh":
                default_session_pool.close_all()

    assert not forced_shutdown, _log_tail(log_path)
    assert returncode == 0, _log_tail(log_path)
    for port in (websocket_port, grpc_port, rerun_web_port, websocket_vis_port):
        assert _wait_until(lambda port=port: _port_can_bind(port), 5.0), (
            f"port {port} was not released\n{_log_tail(log_path)}"
        )
