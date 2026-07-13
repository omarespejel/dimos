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

"""Unit tests for UnitreeWebRTCConnection.

Pure-Python test suite with no hardware or network. Covers connect() error propagation,
aes_128_key forwarding, and the UNITREE_AES_128_KEY env var via GlobalConfig.
"""

import asyncio
import json
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, call

import pytest
from unitree_webrtc_connect.constants import DATA_CHANNEL_TYPE, RTC_TOPIC, SPORT_CMD

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.global_config import GlobalConfig
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.robot.unitree import connection as conn_mod
from dimos.robot.unitree.connection import UnitreeWebRTCConnection


def _stub_driver(connect_exc: Exception | None = None) -> MagicMock:
    """A LegionConnection instance double covering everything connect() touches."""
    driver = MagicMock(name="LegionConnection-instance")
    driver.connect = AsyncMock(side_effect=connect_exc)
    driver.datachannel.disableTrafficSaving = AsyncMock()
    driver.datachannel.set_decoder = MagicMock()
    driver.datachannel.pub_sub.publish_request_new = AsyncMock()
    return driver


def test_connect_failure_propagates_to_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    """A driver connect failure must raise from the constructor, not hang."""
    driver = _stub_driver(connect_exc=RuntimeError("aes_128_key required (data2=3)"))
    monkeypatch.setattr(conn_mod, "LegionConnection", MagicMock(return_value=driver))

    with pytest.raises(RuntimeError, match="aes_128_key required"):
        UnitreeWebRTCConnection(ip="10.0.0.99")


@pytest.fixture
def built_connection(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A live UnitreeWebRTCConnection over a stubbed driver, torn down (loop
    stopped, thread joined) unconditionally so a failed assert can't leak it."""
    driver = _stub_driver()
    monkeypatch.setattr(conn_mod, "LegionConnection", MagicMock(return_value=driver))

    conn = UnitreeWebRTCConnection(ip="10.0.0.99")
    try:
        yield conn, driver
    finally:
        conn.loop.call_soon_threadsafe(conn.loop.stop)
        conn.thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)


def test_connect_success_completes_setup(built_connection: Any) -> None:
    """Happy path: constructor returns after the setup sequence ran."""
    _conn, driver = built_connection

    driver.connect.assert_awaited_once()
    driver.datachannel.pub_sub.publish_request_new.assert_awaited_once()


@pytest.fixture
def connection_factory(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Create stubbed connections and always stop their loop/thread."""
    connections: list[UnitreeWebRTCConnection] = []

    def build(**connection_options: bool) -> tuple[UnitreeWebRTCConnection, MagicMock]:
        driver = _stub_driver()
        monkeypatch.setattr(conn_mod, "LegionConnection", MagicMock(return_value=driver))
        connection = UnitreeWebRTCConnection(ip="10.0.0.99", **connection_options)
        connections.append(connection)
        return connection, driver

    try:
        yield build
    finally:
        for connection in connections:
            if connection.loop.is_running():
                connection.stop_movement()
                connection.loop.call_soon_threadsafe(connection.loop.stop)
            connection.thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)


def _run_on_connection_loop(
    connection: UnitreeWebRTCConnection,
    callback: Any,
    *args: Any,
) -> None:
    async def run_callback() -> None:
        callback(*args)

    future = asyncio.run_coroutine_threadsafe(run_callback(), connection.loop)
    future.result(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)


@pytest.mark.parametrize(
    ("connection_options", "expected_call"),
    [
        pytest.param(
            {},
            call(
                RTC_TOPIC["WIRELESS_CONTROLLER"],
                data={"lx": 0.4, "ly": 1.5, "rx": -0.8, "ry": 0},
            ),
            id="joystick",
        ),
        pytest.param(
            {"velocity_api": True},
            call(
                RTC_TOPIC["SPORT_MOD"],
                data={
                    "header": {
                        "identity": {
                            "id": ANY,
                            "api_id": SPORT_CMD["Move"],
                        }
                    },
                    "parameter": json.dumps({"x": 1.5, "y": -0.4, "z": 0.8}),
                },
                msg_type=DATA_CHANNEL_TYPE["REQUEST"],
            ),
            id="velocity",
        ),
    ],
)
def test_move_api_toggle_sends_selected_wire_command(
    connection_factory: Any,
    connection_options: dict[str, bool],
    expected_call: Any,
) -> None:
    connection, driver = connection_factory(**connection_options)
    twist = Twist(
        linear=Vector3(1.5, -0.4, 0.0),
        angular=Vector3(0.0, 0.0, 0.8),
    )

    driver.datachannel.pub_sub.publish_without_callback.reset_mock()
    assert connection.move(twist)
    driver.datachannel.pub_sub.publish_without_callback.assert_called_once_with(
        *expected_call.args,
        **expected_call.kwargs,
    )


@pytest.mark.parametrize(
    ("connection_options", "expected_call"),
    [
        pytest.param(
            {},
            call(
                RTC_TOPIC["WIRELESS_CONTROLLER"],
                data={"lx": -0.0, "ly": 0.0, "rx": -0.0, "ry": 0},
            ),
            id="joystick",
        ),
        pytest.param(
            {"velocity_api": True},
            call(
                RTC_TOPIC["SPORT_MOD"],
                data={
                    "header": {
                        "identity": {
                            "id": ANY,
                            "api_id": SPORT_CMD["Move"],
                        }
                    },
                    "parameter": json.dumps({"x": 0.0, "y": 0.0, "z": 0.0}),
                },
                msg_type=DATA_CHANNEL_TYPE["REQUEST"],
            ),
            id="velocity",
        ),
    ],
)
def test_move_watchdog_sends_zero_command(
    monkeypatch: pytest.MonkeyPatch,
    connection_factory: Any,
    connection_options: dict[str, bool],
    expected_call: Any,
) -> None:
    connection, driver = connection_factory(**connection_options)
    timer_handle = MagicMock()
    scheduled: list[tuple[Any, tuple[Any, ...]]] = []

    def call_later(_delay: float, callback: Any, *args: Any) -> MagicMock:
        scheduled.append((callback, args))
        return timer_handle

    monkeypatch.setattr(connection.loop, "call_later", call_later)
    assert connection.move(Twist(linear=Vector3(x=0.4)))
    watchdog, args = scheduled.pop()
    driver.datachannel.pub_sub.publish_without_callback.reset_mock()

    _run_on_connection_loop(connection, watchdog, *args)

    driver.datachannel.pub_sub.publish_without_callback.assert_called_once_with(
        *expected_call.args,
        **expected_call.kwargs,
    )


def test_stale_watchdog_cannot_stop_replacement_command(
    monkeypatch: pytest.MonkeyPatch,
    connection_factory: Any,
) -> None:
    connection, driver = connection_factory(velocity_api=True)
    timer_handles = [MagicMock(), MagicMock()]
    scheduled: list[tuple[Any, tuple[Any, ...]]] = []

    def call_later(_delay: float, callback: Any, *args: Any) -> MagicMock:
        scheduled.append((callback, args))
        return timer_handles[len(scheduled) - 1]

    monkeypatch.setattr(connection.loop, "call_later", call_later)
    assert connection.move(Twist(linear=Vector3(x=0.2)))
    stale_watchdog, stale_args = scheduled[0]
    assert connection.move(Twist(linear=Vector3(x=0.4)))
    timer_handles[0].cancel.assert_called_once_with()
    driver.datachannel.pub_sub.publish_without_callback.reset_mock()

    _run_on_connection_loop(connection, stale_watchdog, *stale_args)

    driver.datachannel.pub_sub.publish_without_callback.assert_not_called()
    assert connection.stop_timer is timer_handles[1]


def test_duration_refreshes_watchdog_until_final_zero(connection_factory: Any) -> None:
    connection, driver = connection_factory(velocity_api=True)
    connection.cmd_vel_timeout = 0.1
    expected_stop = json.dumps({"x": 0.0, "y": 0.0, "z": 0.0})
    driver.datachannel.pub_sub.publish_without_callback.reset_mock()

    assert connection.move(Twist(linear=Vector3(x=0.2)), duration=0.25)

    parameters = [
        mock_call.kwargs["data"]["parameter"]
        for mock_call in driver.datachannel.pub_sub.publish_without_callback.call_args_list
    ]
    assert parameters[-1] == expected_stop
    assert parameters.count(expected_stop) == 1


@pytest.fixture
def stub_legion(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace LegionConnection with a mock and no-op connect() so __init__
    stays inside the aes_128_key resolution without dialing out."""
    monkeypatch.setattr(UnitreeWebRTCConnection, "connect", lambda self: None)
    legion = MagicMock(name="LegionConnection")
    monkeypatch.setattr(conn_mod, "LegionConnection", legion)
    return legion


def _aes_kwarg(legion: MagicMock) -> Any:
    """The aes_128_key passed to LegionConnection, or None if absent."""
    return legion.call_args.kwargs.get("aes_128_key")


def test_no_key_forwards_falsy(stub_legion: MagicMock) -> None:
    """No key → a falsy value reaches the driver, which treats it as no key."""
    UnitreeWebRTCConnection(ip="192.168.123.161")
    assert not _aes_kwarg(stub_legion)


def test_aes_key_forwarded_when_provided(stub_legion: MagicMock) -> None:
    """A provided key is forwarded verbatim to the driver."""
    UnitreeWebRTCConnection(ip="192.168.123.161", aes_128_key="aa" * 16)
    assert _aes_kwarg(stub_legion) == "aa" * 16


def test_empty_string_key_forwarded_as_falsy(stub_legion: MagicMock) -> None:
    """Empty-string key stays falsy → the driver treats it as no key."""
    UnitreeWebRTCConnection(ip="192.168.123.161", aes_128_key="")
    assert not _aes_kwarg(stub_legion)


def test_global_config_reads_unitree_aes_128_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The key enters via GlobalConfig, read from the UNITREE_AES_128_KEY env var."""
    monkeypatch.setenv("UNITREE_AES_128_KEY", "ee" * 16)
    assert GlobalConfig().unitree_aes_128_key == "ee" * 16
