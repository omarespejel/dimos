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

"""Tests for Go2 connection routing and replay lifecycle.

The leaf (UnitreeWebRTCConnection.__init__) is covered in
dimos/robot/unitree/test_connection.py; this pins the go2-local routing.
"""

from pathlib import Path
from queue import Queue
from threading import Event, Thread
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.global_config import GlobalConfig
from dimos.memory2.store.sqlite import SqliteStore
from dimos.robot.unitree.go2 import connection as go2_conn
from dimos.robot.unitree.go2.connection import ConnectionConfig, ReplayConnection


@pytest.fixture
def stub_webrtc(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace UnitreeWebRTCConnection in go2.connection so the webrtc branch
    runs without dialing out."""
    stub = MagicMock(name="UnitreeWebRTCConnection")
    monkeypatch.setattr(go2_conn, "UnitreeWebRTCConnection", stub)
    return stub


def test_make_connection_webrtc_forwards_aes_128_key(stub_webrtc: MagicMock) -> None:
    """Webrtc branch forwards aes_128_key as a kwarg to UnitreeWebRTCConnection."""
    cfg = cast("GlobalConfig", SimpleNamespace(unitree_connection_type="webrtc"))
    go2_conn.make_connection("192.168.123.161", cfg, aes_128_key="cafe" * 8)
    stub_webrtc.assert_called_once_with(
        "192.168.123.161",
        aes_128_key="cafe" * 8,
        velocity_api=False,
    )


def test_connection_config_aes_key_defaults_from_global_config() -> None:
    """ConnectionConfig.aes_128_key defaults from GlobalConfig.unitree_aes_128_key."""
    g = GlobalConfig(robot_ip="127.0.0.1", unitree_aes_128_key="dd" * 16)
    assert ConnectionConfig(g=g).aes_128_key == "dd" * 16


def test_replay_connection_shutdown_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = MagicMock(spec=SqliteStore)
    replay = MagicMock()
    store.replay.return_value = replay
    store_factory = MagicMock(return_value=store)
    replay_path = tmp_path / "replay.db"
    resolve_db_path = MagicMock(return_value=replay_path)
    monkeypatch.setattr(go2_conn, "SqliteStore", store_factory)
    monkeypatch.setattr(go2_conn, "resolve_db_path", resolve_db_path)
    connection = ReplayConnection(dataset="recording")

    resolved_replay = connection.replay
    connection.stop()
    connection.stop()
    connection.disconnect()

    assert resolved_replay is replay
    resolve_db_path.assert_called_once_with("recording")
    store_factory.assert_called_once_with(path=str(replay_path), must_exist=True)
    store.start.assert_called_once_with()
    store.dispose.assert_called_once_with()


def test_replay_connection_rejects_first_access_after_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_factory = MagicMock()
    resolve_db_path = MagicMock()
    monkeypatch.setattr(go2_conn, "SqliteStore", store_factory)
    monkeypatch.setattr(go2_conn, "resolve_db_path", resolve_db_path)
    connection = ReplayConnection(dataset="recording")

    connection.stop()

    with pytest.raises(RuntimeError, match="replay connection is stopped"):
        _ = connection.replay
    resolve_db_path.assert_not_called()
    store_factory.assert_not_called()


def test_replay_connection_serializes_first_access_and_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_started = Event()
    store_start_finished = Event()
    release_store = Event()
    replay_returned = Event()
    stop_entered = Event()
    stop_returned = Event()
    replay_errors: Queue[BaseException] = Queue()
    stop_errors: Queue[BaseException] = Queue()
    store = MagicMock(spec=SqliteStore)
    replay = MagicMock()
    store.replay.return_value = replay

    def start_store() -> None:
        store_started.set()
        assert release_store.wait(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        store_start_finished.set()

    def dispose_store() -> None:
        assert store_start_finished.is_set()

    store.start.side_effect = start_store
    store.dispose.side_effect = dispose_store
    monkeypatch.setattr(go2_conn, "SqliteStore", MagicMock(return_value=store))
    monkeypatch.setattr(
        go2_conn,
        "resolve_db_path",
        MagicMock(return_value=tmp_path / "replay.db"),
    )
    connection = ReplayConnection(dataset="recording")

    def open_replay() -> None:
        try:
            assert connection.replay is replay
        except BaseException as error:
            replay_errors.put(error)
        finally:
            replay_returned.set()

    def stop_connection() -> None:
        stop_entered.set()
        try:
            connection.stop()
        except BaseException as error:
            stop_errors.put(error)
        finally:
            stop_returned.set()

    replay_thread = Thread(target=open_replay, daemon=True)
    stop_thread = Thread(target=stop_connection, daemon=True)
    replay_thread.start()
    try:
        assert store_started.wait(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        stop_thread.start()
        assert stop_entered.wait(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        assert not stop_returned.wait(timeout=0.1)
    finally:
        release_store.set()
        replay_thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        if stop_thread.ident is not None:
            stop_thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)

    assert not replay_thread.is_alive()
    assert not stop_thread.is_alive()
    assert replay_returned.is_set()
    assert stop_returned.is_set()
    assert replay_errors.empty()
    assert stop_errors.empty()
    store.dispose.assert_called_once_with()
    with pytest.raises(RuntimeError, match="replay connection is stopped"):
        _ = connection.replay


def test_replay_connection_repeats_structural_shutdown_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    shutdown_error = RuntimeError("store disposal failed")
    store = MagicMock(spec=SqliteStore)
    store.dispose.side_effect = shutdown_error
    store.replay.return_value = MagicMock()
    monkeypatch.setattr(go2_conn, "SqliteStore", MagicMock(return_value=store))
    monkeypatch.setattr(
        go2_conn,
        "resolve_db_path",
        MagicMock(return_value=tmp_path / "replay.db"),
    )
    connection = ReplayConnection(dataset="recording")
    _ = connection.replay

    with pytest.raises(RuntimeError, match="store disposal failed") as first:
        connection.stop()
    with pytest.raises(RuntimeError, match="store disposal failed") as second:
        connection.stop()

    assert first.value is shutdown_error
    assert second.value is shutdown_error
    store.dispose.assert_called_once_with()
