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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from reactivex.disposable import Disposable

from dimos.core.global_config import GlobalConfig
from dimos.core.module import Module
from dimos.core.resource import CompositeResource
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
    cfg = SimpleNamespace(unitree_connection_type="webrtc")
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


def test_go2_replay_shutdown_disposes_subscriptions_before_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    connection = ReplayConnection(dataset="recording")
    monkeypatch.setattr(
        connection,
        "stop",
        MagicMock(side_effect=lambda: events.append("store")),
    )
    module = object.__new__(go2_conn.GO2Connection)
    module.connection = connection
    module._camera_info_thread = None
    module._camera_info_stop_event = MagicMock()
    module._disposables = None
    module.register_disposable(Disposable(lambda: events.append("subscriptions")))
    monkeypatch.setattr(Module, "stop", CompositeResource.stop)

    module.stop()

    assert events == ["subscriptions", "store"]
    module._camera_info_stop_event.set.assert_called_once_with()
