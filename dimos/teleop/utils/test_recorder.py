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

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from reactivex.disposable import Disposable

from dimos.memory2 import module as memory_module
from dimos.memory2.module import Recorder
from dimos.memory2.store.sqlite import SqliteStore
from dimos.protocol.rpc.spec import Args, RPCSpec
from dimos.teleop.utils import recorder as teleop_recorder_module
from dimos.teleop.utils.recorder import TeleopRecorder


class _TestRPC(RPCSpec):
    def __init__(self, **_kwargs: Any) -> None:
        pass

    def serve_rpc(self, _f: Any, _name: str) -> Any:
        return lambda: None

    def call(self, _name: str, _arguments: Args, _cb: Any) -> Any:
        return None

    def call_nowait(self, _name: str, _arguments: Args) -> None:
        pass


def test_teleop_recorder_leaves_store_open_until_subscriptions_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    store = MagicMock(spec=SqliteStore)
    store.start.side_effect = lambda: events.append("store-started")
    store.stop.side_effect = lambda: events.append("store-stopped")
    store.dispose.side_effect = lambda: events.append("store-disposed")
    store_factory = MagicMock(return_value=store)
    monkeypatch.setattr(memory_module, "SqliteStore", store_factory)
    monkeypatch.setattr(Recorder, "start", MagicMock())
    module = TeleopRecorder(
        db_path=tmp_path / "teleop.db",
        generate_report=False,
        rpc_transport=_TestRPC,
    )

    stopped = False
    try:
        assert module.start() is None
        module.register_disposable(Disposable(lambda: events.append("subscription")))
        module.stop()
        stopped = True
    finally:
        if not stopped:
            module.stop()

    assert events == ["store-started", "subscription", "store-stopped"]
    store_factory.assert_called_once_with(path=str(module._db_path))
    store.start.assert_called_once_with()
    store.stop.assert_called_once_with()
    store.dispose.assert_not_called()


def test_teleop_recorder_cleans_up_after_start_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    store = MagicMock(spec=SqliteStore)
    store.start.side_effect = lambda: events.append("store-started")
    store.stop.side_effect = lambda: events.append("store-stopped")
    monkeypatch.setattr(memory_module, "SqliteStore", MagicMock(return_value=store))
    generate_report = MagicMock()
    monkeypatch.setattr(teleop_recorder_module, "generate_report", generate_report)

    def fail_after_subscribing(module: Recorder) -> None:
        module.register_disposable(Disposable(lambda: events.append("subscription")))
        raise RuntimeError("recorder start failed")

    monkeypatch.setattr(Recorder, "start", fail_after_subscribing)
    module = TeleopRecorder(
        db_path=tmp_path / "teleop.db",
        generate_report=True,
        rpc_transport=_TestRPC,
    )

    try:
        with pytest.raises(RuntimeError, match="recorder start failed"):
            module.start()

        assert events == ["store-started", "subscription", "store-stopped"]
        assert module._store is None
        assert module._db_path is None
        assert module._memory_stopped.is_set()
        store.stop.assert_called_once_with()
        generate_report.assert_not_called()
    finally:
        if not module._memory_stopped.is_set():
            super(TeleopRecorder, module).stop()

    module.stop()
    generate_report.assert_not_called()
