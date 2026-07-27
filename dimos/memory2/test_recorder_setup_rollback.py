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

"""Tests for Recorder setup rollback during concurrent shutdown."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from reactivex.disposable import Disposable

from dimos.core.stream import In
from dimos.memory2 import module as memory_module
from dimos.memory2.module import Recorder
from dimos.memory2.store.sqlite import SqliteStore
from dimos.memory2.stream import Stream
from dimos.protocol.rpc.spec import Args, RPCSpec


class _TestRPC(RPCSpec):
    def __init__(self, **_kwargs: Any) -> None:
        pass

    def serve_rpc(self, _f: Any, _name: str) -> Any:
        return lambda: None

    def call(self, _name: str, _arguments: Args, _cb: Any) -> Any:
        return None

    def call_nowait(self, _name: str, _arguments: Args) -> None:
        pass


SYNC_TIMEOUT: float = 2.0


def test_recorder_stop_is_not_blocked_by_dispatcher_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = MagicMock(spec=SqliteStore)
    stream = MagicMock(spec=Stream)
    input_topic = MagicMock(spec=In)
    module = Recorder(
        db_path=tmp_path / "recording.db",
        record_tf=False,
        rpc_transport=_TestRPC,
    )
    module._store = store
    bootstrap_started = threading.Event()
    bootstrap_release = threading.Event()
    dispatcher_disposed = threading.Event()
    store_stopped = threading.Event()

    def make_dispatch(
        _async_callback: Callable[[Any], Any],
    ) -> tuple[Callable[[Any], None], Disposable]:
        bootstrap_started.set()
        assert bootstrap_release.wait(timeout=2 * SYNC_TIMEOUT)
        return lambda _stamped: None, Disposable(dispatcher_disposed.set)

    monkeypatch.setattr(module, "_make_async_dispatch", make_dispatch)
    store.stop.side_effect = store_stopped.set

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            setup_future = pool.submit(
                module._port_to_stream,
                "color_image",
                input_topic,
                stream,
            )
            try:
                assert bootstrap_started.wait(timeout=SYNC_TIMEOUT)
                stop_future = pool.submit(module.stop)
                assert store_stopped.wait(timeout=SYNC_TIMEOUT)
                stop_future.result(timeout=SYNC_TIMEOUT)
                assert not setup_future.done()
            finally:
                bootstrap_release.set()

            with pytest.raises(RuntimeError, match="stopping or stopped"):
                setup_future.result(timeout=SYNC_TIMEOUT)

        assert dispatcher_disposed.is_set()
        assert module._input_cleanups == []
        assert module._memory_stopped.is_set()
        store.stop.assert_called_once_with()
    finally:
        bootstrap_release.set()
        module.stop()


def test_recorder_removes_input_cleanup_after_subscription_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class RollbackLockProbe:
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._armed = threading.Event()
            self.rollback_acquired = threading.Event()

        def arm(self) -> None:
            self._armed.set()

        def __enter__(self) -> RollbackLockProbe:
            self._lock.acquire()
            if self._armed.is_set():
                self.rollback_acquired.set()
            return self

        def __exit__(self, *_args: Any) -> None:
            self._lock.release()

    setup_error = RuntimeError("subscribe failed")
    store = MagicMock(spec=SqliteStore)
    stream = MagicMock(spec=Stream)
    input_topic = MagicMock(spec=In)
    observable = MagicMock()
    stamped_observable = MagicMock()
    input_topic.pure_observable.return_value = observable
    observable.pipe.return_value = stamped_observable
    module = Recorder(
        db_path=tmp_path / "recording.db",
        record_tf=False,
        rpc_transport=_TestRPC,
    )
    module._store = store
    lock_probe = RollbackLockProbe()
    monkeypatch.setattr(module, "_memory_stop_lock", lock_probe)
    callback_waiting = threading.Event()
    store_accessed = threading.Event()
    append_started = threading.Event()
    append_release = threading.Event()
    append_finished = threading.Event()
    store_stopped = threading.Event()
    message = SimpleNamespace(ts=1.0)

    async def resolve_pose(_name: str, _msg: Any, _ts: float) -> None:
        callback_waiting.set()
        assert lock_probe.rollback_acquired.wait(timeout=SYNC_TIMEOUT)
        assert module.store is store
        store_accessed.set()
        return None

    def append(*_args: Any, **_kwargs: Any) -> None:
        append_started.set()
        assert append_release.wait(timeout=SYNC_TIMEOUT)
        append_finished.set()

    def stop_store() -> None:
        assert append_finished.is_set()
        store_stopped.set()

    def fail_subscribe(on_next: Callable[[Any], None]) -> None:
        on_next((10.0, message))
        assert callback_waiting.wait(timeout=SYNC_TIMEOUT)
        lock_probe.arm()
        raise setup_error

    monkeypatch.setattr(module, "_resolve_pose", resolve_pose)
    monkeypatch.setattr(memory_module, "_INPUT_DRAIN_TIMEOUT_SECONDS", 0.1)
    stamped_observable.subscribe.side_effect = fail_subscribe
    stream.append.side_effect = append
    store.stop.side_effect = stop_store

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            setup_future = pool.submit(
                module._port_to_stream,
                "color_image",
                input_topic,
                stream,
            )
            assert lock_probe.rollback_acquired.wait(timeout=SYNC_TIMEOUT)
            stop_future = pool.submit(module.stop)
            assert store_accessed.wait(timeout=SYNC_TIMEOUT)
            assert append_started.wait(timeout=SYNC_TIMEOUT)
            assert not stop_future.done()
            store.stop.assert_not_called()
            assert not setup_future.done()
            append_release.set()
            with pytest.raises(RuntimeError) as exc_info:
                setup_future.result(timeout=SYNC_TIMEOUT)
            stop_future.result(timeout=SYNC_TIMEOUT)

        assert exc_info.value is setup_error
        assert module._input_cleanups == []
        assert not module._memory_teardown_failed
        assert module._memory_stopped.is_set()
    finally:
        append_release.set()
        module.stop()

    store.stop.assert_called_once_with()
    assert store_stopped.is_set()
