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

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from reactivex.disposable import Disposable

from dimos.core.module import Module
from dimos.core.stream import In
from dimos.memory2 import module as memory_module
from dimos.memory2.module import Recorder
from dimos.memory2.store.sqlite import SqliteStore
from dimos.memory2.stream import Stream
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
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


def test_recorder_removes_tf_cleanup_after_subscription_setup_fails(tmp_path: Path) -> None:
    setup_error = RuntimeError("subscribe failed")
    store = MagicMock(spec=SqliteStore)
    tf_stream = MagicMock(spec=Stream)
    store.stream.return_value = tf_stream
    retained_callbacks: list[Callable[[TFMessage, Any], None]] = []
    pubsub = MagicMock()

    def fail_subscribe(
        _topic: str,
        callback: Callable[[TFMessage, Any], None],
    ) -> None:
        retained_callbacks.append(callback)
        raise setup_error

    pubsub.subscribe.side_effect = fail_subscribe
    tf = MagicMock()
    tf.config.topic = "/tf"
    tf.pubsub = pubsub
    module = Recorder(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    module._store = store
    module._tf = tf

    try:
        with pytest.raises(RuntimeError) as exc_info:
            module._record_tf()

        assert exc_info.value is setup_error
        assert module._tf_cleanup is None
        assert len(retained_callbacks) == 1
        retained_callbacks[0](TFMessage(Transform(ts=1.0)), "/tf")
        tf_stream.append.assert_not_called()
        assert not module._memory_teardown_failed
        module.stop()
    finally:
        with suppress(BaseException):
            module.stop()

    store.stop.assert_called_once_with()


def test_recorder_stop_can_win_tf_subscription_setup_failure(tmp_path: Path) -> None:
    setup_error = RuntimeError("subscribe failed")
    store = MagicMock(spec=SqliteStore)
    tf_stream = MagicMock(spec=Stream)
    store.stream.return_value = tf_stream
    subscribe_started = threading.Event()
    subscribe_release = threading.Event()
    store_stopped = threading.Event()
    retained_callbacks: list[Callable[[TFMessage, Any], None]] = []
    pubsub = MagicMock()

    def fail_subscribe(
        _topic: str,
        callback: Callable[[TFMessage, Any], None],
    ) -> None:
        retained_callbacks.append(callback)
        subscribe_started.set()
        assert subscribe_release.wait(timeout=SYNC_TIMEOUT)
        raise setup_error

    pubsub.subscribe.side_effect = fail_subscribe
    tf = MagicMock()
    tf.config.topic = "/tf"
    tf.pubsub = pubsub
    store.stop.side_effect = store_stopped.set
    module = Recorder(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    module._store = store
    module._tf = tf

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            record_future = pool.submit(module._record_tf)
            assert subscribe_started.wait(timeout=SYNC_TIMEOUT)
            stop_future = pool.submit(module.stop)
            assert store_stopped.wait(timeout=SYNC_TIMEOUT)
            subscribe_release.set()

            with pytest.raises(RuntimeError) as exc_info:
                record_future.result(timeout=SYNC_TIMEOUT)
            stop_future.result(timeout=SYNC_TIMEOUT)

        assert exc_info.value is setup_error
        assert module._tf_cleanup is None
        assert len(retained_callbacks) == 1
        retained_callbacks[0](TFMessage(Transform(ts=1.0)), "/tf")
        tf_stream.append.assert_not_called()
    finally:
        subscribe_release.set()
        with suppress(BaseException):
            module.stop()

    store.stop.assert_called_once_with()


def test_recorder_tf_setup_drain_timeout_blocks_later_store_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    setup_error = RuntimeError("subscribe failed")
    store = MagicMock(spec=SqliteStore)
    tf_stream = MagicMock(spec=Stream)
    store.stream.return_value = tf_stream
    callback_started = threading.Event()
    callback_release = threading.Event()
    callback_threads: list[threading.Thread] = []
    pubsub = MagicMock()
    message = TFMessage(Transform(ts=1.0))

    def append(*_args: Any, **_kwargs: Any) -> None:
        callback_started.set()
        assert callback_release.wait(timeout=SYNC_TIMEOUT)

    def fail_subscribe(
        _topic: str,
        callback: Callable[[TFMessage, Any], None],
    ) -> None:
        thread = threading.Thread(target=callback, args=(message, "/tf"))
        callback_threads.append(thread)
        thread.start()
        assert callback_started.wait(timeout=SYNC_TIMEOUT)
        raise setup_error

    tf_stream.append.side_effect = append
    pubsub.subscribe.side_effect = fail_subscribe
    tf = MagicMock()
    tf.config.topic = "/tf"
    tf.pubsub = pubsub
    module = Recorder(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    module._store = store
    module._tf = tf
    monkeypatch.setattr(memory_module, "_TF_DRAIN_LOG_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(memory_module, "_TF_DRAIN_TIMEOUT_SECONDS", 0.05)

    try:
        with pytest.raises(memory_module._DrainIncompleteError) as exc_info:
            module._record_tf()

        assert exc_info.value.__cause__ is setup_error
        assert module._tf_cleanup is None
        assert module._memory_stopping
        assert module._memory_teardown_failed
        assert module._store is store
        store.stop.assert_not_called()

        with pytest.raises(RuntimeError, match="teardown previously failed"):
            module.stop()
        store.stop.assert_not_called()
    finally:
        callback_release.set()
        for thread in callback_threads:
            thread.join(timeout=SYNC_TIMEOUT)
        Module.stop(module)


def test_recorder_callback_gate_error_keeps_store_open(tmp_path: Path) -> None:
    store = MagicMock(spec=SqliteStore)
    module = Recorder(
        db_path=tmp_path / "recording.db",
        record_tf=False,
        rpc_transport=_TestRPC,
    )
    module._store = store
    gate_error = RuntimeError("callback gate failed")

    def fail_gate() -> None:
        raise gate_error

    def noop() -> None:
        pass

    module._input_cleanups = [
        memory_module._RecorderCleanup(
            fail_gate,
            noop,
            noop,
        )
    ]

    try:
        with pytest.raises(
            memory_module._DrainIncompleteError,
            match="Failed to block recorder callbacks",
        ) as exc_info:
            module.stop()

        assert exc_info.value.__cause__ is gate_error
        store.stop.assert_not_called()
        assert module._store is store
        assert module._memory_teardown_failed
        assert not module._memory_stopped.is_set()
    finally:
        Module.stop(module)


def test_recorder_stop_blocks_all_inputs_before_draining_any(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = MagicMock(spec=SqliteStore)
    module = Recorder(
        db_path=tmp_path / "recording.db",
        record_tf=False,
        rpc_transport=_TestRPC,
    )
    module._store = store
    first_stream = MagicMock(spec=Stream)
    second_stream = MagicMock(spec=Stream)
    callbacks: list[Callable[[Any], Any]] = []
    first_started = threading.Event()
    first_release = threading.Event()
    second_started = threading.Event()
    first_unsubscribed = threading.Event()
    second_unsubscribed = threading.Event()
    stop_errors: list[BaseException] = []

    async def resolve_pose(name: str, _msg: Any, _ts: float) -> None:
        if name == "first":
            first_started.set()
            assert first_release.wait(timeout=SYNC_TIMEOUT)
        else:
            second_started.set()
        return None

    def make_dispatch(
        async_callback: Callable[[Any], Any],
    ) -> tuple[Callable[[Any], None], Disposable]:
        callbacks.append(async_callback)

        def dispatch(stamped: Any) -> None:
            threading.Thread(target=lambda: asyncio.run(async_callback(stamped))).start()

        return dispatch, Disposable()

    def make_input(on_dispose: Callable[[], None]) -> MagicMock:
        input_topic = MagicMock(spec=In)
        observable = MagicMock()
        stamped = MagicMock()
        input_topic.pure_observable.return_value = observable
        observable.pipe.return_value = stamped
        stamped.subscribe.return_value = Disposable(on_dispose)
        return input_topic

    def stop_module() -> None:
        try:
            module.stop()
        except BaseException as exc:
            stop_errors.append(exc)

    monkeypatch.setattr(module, "_resolve_pose", resolve_pose)
    monkeypatch.setattr(module, "_make_async_dispatch", make_dispatch)
    monkeypatch.setattr(memory_module, "_INPUT_DRAIN_TIMEOUT_SECONDS", 0.5)
    module._port_to_stream("first", make_input(first_unsubscribed.set), first_stream)
    module._port_to_stream("second", make_input(second_unsubscribed.set), second_stream)

    first_thread = threading.Thread(
        target=lambda: asyncio.run(callbacks[0]((10.0, SimpleNamespace(ts=1.0))))
    )
    second_thread: threading.Thread | None = None
    stop_thread = threading.Thread(target=stop_module)
    try:
        first_thread.start()
        assert first_started.wait(timeout=SYNC_TIMEOUT)
        stop_thread.start()
        assert first_unsubscribed.wait(timeout=SYNC_TIMEOUT)
        assert second_unsubscribed.wait(timeout=SYNC_TIMEOUT)

        second_thread = threading.Thread(
            target=lambda: asyncio.run(callbacks[1]((20.0, SimpleNamespace(ts=2.0))))
        )
        second_thread.start()
        second_thread.join(timeout=SYNC_TIMEOUT)
        assert not second_thread.is_alive()
        first_release.set()
        stop_thread.join(timeout=SYNC_TIMEOUT)

        assert not stop_thread.is_alive()
        assert not second_started.is_set()
        assert stop_errors == []
        second_stream.append.assert_not_called()
        store.stop.assert_called_once_with()
    finally:
        first_release.set()
        first_thread.join(timeout=SYNC_TIMEOUT)
        if second_thread is not None:
            second_thread.join(timeout=SYNC_TIMEOUT)
        stop_thread.join(timeout=SYNC_TIMEOUT)
        with suppress(BaseException):
            Module.stop(module)


def test_recorder_stop_waits_for_input_stream_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class InputRecorder(Recorder):
        color_image: In[int]

    store = MagicMock(spec=SqliteStore)
    stream = MagicMock(spec=Stream)
    stream_started = threading.Event()
    stream_release = threading.Event()
    stream_finished = threading.Event()
    stop_started = threading.Event()
    store_stopped = threading.Event()

    def open_stream(*_args: Any, **_kwargs: Any) -> MagicMock:
        stream_started.set()
        assert stream_release.wait(timeout=SYNC_TIMEOUT)
        stream_finished.set()
        return stream

    def stop_store() -> None:
        store_stopped.set()
        assert stream_finished.is_set()

    store.stream.side_effect = open_stream
    store.stop.side_effect = stop_store
    module = InputRecorder(
        db_path=tmp_path / "recording.db",
        record_tf=False,
        rpc_transport=_TestRPC,
    )
    module._store = store
    monkeypatch.setattr(module, "_port_to_stream", MagicMock())

    def stop_module() -> None:
        stop_started.set()
        module.stop()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            start_future = pool.submit(module.start)
            assert stream_started.wait(timeout=SYNC_TIMEOUT)
            stop_future = pool.submit(stop_module)
            try:
                assert stop_started.wait(timeout=SYNC_TIMEOUT)
                assert not store_stopped.wait(timeout=0.05)
            finally:
                stream_release.set()

            start_future.result(timeout=SYNC_TIMEOUT)
            stop_future.result(timeout=SYNC_TIMEOUT)
    finally:
        stream_release.set()
        with suppress(BaseException):
            module.stop()

    store.stop.assert_called_once_with()
    assert store_stopped.is_set()


def test_recorder_stop_waits_for_append_stream_preparation(tmp_path: Path) -> None:
    store = MagicMock(spec=SqliteStore)
    prepare_started = threading.Event()
    prepare_release = threading.Event()
    prepare_finished = threading.Event()
    stop_started = threading.Event()
    store_stopped = threading.Event()

    def delete_stream(_name: str) -> None:
        prepare_started.set()
        assert prepare_release.wait(timeout=SYNC_TIMEOUT)
        prepare_finished.set()

    def stop_store() -> None:
        store_stopped.set()
        assert prepare_finished.is_set()

    store.list_streams.return_value = ["tf"]
    store.delete_stream.side_effect = delete_stream
    store.stop.side_effect = stop_store
    module = Recorder(
        db_path=tmp_path / "recording.db",
        on_existing=memory_module.OnExisting.APPEND,
        rpc_transport=_TestRPC,
    )
    module._store = store

    def stop_module() -> None:
        stop_started.set()
        module.stop()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            prepare_future = pool.submit(module._prepare_streams)
            assert prepare_started.wait(timeout=SYNC_TIMEOUT)
            stop_future = pool.submit(stop_module)
            try:
                assert stop_started.wait(timeout=SYNC_TIMEOUT)
                assert not store_stopped.wait(timeout=0.05)
            finally:
                prepare_release.set()

            prepare_future.result(timeout=SYNC_TIMEOUT)
            stop_future.result(timeout=SYNC_TIMEOUT)
    finally:
        prepare_release.set()
        with suppress(BaseException):
            module.stop()

    store.stop.assert_called_once_with()
    store.delete_stream.assert_called_once_with("tf")
    assert store_stopped.is_set()
