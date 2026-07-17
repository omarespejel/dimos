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

"""Grid tests for StreamModule — same e2e logic across all pipeline styles."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pickle
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from reactivex.disposable import Disposable
from reactivex.subject import Subject

from dimos.core.module import ModuleConfig
from dimos.core.stream import In, Out
from dimos.memory2 import module as memory_module
from dimos.memory2.module import MemoryModule, Recorder, StreamModule
from dimos.memory2.store.sqlite import SqliteStore
from dimos.memory2.stream import Stream
from dimos.memory2.transform import Transformer
from dimos.memory2.type.observation import Observation
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.protocol.rpc.spec import Args, RPCSpec

SYNC_TIMEOUT = 5.0


class _TestRPC(RPCSpec):
    def __init__(self, **_kwargs: Any) -> None:
        pass

    def serve_rpc(self, _f: Any, _name: str) -> Any:
        return lambda: None

    def call(self, _name: str, _arguments: Args, _cb: Any) -> Any:
        return None

    def call_nowait(self, _name: str, _arguments: Args) -> None:
        pass


# -- Shared transformer ---------------------------------------------------


class Double(Transformer[int, int]):
    def __init__(self, factor: int = 2) -> None:
        self.factor = factor

    def __call__(self, upstream: Iterator[Observation[int]]) -> Iterator[Observation[int]]:
        for obs in upstream:
            yield obs.derive(data=obs.data * self.factor)


# -- Pipeline styles -------------------------------------------------------


class StaticStreamModule(StreamModule[int, int]):
    """Pipeline as a static Stream chain on the class."""

    pipeline = Stream().transform(Double())
    numbers: In[int]
    doubled: Out[int]


class StaticTransformerModule(StreamModule[int, int]):
    """Pipeline as a bare Transformer on the class."""

    pipeline = Double()
    numbers: In[int]
    doubled: Out[int]


class MethodPipelineConfig(ModuleConfig):
    factor: int = 2


class MethodPipelineModule(StreamModule[int, int]):
    """Pipeline as a method with access to self.config."""

    config: MethodPipelineConfig

    def pipeline(self, stream: Stream[int]) -> Stream[int]:
        return stream.transform(Double(factor=self.config.factor))

    numbers: In[int]
    doubled: Out[int]


# -- Grid ------------------------------------------------------------------

module_cases = [
    pytest.param(StaticStreamModule, id="static-stream"),
    pytest.param(StaticTransformerModule, id="static-transformer"),
    pytest.param(MethodPipelineModule, id="method-pipeline"),
]

TFCallback = Callable[[TFMessage, Any], None]
TFRecorderFixture = tuple[Recorder, MagicMock, MagicMock, TFCallback, MagicMock]


class _ObservedLock:
    def __init__(self, lock: Any) -> None:
        self._lock = lock
        self._attempt_lock = threading.Lock()
        self._attempts = 0
        self.second_attempted = threading.Event()

    def __enter__(self) -> _ObservedLock:
        with self._attempt_lock:
            self._attempts += 1
            if self._attempts == 2:
                self.second_attempted.set()
        self._lock.acquire()
        return self

    def __exit__(self, *_args: Any) -> None:
        self._lock.release()


class _ObservedCondition:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self.stop_entered = threading.Event()

    def __enter__(self) -> _ObservedCondition:
        self._condition.acquire()
        if threading.current_thread().name.startswith("recorder-stop"):
            self.stop_entered.set()
        return self

    def __exit__(self, *_args: Any) -> None:
        self._condition.release()

    def notify_all(self) -> None:
        self._condition.notify_all()

    def wait_for(self, predicate: Callable[[], bool]) -> bool:
        return self._condition.wait_for(predicate, timeout=SYNC_TIMEOUT)


def _dispatcher_task(module: Recorder) -> asyncio.Task[Any]:
    async def find() -> asyncio.Task[Any]:
        current = asyncio.current_task()
        return next(
            task
            for task in asyncio.all_tasks()
            if task is not current
            and getattr(task.get_coro(), "__qualname__", "").endswith("dispatcher")
        )

    assert module._loop is not None
    return asyncio.run_coroutine_threadsafe(find(), module._loop).result(timeout=SYNC_TIMEOUT)


@pytest.fixture
def tf_recorder(tmp_path: Path) -> Iterator[TFRecorderFixture]:
    store = MagicMock(spec=SqliteStore)
    tf_stream = MagicMock(spec=Stream)
    store.stream.return_value = tf_stream
    unsubscribe = MagicMock()
    pubsub = MagicMock()
    pubsub.subscribe.return_value = unsubscribe
    tf = MagicMock()
    tf.config.topic = "/tf"
    tf.pubsub = pubsub
    module = Recorder(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    module._store = store
    module._tf = tf
    module._record_tf()
    callback = pubsub.subscribe.call_args.args[1]

    try:
        yield module, store, tf_stream, callback, unsubscribe
    finally:
        module.stop()


@pytest.mark.parametrize("module_cls", module_cases)
def test_blueprint_ports(module_cls: type[StreamModule[Any, Any]]) -> None:
    """All pipeline styles produce a blueprint with the correct In/Out ports."""
    bp = module_cls.blueprint()

    assert len(bp.blueprints) == 1
    atom = bp.blueprints[0]
    stream_names = {s.name for s in atom.streams}
    assert "numbers" in stream_names
    assert "doubled" in stream_names


def test_memory_module_stops_subscriptions_before_store(
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
    module = MemoryModule(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    stopped = False

    try:
        assert module.store is store
        module.register_disposable(Disposable(lambda: events.append("subscription")))

        module.stop()
        stopped = True
    finally:
        if not stopped:
            module.stop()

    assert events == ["store-started", "subscription", "store-stopped"]
    store_factory.assert_called_once_with(path=str(tmp_path / "recording.db"))
    store.start.assert_called_once_with()
    store.stop.assert_called_once_with()
    store.dispose.assert_not_called()


def test_memory_module_cleans_up_store_after_start_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = MagicMock(spec=SqliteStore)
    store.start.side_effect = RuntimeError("start failed")
    monkeypatch.setattr(memory_module, "SqliteStore", MagicMock(return_value=store))
    module = MemoryModule(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )

    try:
        with pytest.raises(RuntimeError, match="start failed"):
            assert module.store is not None

        assert module._store is None
        store.start.assert_called_once_with()
        store.stop.assert_called_once_with()
    finally:
        module.stop()


def test_recorder_stop_waits_for_active_input_append(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = MagicMock(spec=SqliteStore)
    stream = MagicMock(spec=Stream)
    subject: Subject[Any] = Subject()
    input_topic = MagicMock(spec=In)
    input_topic.pure_observable.return_value = subject
    tf = MagicMock()
    tf.get.return_value = None
    module = Recorder(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    module._store = store
    module._tf = tf
    observed_condition = _ObservedCondition()
    threading_proxy = SimpleNamespace(
        Condition=lambda: observed_condition,
        Event=threading.Event,
        Lock=threading.Lock,
        RLock=threading.RLock,
        current_thread=threading.current_thread,
    )
    with monkeypatch.context() as context:
        context.setattr(memory_module, "threading", threading_proxy)
        module._port_to_stream("color_image", input_topic, stream)

    append_started = threading.Event()
    append_release = threading.Event()
    append_finished = threading.Event()
    store_stopped = threading.Event()

    def append(*_args: Any, **_kwargs: Any) -> None:
        append_started.set()
        assert append_release.wait(timeout=SYNC_TIMEOUT)
        append_finished.set()

    def stop_store() -> None:
        assert append_finished.is_set()
        store_stopped.set()

    stream.append.side_effect = append
    store.stop.side_effect = stop_store

    try:
        subject.on_next(SimpleNamespace(ts=1.0, frame_id="camera"))
        assert append_started.wait(timeout=SYNC_TIMEOUT)

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="recorder-stop") as pool:
            stop_future = pool.submit(module.stop)
            try:
                assert observed_condition.stop_entered.wait(timeout=SYNC_TIMEOUT)
                assert not store_stopped.is_set()
                assert not stop_future.done()
            finally:
                append_release.set()
            stop_future.result(timeout=SYNC_TIMEOUT)
    finally:
        append_release.set()
        module.stop()

    stream.append.assert_called_once()
    store.stop.assert_called_once_with()
    assert store_stopped.is_set()


def test_recorder_cancels_awaiting_input_before_drain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = MagicMock(spec=SqliteStore)
    stream = MagicMock(spec=Stream)
    subject: Subject[Any] = Subject()
    input_topic = MagicMock(spec=In)
    input_topic.pure_observable.return_value = subject
    module = Recorder(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    module._store = store
    pose_started = threading.Event()
    pose_cancelled = threading.Event()

    async def wait_for_pose(_name: str, _msg: Any, _ts: float) -> None:
        pose_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            pose_cancelled.set()

    monkeypatch.setattr(module, "_resolve_pose", wait_for_pose)
    module._port_to_stream("color_image", input_topic, stream)

    try:
        subject.on_next(SimpleNamespace(ts=1.0, frame_id="camera"))
        assert pose_started.wait(timeout=SYNC_TIMEOUT)
        dispatcher_task = _dispatcher_task(module)

        with ThreadPoolExecutor(max_workers=1) as pool:
            stop_future = pool.submit(module.stop)
            stop_future.result(timeout=SYNC_TIMEOUT)
    finally:
        module.stop()

    assert pose_cancelled.is_set()
    assert dispatcher_task.done()
    stream.append.assert_not_called()
    store.stop.assert_called_once_with()


def test_recorder_stop_settles_idle_input_dispatcher(tmp_path: Path) -> None:
    store = MagicMock(spec=SqliteStore)
    stream = MagicMock(spec=Stream)
    subject: Subject[Any] = Subject()
    input_topic = MagicMock(spec=In)
    input_topic.pure_observable.return_value = subject
    module = Recorder(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    module._store = store
    module._port_to_stream("color_image", input_topic, stream)
    dispatcher_task = _dispatcher_task(module)

    try:
        module.stop()
    finally:
        module.stop()

    assert dispatcher_task.done()
    store.stop.assert_called_once_with()


def test_recorder_stop_waits_for_active_tf_callback(
    tf_recorder: TFRecorderFixture,
) -> None:
    module, store, tf_stream, callback, unsubscribe = tf_recorder
    append_started = threading.Event()
    append_release = threading.Event()
    append_finished = threading.Event()
    unsubscribed = threading.Event()
    store_stopped = threading.Event()

    def append(*_args: Any, **_kwargs: Any) -> None:
        append_started.set()
        assert append_release.wait(timeout=SYNC_TIMEOUT)
        append_finished.set()

    def stop_store() -> None:
        assert append_finished.is_set()
        store_stopped.set()

    tf_stream.append.side_effect = append
    unsubscribe.side_effect = unsubscribed.set
    store.stop.side_effect = stop_store
    transform = Transform(ts=1.0)
    message = TFMessage(transform)

    with ThreadPoolExecutor(max_workers=2) as pool:
        callback_future = pool.submit(callback, message, "/tf")
        assert append_started.wait(timeout=SYNC_TIMEOUT)
        stop_future = pool.submit(module.stop)
        try:
            assert unsubscribed.wait(timeout=SYNC_TIMEOUT)
            assert not store_stopped.is_set()
        finally:
            append_release.set()

        callback_future.result(timeout=SYNC_TIMEOUT)
        stop_future.result(timeout=SYNC_TIMEOUT)

    tf_stream.append.assert_called_once()
    recorded_message = tf_stream.append.call_args.args[0]
    assert recorded_message.transforms == [transform]
    assert tf_stream.append.call_args.kwargs == {"ts": 1.0, "pose": None}
    unsubscribe.assert_called_once_with()
    store.stop.assert_called_once_with()
    assert store_stopped.is_set()


def test_recorder_drains_all_tf_callbacks_admitted_before_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = MagicMock(spec=SqliteStore)
    tf_stream = MagicMock(spec=Stream)
    store.stream.return_value = tf_stream
    unsubscribe = MagicMock()
    pubsub = MagicMock()
    pubsub.subscribe.return_value = unsubscribe
    tf = MagicMock()
    tf.config.topic = "/tf"
    tf.pubsub = pubsub
    module = Recorder(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    module._store = store
    module._tf = tf

    append_lock = _ObservedLock(threading.Lock())
    threading_proxy = SimpleNamespace(
        Condition=threading.Condition,
        Event=threading.Event,
        Lock=lambda: append_lock,
        RLock=threading.RLock,
    )
    with monkeypatch.context() as context:
        context.setattr(memory_module, "threading", threading_proxy)
        module._record_tf()

    callback = pubsub.subscribe.call_args.args[1]
    first_append_started = threading.Event()
    first_append_release = threading.Event()
    unsubscribed = threading.Event()
    store_stopped = threading.Event()
    persisted: list[float] = []

    def append(message: TFMessage, **_kwargs: Any) -> None:
        transform = message.transforms[0]
        if transform.ts == 1.0:
            first_append_started.set()
            assert first_append_release.wait(timeout=SYNC_TIMEOUT)
        persisted.append(transform.ts)

    def stop_store() -> None:
        assert persisted == [1.0, 2.0]
        store_stopped.set()

    tf_stream.append.side_effect = append
    unsubscribe.side_effect = unsubscribed.set
    store.stop.side_effect = stop_store
    first_message = TFMessage(Transform(ts=1.0))
    second_message = TFMessage(Transform(ts=2.0))

    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            first_future = pool.submit(callback, first_message, "/tf")
            second_future = None
            stop_future = None
            try:
                assert first_append_started.wait(timeout=SYNC_TIMEOUT)
                second_future = pool.submit(callback, second_message, "/tf")
                assert append_lock.second_attempted.wait(timeout=SYNC_TIMEOUT)
                stop_future = pool.submit(module.stop)
                assert unsubscribed.wait(timeout=SYNC_TIMEOUT)
                assert not stop_future.done()
            finally:
                first_append_release.set()

            first_future.result(timeout=SYNC_TIMEOUT)
            assert second_future is not None
            second_future.result(timeout=SYNC_TIMEOUT)
            assert stop_future is not None
            stop_future.result(timeout=SYNC_TIMEOUT)
    finally:
        first_append_release.set()
        module.stop()

    assert persisted == [1.0, 2.0]
    assert tf_stream.append.call_count == 2
    unsubscribe.assert_called_once_with()
    store.stop.assert_called_once_with()
    assert store_stopped.is_set()


def test_recorder_ignores_tf_callback_entering_after_unsubscribe(
    tf_recorder: TFRecorderFixture,
) -> None:
    module, store, tf_stream, callback, unsubscribe = tf_recorder
    callback_scheduled = threading.Event()
    callback_release = threading.Event()
    message = TFMessage(Transform(ts=1.0))

    def invoke_callback() -> None:
        callback_scheduled.set()
        assert callback_release.wait(timeout=SYNC_TIMEOUT)
        callback(message, "/tf")

    with ThreadPoolExecutor(max_workers=1) as pool:
        callback_future = pool.submit(invoke_callback)
        assert callback_scheduled.wait(timeout=SYNC_TIMEOUT)
        module.stop()
        callback_release.set()
        callback_future.result(timeout=SYNC_TIMEOUT)

    tf_stream.append.assert_not_called()
    unsubscribe.assert_called_once_with()
    store.stop.assert_called_once_with()


def test_recorder_rejects_tf_callback_while_subscribe_races_stop(tmp_path: Path) -> None:
    store = MagicMock(spec=SqliteStore)
    tf_stream = MagicMock(spec=Stream)
    store.stream.return_value = tf_stream
    store_stopped = threading.Event()
    store.stop.side_effect = store_stopped.set
    subscribe_started = threading.Event()
    unsubscribe = MagicMock()
    pubsub = MagicMock()
    tf = MagicMock()
    tf.config.topic = "/tf"
    tf.pubsub = pubsub
    module = Recorder(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    module._store = store
    module._tf = tf
    message = TFMessage(Transform(ts=1.0))

    def subscribe(_topic: str, callback: TFCallback) -> MagicMock:
        subscribe_started.set()
        assert store_stopped.wait(timeout=SYNC_TIMEOUT)
        callback(message, "/tf")
        return unsubscribe

    pubsub.subscribe.side_effect = subscribe

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            start_future = pool.submit(module._record_tf)
            assert subscribe_started.wait(timeout=SYNC_TIMEOUT)
            stop_future = pool.submit(module.stop)
            stop_future.result(timeout=SYNC_TIMEOUT)
            start_future.result(timeout=SYNC_TIMEOUT)
    finally:
        module.stop()

    tf_stream.append.assert_not_called()
    unsubscribe.assert_called_once_with()
    store.stop.assert_called_once_with()


def test_memory_module_serializes_concurrent_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    close_started = threading.Event()
    close_release = threading.Event()
    store_stopped = threading.Event()
    second_started = threading.Event()
    store = MagicMock(spec=SqliteStore)

    def stop_store() -> None:
        events.append("store-stopped")
        store_stopped.set()

    store.stop.side_effect = stop_store
    monkeypatch.setattr(memory_module, "SqliteStore", MagicMock(return_value=store))
    module = MemoryModule(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    assert module.store is store
    close_rpc = module._close_rpc

    def blocking_close_rpc() -> None:
        events.append("close-started")
        close_started.set()
        assert close_release.wait(timeout=SYNC_TIMEOUT)
        events.append("close-finished")
        close_rpc()

    monkeypatch.setattr(module, "_close_rpc", blocking_close_rpc)

    def stop_again() -> None:
        second_started.set()
        module.stop()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_stop = pool.submit(module.stop)
        second_stop = None
        try:
            assert close_started.wait(timeout=SYNC_TIMEOUT)
            second_stop = pool.submit(stop_again)
            assert second_started.wait(timeout=SYNC_TIMEOUT)
            assert not store_stopped.is_set()
            assert not second_stop.done()
        finally:
            close_release.set()

        first_stop.result(timeout=SYNC_TIMEOUT)
        assert second_stop is not None
        second_stop.result(timeout=SYNC_TIMEOUT)

    assert events == ["close-started", "close-finished", "store-stopped"]
    store.stop.assert_called_once_with()


def test_memory_module_serializes_store_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    factory_entered = threading.Event()
    factory_release = threading.Event()
    second_started = threading.Event()
    store = MagicMock(spec=SqliteStore)

    def make_store(**_kwargs: Any) -> SqliteStore:
        factory_entered.set()
        assert factory_release.wait(timeout=SYNC_TIMEOUT)
        return store

    store_factory = MagicMock(side_effect=make_store)
    monkeypatch.setattr(memory_module, "SqliteStore", store_factory)
    module = MemoryModule(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )

    def get_store_again() -> SqliteStore:
        second_started.set()
        return module.store

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_store = pool.submit(lambda: module.store)
        second_store = None
        try:
            assert factory_entered.wait(timeout=SYNC_TIMEOUT)
            second_store = pool.submit(get_store_again)
            assert second_started.wait(timeout=SYNC_TIMEOUT)
            assert not second_store.done()
        finally:
            factory_release.set()

        assert first_store.result(timeout=SYNC_TIMEOUT) is store
        assert second_store is not None
        assert second_store.result(timeout=SYNC_TIMEOUT) is store

    module.stop()

    store_factory.assert_called_once_with(path=str(tmp_path / "recording.db"))
    store.start.assert_called_once_with()
    store.stop.assert_called_once_with()


def test_memory_module_stop_waits_for_store_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    start_entered = threading.Event()
    start_release = threading.Event()
    start_finished = threading.Event()
    stop_started = threading.Event()
    store_stop_called = threading.Event()
    store = MagicMock(spec=SqliteStore)

    def start_store() -> None:
        start_entered.set()
        assert start_release.wait(timeout=SYNC_TIMEOUT)
        start_finished.set()

    def stop_store() -> None:
        store_stop_called.set()
        assert start_finished.is_set()

    store.start.side_effect = start_store
    store.stop.side_effect = stop_store
    monkeypatch.setattr(memory_module, "SqliteStore", MagicMock(return_value=store))
    module = MemoryModule(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )

    def stop_module() -> None:
        stop_started.set()
        module.stop()

    with ThreadPoolExecutor(max_workers=2) as pool:
        getter = pool.submit(lambda: module.store)
        assert start_entered.wait(timeout=SYNC_TIMEOUT)

        stopper = pool.submit(stop_module)
        assert stop_started.wait(timeout=SYNC_TIMEOUT)

        try:
            assert not store_stop_called.is_set()
            assert not stopper.done()
        finally:
            start_release.set()

        assert getter.result(timeout=SYNC_TIMEOUT) is store
        stopper.result(timeout=SYNC_TIMEOUT)

    store.start.assert_called_once_with()
    store.stop.assert_called_once_with()


def test_memory_module_refuses_store_creation_after_stop_begins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    close_started = threading.Event()
    close_release = threading.Event()
    getter_started = threading.Event()
    store_factory = MagicMock()
    monkeypatch.setattr(memory_module, "SqliteStore", store_factory)
    module = MemoryModule(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    close_rpc = module._close_rpc

    def blocking_close_rpc() -> None:
        close_started.set()
        assert close_release.wait(timeout=SYNC_TIMEOUT)
        close_rpc()

    monkeypatch.setattr(module, "_close_rpc", blocking_close_rpc)

    def get_store() -> SqliteStore:
        getter_started.set()
        return module.store

    with ThreadPoolExecutor(max_workers=2) as pool:
        stop = pool.submit(module.stop)
        getter = None
        try:
            assert close_started.wait(timeout=SYNC_TIMEOUT)
            getter = pool.submit(get_store)
            assert getter_started.wait(timeout=SYNC_TIMEOUT)
            assert not getter.done()
        finally:
            close_release.set()

        stop.result(timeout=SYNC_TIMEOUT)
        assert getter is not None
        with pytest.raises(RuntimeError, match="stopping or stopped"):
            getter.result(timeout=SYNC_TIMEOUT)

    store_factory.assert_not_called()


def test_memory_module_retries_failed_store_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = MagicMock(spec=SqliteStore)
    store.stop.side_effect = [RuntimeError("close failed"), None]
    monkeypatch.setattr(memory_module, "SqliteStore", MagicMock(return_value=store))
    module = MemoryModule(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    assert module.store is store

    with pytest.raises(RuntimeError, match="close failed"):
        module.stop()

    assert module._memory_stopping
    assert not module._memory_stopped.is_set()
    assert module._store is store
    with pytest.raises(RuntimeError, match="stopping or stopped"):
        assert module.store is not None

    module.stop()

    assert store.stop.call_count == 2
    store.stop.assert_called_with()
    assert module._store is None
    assert module._memory_stopped.is_set()


def test_memory_module_restores_fresh_runtime_store_state(tmp_path: Path) -> None:
    module = MemoryModule(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    module._store = MagicMock(spec=SqliteStore)

    state = module.__getstate__()
    restored = pickle.loads(pickle.dumps(module))

    module._store = None
    module.stop()

    assert "_memory_stop_lock" not in state
    assert "_memory_stopping" not in state
    assert "_memory_stopped" not in state
    assert "_store" not in state
    assert restored._store is None
    assert not restored._memory_stopping
    assert not restored._memory_stopped.is_set()


def test_memory_module_preserves_stopped_state_when_restored(tmp_path: Path) -> None:
    module = MemoryModule(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    module.stop()

    restored = pickle.loads(pickle.dumps(module))

    assert restored._module_closed
    assert restored._memory_stopping
    assert restored._memory_stopped.is_set()
    with pytest.raises(RuntimeError, match="stopping or stopped"):
        assert restored.store is not None


def test_memory_module_pickle_waits_for_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = MemoryModule(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    stop_entered = threading.Event()
    stop_release = threading.Event()
    stop_lock = _ObservedLock(threading.RLock())
    monkeypatch.setattr(module, "_memory_stop_lock", stop_lock)
    original_stop_main = MemoryModule._stop_main

    def blocking_stop_main(self: MemoryModule) -> None:
        stop_entered.set()
        assert stop_release.wait(timeout=SYNC_TIMEOUT)
        original_stop_main(self)

    monkeypatch.setattr(MemoryModule, "_stop_main", blocking_stop_main)

    with ThreadPoolExecutor(max_workers=2) as pool:
        stop_future = pool.submit(module.stop)
        snapshot_future = None
        try:
            assert stop_entered.wait(timeout=SYNC_TIMEOUT)
            assert module._memory_stopping
            assert not module._module_closed
            snapshot_future = pool.submit(pickle.dumps, module)
            assert stop_lock.second_attempted.wait(timeout=SYNC_TIMEOUT)
            assert not snapshot_future.done()
        finally:
            stop_release.set()

        stop_future.result(timeout=SYNC_TIMEOUT)
        assert snapshot_future is not None
        restored = pickle.loads(snapshot_future.result(timeout=SYNC_TIMEOUT))

    assert restored._module_closed
    assert restored._memory_stopping
    assert restored._memory_stopped.is_set()
    with pytest.raises(RuntimeError, match="stopping or stopped"):
        assert restored.store is not None
