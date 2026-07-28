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
from contextlib import suppress
from pathlib import Path
import pickle
import threading
import time
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import ANY, MagicMock

import pytest
from reactivex import create
from reactivex.disposable import Disposable
from reactivex.subject import Subject

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.module import Module, ModuleConfig
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
from dimos.protocol.tf.tf import TFSpec


class _TestRPC(RPCSpec):
    def __init__(self, **_kwargs: Any) -> None:
        pass

    def serve_rpc(self, _f: Any, _name: str) -> Any:
        return lambda: None

    def call(self, _name: str, _arguments: Args, _cb: Any) -> Any:
        return None

    def call_nowait(self, _name: str, _arguments: Args) -> None:
        pass


class _CountingTF(TFSpec):
    instances: ClassVar[int] = 0

    def __init__(self, **kwargs: Any) -> None:
        type(self).instances += 1
        super().__init__(**kwargs)

    def publish(self, *args: Transform) -> None:
        pass

    def publish_static(self, *args: Transform) -> None:
        pass

    def get(
        self,
        parent_frame: str,
        child_frame: str,
        time_point: float | None = None,
        time_tolerance: float | None = None,
        *,
        forward_tolerance: float = 0.0,
    ) -> Transform | None:
        return None


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

TFRecorderFixture = tuple[
    Recorder,
    MagicMock,
    MagicMock,
    Callable[[TFMessage, Any], None],
    MagicMock,
]
SYNC_TIMEOUT: float = 2.0


def _recorder_cleanup(
    block: Callable[[], None] = lambda: None,
    unsubscribe: Callable[[], None] = lambda: None,
    drain: Callable[[], None] = lambda: None,
) -> memory_module._RecorderCleanup:
    return memory_module._RecorderCleanup(block, unsubscribe, drain)


def test_recorder_default_drain_budget_uses_thread_shutdown_timeout() -> None:
    drain_timeouts = (
        memory_module._INPUT_DRAIN_TIMEOUT_SECONDS,
        memory_module._TF_DRAIN_TIMEOUT_SECONDS,
    )

    assert drain_timeouts == (
        DEFAULT_THREAD_JOIN_TIMEOUT,
        DEFAULT_THREAD_JOIN_TIMEOUT,
    )
    # The CLI and worker process escalate teardown after five seconds.
    assert max(drain_timeouts) < 5.0


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

    try:
        module._store = store
        module._tf = tf
        module._record_tf()
        callback = pubsub.subscribe.call_args.args[1]
        yield module, store, tf_stream, callback, unsubscribe
    finally:
        module.stop()


def test_recorder_stop_waits_for_active_input_append(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reception_ts = 10.0
    store = MagicMock(spec=SqliteStore)
    stream = MagicMock(spec=Stream)
    input_topic = MagicMock(spec=In)
    subject: Subject[Any] = Subject()
    input_topic.pure_observable.return_value = subject
    module = Recorder(
        db_path=tmp_path / "recording.db",
        record_tf=False,
        rpc_transport=_TestRPC,
    )
    module._store = store
    append_started = threading.Event()
    append_release = threading.Event()
    append_finished = threading.Event()
    store_stopped = threading.Event()
    warning_logged = threading.Event()
    test_logger = MagicMock()

    def observe_warning(message: str, *_args: Any, **_kwargs: Any) -> None:
        if message == "Still waiting for recorder input callbacks":
            warning_logged.set()

    test_logger.warning.side_effect = observe_warning

    async def resolve_pose(_name: str, _msg: Any, _ts: float) -> None:
        return None

    def append(*_args: Any, **_kwargs: Any) -> None:
        append_started.set()
        assert append_release.wait(timeout=SYNC_TIMEOUT)
        append_finished.set()

    def stop_store() -> None:
        assert append_finished.is_set()
        store_stopped.set()

    def current_time() -> float:
        return reception_ts

    monkeypatch.setattr(module, "_resolve_pose", resolve_pose)
    monkeypatch.setattr(memory_module, "_now", current_time)
    monkeypatch.setattr(memory_module, "_INPUT_DRAIN_LOG_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(memory_module, "logger", test_logger)
    stream.append.side_effect = append
    store.stop.side_effect = stop_store
    module._port_to_stream("color_image", input_topic, stream)
    message = SimpleNamespace(ts=1.0)

    try:
        subject.on_next(message)
        assert append_started.wait(timeout=SYNC_TIMEOUT)
        with ThreadPoolExecutor(max_workers=1) as pool:
            stop_future = pool.submit(module.stop)
            try:
                assert warning_logged.wait(timeout=SYNC_TIMEOUT)
                assert not store_stopped.is_set()
                assert not stop_future.done()
            finally:
                append_release.set()

            stop_future.result(timeout=SYNC_TIMEOUT)
    finally:
        append_release.set()
        module.stop()

    stream.append.assert_called_once_with(
        message,
        ts=1.0,
        pose=None,
        tags={"reception_ts": reception_ts},
    )
    store.stop.assert_called_once_with()
    assert store_stopped.is_set()
    test_logger.warning.assert_any_call(
        "Still waiting for recorder input callbacks",
        input_name="color_image",
        active_callbacks=1,
        elapsed_seconds=ANY,
    )


def test_recorder_input_drain_reports_warning_error_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reception_ts = 10.0
    store = MagicMock(spec=SqliteStore)
    stream = MagicMock(spec=Stream)
    input_topic = MagicMock(spec=In)
    subject: Subject[Any] = Subject()
    input_topic.pure_observable.return_value = subject
    module = Recorder(
        db_path=tmp_path / "recording.db",
        record_tf=False,
        rpc_transport=_TestRPC,
    )
    module._store = store
    append_started = threading.Event()
    append_release = threading.Event()
    append_finished = threading.Event()
    store_stopped = threading.Event()
    warning_attempted = threading.Event()
    warning_error = RuntimeError("warning failed")
    test_logger = MagicMock()

    def fail_warning(message: str, *_args: Any, **_kwargs: Any) -> None:
        if message == "Still waiting for recorder input callbacks":
            warning_attempted.set()
            raise warning_error

    async def resolve_pose(_name: str, _msg: Any, _ts: float) -> None:
        return None

    def append(*_args: Any, **_kwargs: Any) -> None:
        append_started.set()
        assert append_release.wait(timeout=SYNC_TIMEOUT)
        append_finished.set()

    def stop_store() -> None:
        assert append_finished.is_set()
        store_stopped.set()

    def current_time() -> float:
        return reception_ts

    monkeypatch.setattr(module, "_resolve_pose", resolve_pose)
    monkeypatch.setattr(memory_module, "_now", current_time)
    monkeypatch.setattr(memory_module, "_INPUT_DRAIN_LOG_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(memory_module, "logger", test_logger)
    test_logger.warning.side_effect = fail_warning
    stream.append.side_effect = append
    store.stop.side_effect = stop_store
    module._port_to_stream("color_image", input_topic, stream)
    message = SimpleNamespace(ts=1.0)

    subject.on_next(message)
    assert append_started.wait(timeout=SYNC_TIMEOUT)
    with ThreadPoolExecutor(max_workers=1) as pool:
        stop_future = pool.submit(module.stop)
        try:
            assert warning_attempted.wait(timeout=SYNC_TIMEOUT)
            assert not store_stopped.is_set()
            assert not stop_future.done()
        finally:
            append_release.set()

        with pytest.raises(RuntimeError) as exc_info:
            stop_future.result(timeout=SYNC_TIMEOUT)

    assert exc_info.value is warning_error
    assert append_finished.is_set()
    store.stop.assert_called_once_with()
    assert store_stopped.is_set()
    drain_warnings = [
        call
        for call in test_logger.warning.call_args_list
        if call.args == ("Still waiting for recorder input callbacks",)
    ]
    assert len(drain_warnings) == 1
    assert drain_warnings[0].kwargs == {
        "input_name": "color_image",
        "active_callbacks": 1,
        "elapsed_seconds": ANY,
    }


def test_recorder_input_drain_wait_error_still_waits_for_callback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = MagicMock(spec=SqliteStore)
    stream = MagicMock(spec=Stream)
    input_topic = MagicMock(spec=In)
    subject: Subject[Any] = Subject()
    input_topic.pure_observable.return_value = subject
    module = Recorder(
        db_path=tmp_path / "recording.db",
        record_tf=False,
        rpc_transport=_TestRPC,
    )
    # Prevent the loop-thread join from masking an abandoned input drain.
    module._loop_thread_timeout = 0.0
    module._store = store
    append_started = threading.Event()
    append_release = threading.Event()
    append_finished = threading.Event()
    store_stopped = threading.Event()
    wait_error = RuntimeError("wait failed")
    original_wait_for = threading.Condition.wait_for
    wait_failures_remaining = 1
    stop_thread_id: int | None = None

    def wait_for_once_then_normal(
        condition: threading.Condition,
        predicate: Callable[[], bool],
        timeout: float | None = None,
    ) -> bool:
        nonlocal wait_failures_remaining
        if threading.get_ident() == stop_thread_id and wait_failures_remaining:
            wait_failures_remaining -= 1
            raise wait_error
        return original_wait_for(condition, predicate, timeout)

    def stop_module() -> None:
        nonlocal stop_thread_id
        stop_thread_id = threading.get_ident()
        module.stop()

    async def resolve_pose(_name: str, _msg: Any, _ts: float) -> None:
        return None

    def append(*_args: Any, **_kwargs: Any) -> None:
        append_started.set()
        assert append_release.wait(timeout=SYNC_TIMEOUT)
        append_finished.set()

    def stop_store() -> None:
        assert append_finished.is_set()
        store_stopped.set()

    monkeypatch.setattr(module, "_resolve_pose", resolve_pose)
    monkeypatch.setattr(threading.Condition, "wait_for", wait_for_once_then_normal)
    stream.append.side_effect = append
    store.stop.side_effect = stop_store
    module._port_to_stream("color_image", input_topic, stream)

    subject.on_next(SimpleNamespace(ts=1.0))
    assert append_started.wait(timeout=SYNC_TIMEOUT)
    with ThreadPoolExecutor(max_workers=1) as pool:
        stop_future = pool.submit(stop_module)
        try:
            assert not store_stopped.wait(timeout=0.05)
            assert not stop_future.done()
        finally:
            append_release.set()

        with pytest.raises(RuntimeError) as exc_info:
            stop_future.result(timeout=SYNC_TIMEOUT)

    assert exc_info.value is wait_error
    assert append_finished.is_set()
    store.stop.assert_called_once_with()
    assert store_stopped.is_set()


def test_recorder_input_drain_timeout_keeps_store_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = MagicMock(spec=SqliteStore)
    stream = MagicMock(spec=Stream)
    input_topic = MagicMock(spec=In)
    subject: Subject[Any] = Subject()
    input_topic.pure_observable.return_value = subject
    module = Recorder(
        db_path=tmp_path / "recording.db",
        record_tf=False,
        rpc_transport=_TestRPC,
    )
    module._store = store
    pose_started = threading.Event()
    allow_store_touch = threading.Event()
    pose_finished = threading.Event()
    stop_result: list[BaseException | None] = []
    test_logger = MagicMock()
    stop_thread: threading.Thread | None = None

    async def resolve_pose(_name: str, _msg: Any, _ts: float) -> None:
        pose_started.set()
        assert allow_store_touch.wait(timeout=SYNC_TIMEOUT)
        with pytest.raises(RuntimeError, match="stopping or stopped"):
            module.store  # noqa: B018
        pose_finished.set()
        return None

    def stop_module() -> None:
        try:
            module.stop()
        except BaseException as exc:
            stop_result.append(exc)
        else:
            stop_result.append(None)

    monkeypatch.setattr(module, "_resolve_pose", resolve_pose)
    monkeypatch.setattr(memory_module, "_INPUT_DRAIN_LOG_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(memory_module, "_INPUT_DRAIN_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(memory_module, "logger", test_logger)

    try:
        module._port_to_stream("color_image", input_topic, stream)
        subject.on_next(SimpleNamespace(ts=1.0))
        assert pose_started.wait(timeout=SYNC_TIMEOUT)

        stop_thread = threading.Thread(target=stop_module)
        stop_thread.start()
        allow_store_touch.set()
        stop_thread.join(timeout=SYNC_TIMEOUT)

        assert not stop_thread.is_alive()
        assert isinstance(stop_result[0], RuntimeError)
        assert "Timed out waiting for recorder input callbacks" in str(stop_result[0])
        assert pose_finished.wait(timeout=SYNC_TIMEOUT)
        store.stop.assert_not_called()
        assert module._store is store
        assert module._memory_teardown_failed
        assert not module._memory_stopped.is_set()
    finally:
        allow_store_touch.set()
        if stop_thread is not None:
            stop_thread.join(timeout=SYNC_TIMEOUT)
        if pose_started.is_set():
            pose_finished.wait(timeout=SYNC_TIMEOUT)
        try:
            module._before_memory_stop()
        finally:
            Module.stop(module)


def test_recorder_drain_error_takes_precedence_over_cleanup_error(
    tmp_path: Path,
) -> None:
    store = MagicMock(spec=SqliteStore)
    module = Recorder(
        db_path=tmp_path / "recording.db",
        record_tf=False,
        rpc_transport=_TestRPC,
    )
    module._store = store
    cleanup_error = RuntimeError("unsubscribe failed")
    drain_error = memory_module._DrainIncompleteError("drain still active")
    module._input_cleanups = [
        _recorder_cleanup(unsubscribe=lambda: (_ for _ in ()).throw(cleanup_error)),
        _recorder_cleanup(drain=lambda: (_ for _ in ()).throw(drain_error)),
    ]

    try:
        with pytest.raises(RuntimeError) as exc_info:
            module.stop()

        assert exc_info.value is drain_error
        assert exc_info.value.__cause__ is cleanup_error
        store.stop.assert_not_called()
        assert module._store is store
        assert module._memory_teardown_failed
        assert not module._memory_stopped.is_set()
    finally:
        Module.stop(module)


def test_recorder_input_drain_error_takes_precedence_over_subscription_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = MagicMock(spec=SqliteStore)
    stream = MagicMock(spec=Stream)
    input_topic = MagicMock(spec=In)
    source_observers: list[Any] = []
    subscription_error = RuntimeError("input dispose failed")

    def subscribe(observer: Any, _scheduler: Any) -> Disposable:
        source_observers.append(observer)
        return Disposable(lambda: (_ for _ in ()).throw(subscription_error))

    input_topic.pure_observable.return_value = create(subscribe)
    module = Recorder(
        db_path=tmp_path / "recording.db",
        record_tf=False,
        rpc_transport=_TestRPC,
    )
    module._store = store
    pose_started = threading.Event()
    allow_store_touch = threading.Event()
    pose_finished = threading.Event()
    stop_result: list[BaseException | None] = []
    stop_thread: threading.Thread | None = None

    async def resolve_pose(_name: str, _msg: Any, _ts: float) -> None:
        pose_started.set()
        assert allow_store_touch.wait(timeout=SYNC_TIMEOUT)
        with pytest.raises(RuntimeError, match="stopping or stopped"):
            module.store  # noqa: B018
        pose_finished.set()
        return None

    def stop_module() -> None:
        try:
            module.stop()
        except BaseException as exc:
            stop_result.append(exc)
        else:
            stop_result.append(None)

    monkeypatch.setattr(module, "_resolve_pose", resolve_pose)
    monkeypatch.setattr(memory_module, "_INPUT_DRAIN_LOG_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(memory_module, "_INPUT_DRAIN_TIMEOUT_SECONDS", 0.05)

    try:
        module._port_to_stream("color_image", input_topic, stream)
        source_observers[0].on_next(SimpleNamespace(ts=1.0))
        assert pose_started.wait(timeout=SYNC_TIMEOUT)

        stop_thread = threading.Thread(target=stop_module)
        stop_thread.start()
        allow_store_touch.set()
        stop_thread.join(timeout=SYNC_TIMEOUT)

        assert not stop_thread.is_alive()
        assert isinstance(stop_result[0], RuntimeError)
        assert "Timed out waiting for recorder input callbacks" in str(stop_result[0])
        assert stop_result[0].__cause__ is subscription_error
        assert pose_finished.wait(timeout=SYNC_TIMEOUT)
        store.stop.assert_not_called()
        assert module._store is store
        assert module._memory_teardown_failed
        assert not module._memory_stopped.is_set()
    finally:
        allow_store_touch.set()
        if stop_thread is not None:
            stop_thread.join(timeout=SYNC_TIMEOUT)
        if pose_started.is_set():
            pose_finished.wait(timeout=SYNC_TIMEOUT)
        with suppress(BaseException):
            module._before_memory_stop()
        Module.stop(module)


def test_recorder_stop_uses_one_drain_deadline_for_all_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_count = 4
    store = MagicMock(spec=SqliteStore)
    module = Recorder(
        db_path=tmp_path / "recording.db",
        record_tf=False,
        rpc_transport=_TestRPC,
    )
    module._store = store
    callback_started = threading.Event()
    callback_count_lock = threading.Lock()
    started_callbacks = 0
    callbacks_release = threading.Event()
    dispatched: list[Callable[[Any], None]] = []
    callback_threads: list[threading.Thread] = []
    subscription_disposed = [threading.Event() for _ in range(input_count)]
    dispatcher_disposed = [threading.Event() for _ in range(input_count)]
    drain_wait_timeouts: list[float | None] = []
    stop_thread_id = threading.get_ident()
    original_wait_for = threading.Condition.wait_for
    original_monotonic = time.monotonic
    drain_now = [original_monotonic()]

    async def resolve_pose(_name: str, _msg: Any, _ts: float) -> None:
        nonlocal started_callbacks
        with callback_count_lock:
            started_callbacks += 1
            if started_callbacks == input_count:
                callback_started.set()
        assert callbacks_release.wait(timeout=SYNC_TIMEOUT)
        return None

    def make_dispatch(
        async_callback: Callable[[Any], Any],
    ) -> tuple[Callable[[Any], None], Disposable]:
        disposed = dispatcher_disposed[len(dispatched)]

        def dispatch(stamped: Any) -> None:
            thread = threading.Thread(target=lambda: asyncio.run(async_callback(stamped)))
            callback_threads.append(thread)
            thread.start()

        dispatched.append(dispatch)
        return dispatch, Disposable(disposed.set)

    def observe_drain_wait(
        condition: threading.Condition,
        predicate: Callable[[], bool],
        timeout: float | None = None,
    ) -> bool:
        if threading.get_ident() != stop_thread_id:
            return original_wait_for(condition, predicate, timeout)
        drain_wait_timeouts.append(timeout)
        if len(drain_wait_timeouts) == 1 and timeout is not None:
            drain_now[0] += timeout
        return predicate()

    def observe_monotonic() -> float:
        if threading.get_ident() == stop_thread_id:
            return drain_now[0]
        return original_monotonic()

    monkeypatch.setattr(module, "_resolve_pose", resolve_pose)
    monkeypatch.setattr(module, "_make_async_dispatch", make_dispatch)
    monkeypatch.setattr(memory_module, "_INPUT_DRAIN_TIMEOUT_SECONDS", 0.05)

    try:
        for index in range(input_count):
            input_topic = MagicMock(spec=In)
            observable = MagicMock()
            stamped_observable = MagicMock()
            input_topic.pure_observable.return_value = observable
            observable.pipe.return_value = stamped_observable
            stamped_observable.subscribe.return_value = Disposable(subscription_disposed[index].set)
            module._port_to_stream(f"input_{index}", input_topic, MagicMock(spec=Stream))

        for dispatch in dispatched:
            dispatch((10.0, SimpleNamespace(ts=1.0)))
        assert callback_started.wait(timeout=SYNC_TIMEOUT)

        with monkeypatch.context() as drain_patch:
            drain_patch.setattr(threading.Condition, "wait_for", observe_drain_wait)
            drain_patch.setattr(time, "monotonic", observe_monotonic)
            with pytest.raises(memory_module._DrainIncompleteError):
                module.stop()

        first_timeout = drain_wait_timeouts[0]
        assert first_timeout is not None
        assert first_timeout >= 0
        assert first_timeout == pytest.approx(0.05)
        assert drain_wait_timeouts[1:] == [0.0] * (input_count - 1)
        assert all(event.is_set() for event in subscription_disposed)
        assert all(event.is_set() for event in dispatcher_disposed)
        store.stop.assert_not_called()
        assert module._store is store
        assert module._memory_teardown_failed
    finally:
        callbacks_release.set()
        for thread in callback_threads:
            thread.join(timeout=SYNC_TIMEOUT)
        with suppress(BaseException):
            module._before_memory_stop()
        Module.stop(module)


def test_recorder_setup_drain_timeout_blocks_later_store_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
    callback_started = threading.Event()
    callback_release = threading.Event()
    callback_threads: list[threading.Thread] = []
    dispatcher_disposed = threading.Event()

    async def resolve_pose(_name: str, _msg: Any, _ts: float) -> None:
        callback_started.set()
        assert callback_release.wait(timeout=SYNC_TIMEOUT)
        return None

    def make_dispatch(
        async_callback: Callable[[Any], Any],
    ) -> tuple[Callable[[Any], None], Disposable]:
        def dispatch(stamped: Any) -> None:
            thread = threading.Thread(target=lambda: asyncio.run(async_callback(stamped)))
            callback_threads.append(thread)
            thread.start()

        return dispatch, Disposable(dispatcher_disposed.set)

    def fail_subscribe(on_next: Callable[[Any], None]) -> None:
        on_next((10.0, SimpleNamespace(ts=1.0)))
        assert callback_started.wait(timeout=SYNC_TIMEOUT)
        raise setup_error

    monkeypatch.setattr(module, "_resolve_pose", resolve_pose)
    monkeypatch.setattr(module, "_make_async_dispatch", make_dispatch)
    monkeypatch.setattr(memory_module, "_INPUT_DRAIN_LOG_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(memory_module, "_INPUT_DRAIN_TIMEOUT_SECONDS", 0.05)
    stamped_observable.subscribe.side_effect = fail_subscribe

    try:
        with pytest.raises(memory_module._DrainIncompleteError) as exc_info:
            module._port_to_stream("color_image", input_topic, stream)

        assert exc_info.value.__cause__ is setup_error
        assert dispatcher_disposed.is_set()
        assert module._input_cleanups == []
        assert module._memory_stopping
        assert module._memory_teardown_failed
        assert module._store is store
        store.stop.assert_not_called()

        with pytest.raises(RuntimeError, match="teardown previously failed"):
            module.stop()

        assert module._store is store
        store.stop.assert_not_called()
    finally:
        callback_release.set()
        for thread in callback_threads:
            thread.join(timeout=SYNC_TIMEOUT)
        Module.stop(module)


def test_recorder_setup_drain_timeout_stops_existing_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    setup_error = RuntimeError("subscribe failed")
    store = MagicMock(spec=SqliteStore)
    first_stream = MagicMock(spec=Stream)
    second_stream = MagicMock(spec=Stream)
    module = Recorder(
        db_path=tmp_path / "recording.db",
        record_tf=False,
        rpc_transport=_TestRPC,
    )
    module._store = store
    second_callback_started = threading.Event()
    second_callback_release = threading.Event()
    first_pose_called = threading.Event()
    callback_threads: list[threading.Thread] = []
    async_callbacks: list[Callable[[Any], Any]] = []
    dispatcher_disposed = [threading.Event(), threading.Event()]
    first_subscription_disposed = threading.Event()
    tf_cleanup_disposed = threading.Event()
    module._tf_cleanup = _recorder_cleanup(drain=tf_cleanup_disposed.set)

    async def resolve_pose(name: str, _msg: Any, _ts: float) -> None:
        if name == "second":
            second_callback_started.set()
            assert second_callback_release.wait(timeout=SYNC_TIMEOUT)
        else:
            first_pose_called.set()
        return None

    def make_dispatch(
        async_callback: Callable[[Any], Any],
    ) -> tuple[Callable[[Any], None], Disposable]:
        disposed = dispatcher_disposed[len(async_callbacks)]
        async_callbacks.append(async_callback)

        def dispatch(stamped: Any) -> None:
            thread = threading.Thread(target=lambda: asyncio.run(async_callback(stamped)))
            callback_threads.append(thread)
            thread.start()

        return dispatch, Disposable(disposed.set)

    first_input = MagicMock(spec=In)
    first_observable = MagicMock()
    first_stamped = MagicMock()
    first_input.pure_observable.return_value = first_observable
    first_observable.pipe.return_value = first_stamped
    first_stamped.subscribe.return_value = Disposable(first_subscription_disposed.set)

    second_input = MagicMock(spec=In)
    second_observable = MagicMock()
    second_stamped = MagicMock()
    second_input.pure_observable.return_value = second_observable
    second_observable.pipe.return_value = second_stamped

    def fail_subscribe(on_next: Callable[[Any], None]) -> None:
        on_next((10.0, SimpleNamespace(ts=1.0)))
        assert second_callback_started.wait(timeout=SYNC_TIMEOUT)
        raise setup_error

    second_stamped.subscribe.side_effect = fail_subscribe
    monkeypatch.setattr(module, "_resolve_pose", resolve_pose)
    monkeypatch.setattr(module, "_make_async_dispatch", make_dispatch)
    monkeypatch.setattr(memory_module, "_INPUT_DRAIN_LOG_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(memory_module, "_INPUT_DRAIN_TIMEOUT_SECONDS", 0.05)

    try:
        module._port_to_stream("first", first_input, first_stream)

        with pytest.raises(memory_module._DrainIncompleteError) as exc_info:
            module._port_to_stream("second", second_input, second_stream)

        assert exc_info.value.__cause__ is setup_error
        assert first_subscription_disposed.is_set()
        assert all(event.is_set() for event in dispatcher_disposed)
        assert tf_cleanup_disposed.is_set()
        assert module._tf_cleanup is None
        assert module._input_cleanups == []
        assert module._memory_stopping
        assert module._memory_teardown_failed
        assert module._store is store
        store.stop.assert_not_called()

        asyncio.run(async_callbacks[0]((20.0, SimpleNamespace(ts=2.0))))
        assert not first_pose_called.is_set()
        first_stream.append.assert_not_called()
    finally:
        second_callback_release.set()
        for thread in callback_threads:
            thread.join(timeout=SYNC_TIMEOUT)
        Module.stop(module)


def test_recorder_stop_waits_for_active_pose_lookup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reception_ts = 10.0
    store = MagicMock(spec=SqliteStore)
    stream = MagicMock(spec=Stream)
    input_topic = MagicMock(spec=In)
    subject: Subject[Any] = Subject()
    input_topic.pure_observable.return_value = subject
    module = Recorder(
        db_path=tmp_path / "recording.db",
        record_tf=False,
        rpc_transport=_TestRPC,
    )
    module._store = store
    module._pose_setters = {}
    pose_started = threading.Event()
    pose_release = threading.Event()
    pose_finished = threading.Event()
    warning_logged = threading.Event()
    test_logger = MagicMock()

    def observe_warning(message: str, *_args: Any, **_kwargs: Any) -> None:
        if message == "Still waiting for recorder input callbacks":
            warning_logged.set()

    test_logger.warning.side_effect = observe_warning

    async def set_pose(_msg: Any) -> None:
        pose_started.set()
        while not pose_release.is_set():
            await asyncio.sleep(0.001)
        pose_finished.set()
        return None

    def stop_store() -> None:
        assert pose_finished.is_set()
        stream.append.assert_called_once()

    def current_time() -> float:
        return reception_ts

    monkeypatch.setattr(memory_module, "_INPUT_DRAIN_LOG_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(memory_module, "logger", test_logger)
    monkeypatch.setattr(memory_module, "_now", current_time)
    module._pose_setters = {"color_image": set_pose}
    store.stop.side_effect = stop_store
    module._port_to_stream("color_image", input_topic, stream)
    message = SimpleNamespace(ts=1.0, frame_id="camera")

    try:
        subject.on_next(message)
        assert pose_started.wait(timeout=SYNC_TIMEOUT)
        with ThreadPoolExecutor(max_workers=1) as pool:
            stop_future = pool.submit(module.stop)
            assert warning_logged.wait(timeout=SYNC_TIMEOUT)
            assert not stop_future.done()
            pose_release.set()
            stop_future.result(timeout=SYNC_TIMEOUT)
    finally:
        pose_release.set()
        module.stop()

    stream.append.assert_called_once_with(
        message,
        ts=1.0,
        pose=None,
        tags={"reception_ts": reception_ts},
    )
    store.stop.assert_called_once_with()


def test_recorder_rejects_input_append_after_stop_begins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = MagicMock(spec=SqliteStore)
    stream = MagicMock(spec=Stream)
    input_topic = MagicMock(spec=In)
    observable = MagicMock()
    stamped_observable = MagicMock()
    input_topic.pure_observable.return_value = observable
    observable.pipe.return_value = stamped_observable
    stamped_observable.subscribe.return_value = Disposable()
    module = Recorder(
        db_path=tmp_path / "recording.db",
        record_tf=False,
        rpc_transport=_TestRPC,
    )
    module._store = store
    cleanup_started = threading.Event()
    cleanup_release = threading.Event()
    store_stopped = threading.Event()
    callbacks: list[Callable[[Any], Any]] = []
    pose_called = threading.Event()

    async def resolve_pose(_name: str, _msg: Any, _ts: float) -> None:
        pose_called.set()
        return None

    def block_cleanup() -> None:
        cleanup_started.set()
        assert cleanup_release.wait(timeout=SYNC_TIMEOUT)

    def make_dispatch(
        async_callback: Callable[[Any], Any],
    ) -> tuple[Callable[[Any], None], Disposable]:
        def ignore_message(_msg: Any) -> None:
            pass

        callbacks.append(async_callback)
        return ignore_message, Disposable()

    monkeypatch.setattr(module, "_resolve_pose", resolve_pose)
    monkeypatch.setattr(module, "_make_async_dispatch", make_dispatch)
    store.stop.side_effect = store_stopped.set
    module._port_to_stream("color_image", input_topic, stream)
    module.register_disposable(Disposable(block_cleanup))
    message = SimpleNamespace(ts=1.0)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            stop_future = pool.submit(module.stop)
            assert cleanup_started.wait(timeout=SYNC_TIMEOUT)
            try:
                asyncio.run(callbacks[0]((2.0, message)))
                stream.append.assert_not_called()
                assert not pose_called.is_set()
                assert not store_stopped.is_set()
            finally:
                cleanup_release.set()

            stop_future.result(timeout=SYNC_TIMEOUT)
    finally:
        cleanup_release.set()
        module.stop()

    stream.append.assert_not_called()
    store.stop.assert_called_once_with()
    assert store_stopped.is_set()


def test_recorder_rejects_input_setup_after_stop(
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

    module.stop()
    disposables_after_stop = module._disposables

    with pytest.raises(RuntimeError, match="stopping or stopped"):
        module._port_to_stream("color_image", input_topic, stream)

    input_topic.pure_observable.assert_not_called()
    assert module._input_cleanups == []
    assert module._disposables is disposables_after_stop


def test_recorder_disposes_late_input_subscription_after_stop(
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
    setup_started = threading.Event()
    setup_release = threading.Event()
    subscription_disposed = threading.Event()
    setter_started = threading.Event()
    setter_release = threading.Event()
    setter_finished = threading.Event()
    drain_started = threading.Event()
    store_stopped = threading.Event()
    test_logger = MagicMock()

    def subscribe(observer: Any, _scheduler: Any) -> Disposable:
        observer.on_next(SimpleNamespace(ts=1.0))
        setup_started.set()
        assert setup_release.wait(timeout=SYNC_TIMEOUT)
        return Disposable(subscription_disposed.set)

    async def resolve_pose(_msg: Any) -> None:
        setter_started.set()
        while not setter_release.is_set():
            await asyncio.sleep(0.001)
        setter_finished.set()
        return None

    def observe_warning(message: str, *_args: Any, **_kwargs: Any) -> None:
        if message == "Still waiting for recorder input callbacks":
            drain_started.set()

    input_topic.pure_observable.return_value = create(subscribe)
    module._pose_setters = {"color_image": resolve_pose}
    store.stop.side_effect = store_stopped.set
    test_logger.warning.side_effect = observe_warning
    monkeypatch.setattr(memory_module, "_INPUT_DRAIN_LOG_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(memory_module, "logger", test_logger)

    def setup_recording() -> None:
        module._port_to_stream("color_image", input_topic, stream)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            setup_future = pool.submit(setup_recording)
            assert setup_started.wait(timeout=SYNC_TIMEOUT)
            assert setter_started.wait(timeout=SYNC_TIMEOUT)
            stop_future = pool.submit(module.stop)
            try:
                assert drain_started.wait(timeout=SYNC_TIMEOUT)
                assert not store_stopped.is_set()
                assert not stop_future.done()
                assert not setup_future.done()
                assert not subscription_disposed.is_set()
            finally:
                setter_release.set()

            stop_future.result(timeout=SYNC_TIMEOUT)
            assert setter_finished.is_set()
            assert store_stopped.is_set()
            assert not setup_future.done()
            assert not subscription_disposed.is_set()
            setup_release.set()
            setup_future.result(timeout=SYNC_TIMEOUT)
    finally:
        setter_release.set()
        setup_release.set()
        module.stop()

    store.stop.assert_called_once_with()
    assert store_stopped.is_set()
    assert subscription_disposed.is_set()
    stream.append.assert_called_once()


def test_recorder_input_teardown_is_ordered(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    store = MagicMock(spec=SqliteStore)
    stream = MagicMock(spec=Stream)
    input_topic = MagicMock(spec=In)
    observable = MagicMock()
    stamped_observable = MagicMock()
    input_topic.pure_observable.return_value = observable
    observable.pipe.return_value = stamped_observable

    def dispose_subscription() -> None:
        events.append("subscription")

    stamped_observable.subscribe.return_value = Disposable(dispose_subscription)
    module = Recorder(
        db_path=tmp_path / "recording.db",
        record_tf=False,
        rpc_transport=_TestRPC,
    )
    module._store = store

    def make_dispatch(
        _async_callback: Callable[[Any], Any],
    ) -> tuple[Callable[[Any], None], Disposable]:
        def ignore_message(_msg: Any) -> None:
            pass

        def dispose_dispatcher() -> None:
            events.append("dispatcher")

        return ignore_message, Disposable(dispose_dispatcher)

    def stop_store() -> None:
        events.append("store")

    monkeypatch.setattr(module, "_make_async_dispatch", make_dispatch)
    store.stop.side_effect = stop_store

    module._port_to_stream("color_image", input_topic, stream)
    module.stop()

    assert events == ["subscription", "dispatcher", "store"]


def test_recorder_preserves_reception_time_for_poseless_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reception_time = [10.0]
    callbacks: list[Callable[[Any], Any]] = []
    pending: list[Any] = []
    store = MagicMock(spec=SqliteStore)
    stream = MagicMock(spec=Stream)
    input_topic = MagicMock(spec=In)
    subject: Subject[Any] = Subject()
    input_topic.pure_observable.return_value = subject
    module = Recorder(
        db_path=tmp_path / "recording.db",
        record_tf=False,
        poseless_streams=["color_image"],
        rpc_transport=_TestRPC,
    )
    module._store = store
    test_logger = MagicMock()

    async def resolve_pose(_name: str, _msg: Any, _ts: float) -> None:
        return None

    def make_dispatch(
        async_callback: Callable[[Any], Any],
    ) -> tuple[Callable[[Any], None], Disposable]:
        callbacks.append(async_callback)
        return pending.append, Disposable()

    def current_time() -> float:
        return reception_time[0]

    monkeypatch.setattr(module, "_resolve_pose", resolve_pose)
    monkeypatch.setattr(module, "_make_async_dispatch", make_dispatch)
    monkeypatch.setattr(memory_module, "_now", current_time)
    monkeypatch.setattr(memory_module, "logger", test_logger)
    module._port_to_stream("color_image", input_topic, stream)
    message = SimpleNamespace(ts=1.0)

    try:
        subject.on_next(message)
        assert pending == [(10.0, message)]
        reception_time[0] = 20.0
        asyncio.run(callbacks[0](pending[0]))
    finally:
        module.stop()

    stream.append.assert_called_once_with(
        message,
        ts=1.0,
        pose=None,
        tags={"reception_ts": 10.0},
    )
    test_logger.warning.assert_not_called()


def test_recorder_stop_waits_for_active_tf_callback(
    monkeypatch: pytest.MonkeyPatch,
    tf_recorder: TFRecorderFixture,
) -> None:
    module, store, tf_stream, callback, unsubscribe = tf_recorder
    append_started = threading.Event()
    append_release = threading.Event()
    append_finished = threading.Event()
    unsubscribed = threading.Event()
    store_stopped = threading.Event()
    warning_logged = threading.Event()
    test_logger = MagicMock()
    test_logger.warning.side_effect = lambda *_args, **_kwargs: warning_logged.set()
    monkeypatch.setattr(memory_module, "_TF_DRAIN_LOG_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(memory_module, "logger", test_logger)

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
            assert warning_logged.wait(timeout=SYNC_TIMEOUT)
            assert not store_stopped.is_set()
        finally:
            append_release.set()

        callback_future.result(timeout=SYNC_TIMEOUT)
        stop_future.result(timeout=SYNC_TIMEOUT)

    recorded_message = tf_stream.append.call_args.args[0]
    assert recorded_message.transforms == [transform]
    assert tf_stream.append.call_args.kwargs == {"ts": 1.0, "pose": None}
    unsubscribe.assert_called_once_with()
    store.stop.assert_called_once_with()
    assert store_stopped.is_set()
    test_logger.warning.assert_called()
    assert test_logger.warning.call_args.args == ("Still waiting for tf callbacks",)
    assert test_logger.warning.call_args.kwargs["active_callbacks"] == 1


def test_recorder_tf_info_error_still_drains_active_callback(
    monkeypatch: pytest.MonkeyPatch,
    tf_recorder: TFRecorderFixture,
) -> None:
    module, store, tf_stream, callback, unsubscribe = tf_recorder
    append_started = threading.Event()
    append_release = threading.Event()
    append_finished = threading.Event()
    unsubscribed = threading.Event()
    store_stopped = threading.Event()
    info_error = RuntimeError("tf info failed")
    test_logger = MagicMock()
    test_logger.info.side_effect = info_error
    monkeypatch.setattr(memory_module, "logger", test_logger)

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
    message = TFMessage(Transform(ts=1.0))

    with ThreadPoolExecutor(max_workers=2) as pool:
        callback_future = pool.submit(callback, message, "/tf")
        assert append_started.wait(timeout=SYNC_TIMEOUT)
        stop_future = pool.submit(module.stop)
        try:
            assert unsubscribed.wait(timeout=SYNC_TIMEOUT)
            assert not store_stopped.wait(timeout=0.05)
            assert not stop_future.done()
        finally:
            append_release.set()

        callback_future.result(timeout=SYNC_TIMEOUT)
        with pytest.raises(RuntimeError) as exc_info:
            stop_future.result(timeout=SYNC_TIMEOUT)

    assert exc_info.value is info_error
    assert append_finished.is_set()
    unsubscribe.assert_called_once_with()
    store.stop.assert_called_once_with()
    assert store_stopped.is_set()


def test_recorder_tf_drain_reports_warning_error_once(
    monkeypatch: pytest.MonkeyPatch,
    tf_recorder: TFRecorderFixture,
) -> None:
    module, store, tf_stream, callback, unsubscribe = tf_recorder
    append_started = threading.Event()
    append_release = threading.Event()
    append_finished = threading.Event()
    store_stopped = threading.Event()
    warning_attempted = threading.Event()
    warning_error = RuntimeError("tf warning failed")
    test_logger = MagicMock()

    def fail_warning(*_args: Any, **_kwargs: Any) -> None:
        warning_attempted.set()
        raise warning_error

    monkeypatch.setattr(memory_module, "_TF_DRAIN_LOG_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(memory_module, "logger", test_logger)
    test_logger.warning.side_effect = fail_warning

    def append(*_args: Any, **_kwargs: Any) -> None:
        append_started.set()
        assert append_release.wait(timeout=SYNC_TIMEOUT)
        append_finished.set()

    def stop_store() -> None:
        assert append_finished.is_set()
        store_stopped.set()

    tf_stream.append.side_effect = append
    store.stop.side_effect = stop_store
    transform = Transform(ts=1.0)
    message = TFMessage(transform)

    with ThreadPoolExecutor(max_workers=2) as pool:
        callback_future = pool.submit(callback, message, "/tf")
        assert append_started.wait(timeout=SYNC_TIMEOUT)
        stop_future = pool.submit(module.stop)
        try:
            assert warning_attempted.wait(timeout=SYNC_TIMEOUT)
            assert not store_stopped.is_set()
            assert not stop_future.done()
        finally:
            append_release.set()

        callback_future.result(timeout=SYNC_TIMEOUT)
        with pytest.raises(RuntimeError) as exc_info:
            stop_future.result(timeout=SYNC_TIMEOUT)

    assert exc_info.value is warning_error
    assert append_finished.is_set()
    store.stop.assert_called_once_with()
    assert store_stopped.is_set()
    test_logger.warning.assert_called_once_with(
        "Still waiting for tf callbacks",
        active_callbacks=1,
        elapsed_seconds=ANY,
    )


def test_recorder_tf_drain_wait_error_still_waits_for_callback(
    monkeypatch: pytest.MonkeyPatch,
    tf_recorder: TFRecorderFixture,
) -> None:
    module, store, tf_stream, callback, _unsubscribe = tf_recorder
    append_started = threading.Event()
    append_release = threading.Event()
    append_finished = threading.Event()
    store_stopped = threading.Event()
    wait_error = RuntimeError("tf wait failed")
    original_wait_for = threading.Condition.wait_for
    wait_failures_remaining = 1
    stop_thread_id: int | None = None

    def wait_for_once_then_normal(
        condition: threading.Condition,
        predicate: Callable[[], bool],
        timeout: float | None = None,
    ) -> bool:
        nonlocal wait_failures_remaining
        if threading.get_ident() == stop_thread_id and wait_failures_remaining:
            wait_failures_remaining -= 1
            raise wait_error
        return original_wait_for(condition, predicate, timeout)

    def stop_module() -> None:
        nonlocal stop_thread_id
        stop_thread_id = threading.get_ident()
        module.stop()

    def append(*_args: Any, **_kwargs: Any) -> None:
        append_started.set()
        assert append_release.wait(timeout=SYNC_TIMEOUT)
        append_finished.set()

    def stop_store() -> None:
        assert append_finished.is_set()
        store_stopped.set()

    monkeypatch.setattr(threading.Condition, "wait_for", wait_for_once_then_normal)
    tf_stream.append.side_effect = append
    store.stop.side_effect = stop_store
    message = TFMessage(Transform(ts=1.0))

    with ThreadPoolExecutor(max_workers=2) as pool:
        callback_future = pool.submit(callback, message, "/tf")
        assert append_started.wait(timeout=SYNC_TIMEOUT)
        stop_future = pool.submit(stop_module)
        try:
            assert not store_stopped.wait(timeout=0.05)
            assert not stop_future.done()
        finally:
            append_release.set()

        callback_future.result(timeout=SYNC_TIMEOUT)
        with pytest.raises(RuntimeError) as exc_info:
            stop_future.result(timeout=SYNC_TIMEOUT)

    assert exc_info.value is wait_error
    assert append_finished.is_set()
    store.stop.assert_called_once_with()
    assert store_stopped.is_set()


def test_recorder_rejects_tf_callback_when_stop_races_setup(
    tmp_path: Path,
) -> None:
    store = MagicMock(spec=SqliteStore)
    tf_stream = MagicMock(spec=Stream)
    stream_started = threading.Event()
    stream_release = threading.Event()
    stop_started = threading.Event()
    store_stopped = threading.Event()
    store.stop.side_effect = store_stopped.set

    def open_stream(*_args: Any, **_kwargs: Any) -> MagicMock:
        stream_started.set()
        assert stream_release.wait(timeout=SYNC_TIMEOUT)
        return tf_stream

    store.stream.side_effect = open_stream
    unsubscribe = MagicMock()
    retained_callback: list[Callable[[TFMessage, Any], None]] = []
    pubsub = MagicMock()

    def subscribe(_topic: str, callback: Callable[[TFMessage, Any], None]) -> MagicMock:
        retained_callback.append(callback)
        return unsubscribe

    pubsub.subscribe.side_effect = subscribe
    tf = MagicMock()
    tf.config.topic = "/tf"
    tf.pubsub = pubsub
    module = Recorder(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    module._store = store
    module._tf = tf

    def stop_module() -> None:
        stop_started.set()
        module.stop()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            record_future = pool.submit(module._record_tf)
            assert stream_started.wait(timeout=SYNC_TIMEOUT)
            stop_future = pool.submit(stop_module)
            try:
                assert stop_started.wait(timeout=SYNC_TIMEOUT)
                assert not store_stopped.is_set()
            finally:
                stream_release.set()
            record_future.result(timeout=SYNC_TIMEOUT)
            stop_future.result(timeout=SYNC_TIMEOUT)

        assert len(retained_callback) == 1
        retained_callback[0](TFMessage(Transform(ts=1.0)), "/tf")
    finally:
        stream_release.set()
        module.stop()

    unsubscribe.assert_called_once_with()
    store.stop.assert_called_once_with()
    assert store_stopped.is_set()
    tf_stream.append.assert_not_called()


def test_recorder_unsubscribes_when_stop_races_subscribe(
    tmp_path: Path,
) -> None:
    store = MagicMock(spec=SqliteStore)
    tf_stream = MagicMock(spec=Stream)
    store.stream.return_value = tf_stream
    subscribe_started = threading.Event()
    subscribe_release = threading.Event()
    store_stopped = threading.Event()
    store.stop.side_effect = store_stopped.set
    unsubscribe = MagicMock()
    retained_callback: list[Callable[[TFMessage, Any], None]] = []
    pubsub = MagicMock()

    def subscribe(_topic: str, callback: Callable[[TFMessage, Any], None]) -> MagicMock:
        retained_callback.append(callback)
        subscribe_started.set()
        assert subscribe_release.wait(timeout=SYNC_TIMEOUT)
        return unsubscribe

    pubsub.subscribe.side_effect = subscribe
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
        with ThreadPoolExecutor(max_workers=2) as pool:
            record_future = pool.submit(module._record_tf)
            assert subscribe_started.wait(timeout=SYNC_TIMEOUT)
            stop_future = pool.submit(module.stop)
            try:
                assert store_stopped.wait(timeout=SYNC_TIMEOUT)
                unsubscribe.assert_not_called()
            finally:
                subscribe_release.set()

            record_future.result(timeout=SYNC_TIMEOUT)
            stop_future.result(timeout=SYNC_TIMEOUT)

        assert len(retained_callback) == 1
        retained_callback[0](TFMessage(Transform(ts=1.0)), "/tf")
    finally:
        subscribe_release.set()
        module.stop()

    unsubscribe.assert_called_once_with()
    store.stop.assert_called_once_with()
    tf_stream.append.assert_not_called()


def test_recorder_cleanup_errors_complete_shutdown_and_preserve_first_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    input_error = RuntimeError("input unsubscribe failed")
    dispatcher_error = RuntimeError("dispatcher cancellation failed")
    tf_error = RuntimeError("tf unsubscribe failed")
    store = MagicMock(spec=SqliteStore)
    tf_stream = MagicMock(spec=Stream)
    store.stream.return_value = tf_stream
    input_topic = MagicMock(spec=In)
    observable = MagicMock()
    stamped_observable = MagicMock()
    input_topic.pure_observable.return_value = observable
    observable.pipe.return_value = stamped_observable

    def fail_input_unsubscribe() -> None:
        events.append("input-unsubscribe")
        raise input_error

    stamped_observable.subscribe.return_value = Disposable(fail_input_unsubscribe)

    def fail_tf_unsubscribe() -> None:
        events.append("tf-unsubscribe")
        raise tf_error

    pubsub = MagicMock()
    pubsub.subscribe.return_value = fail_tf_unsubscribe
    tf = MagicMock()
    tf.config.topic = "/tf"
    tf.pubsub = pubsub
    module = Recorder(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    module._store = store
    module._tf = tf

    def make_dispatch(
        _async_callback: Callable[[Any], Any],
    ) -> tuple[Callable[[Any], None], Disposable]:
        def ignore_message(_msg: Any) -> None:
            pass

        def dispose_dispatcher() -> None:
            events.append("dispatcher")
            raise dispatcher_error

        return ignore_message, Disposable(dispose_dispatcher)

    def dispose_generic() -> None:
        events.append("generic")

    def stop_store() -> None:
        events.append("store")

    monkeypatch.setattr(module, "_make_async_dispatch", make_dispatch)
    module._port_to_stream("color_image", input_topic, MagicMock(spec=Stream))
    module._record_tf()
    module.register_disposable(Disposable(dispose_generic))
    store.stop.side_effect = stop_store

    with pytest.raises(RuntimeError) as exc_info:
        module.stop()

    assert exc_info.value is input_error
    assert events == [
        "input-unsubscribe",
        "tf-unsubscribe",
        "dispatcher",
        "generic",
        "store",
    ]
    store.stop.assert_called_once_with()
    assert module._store is None
    assert module._memory_stopped.is_set()
    assert module._input_cleanups == []
    assert module._tf_cleanup is None

    module.stop()

    assert events == [
        "input-unsubscribe",
        "tf-unsubscribe",
        "dispatcher",
        "generic",
        "store",
    ]
    store.stop.assert_called_once_with()


def test_recorder_rejects_tf_setup_after_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(_CountingTF, "instances", 0)
    module = Recorder(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
        tf_transport=_CountingTF,
    )

    try:
        module.stop()
        with pytest.raises(RuntimeError, match="stopping or stopped"):
            module._record_tf()
    finally:
        module.stop()

    assert _CountingTF.instances == 0


def test_recorder_restores_fresh_cleanup_state(tmp_path: Path) -> None:
    module = Recorder(
        db_path=tmp_path / "recording.db",
        record_tf=False,
        rpc_transport=_TestRPC,
    )
    module._input_cleanups.append(_recorder_cleanup())
    module._tf_cleanup = _recorder_cleanup()
    module._callback_drain_deadline = 123.0

    state = module.__getstate__()
    restored = pickle.loads(pickle.dumps(module))

    module._input_cleanups = []
    module._tf_cleanup = None
    module._callback_drain_deadline = None
    module.stop()
    restored.stop()

    assert "_input_cleanups" not in state
    assert "_tf_cleanup" not in state
    assert "_callback_drain_deadline" not in state
    assert restored._input_cleanups == []
    assert restored._tf_cleanup is None
    assert restored._callback_drain_deadline is None


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
        assert close_release.wait(timeout=2)
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
            assert close_started.wait(timeout=1)
            second_stop = pool.submit(stop_again)
            assert second_started.wait(timeout=1)
            assert not store_stopped.wait(timeout=0.1)
            assert not second_stop.done()
        finally:
            close_release.set()

        first_stop.result(timeout=2)
        assert second_stop is not None
        second_stop.result(timeout=2)

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
        assert factory_release.wait(timeout=2)
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
            assert factory_entered.wait(timeout=1)
            second_store = pool.submit(get_store_again)
            assert second_started.wait(timeout=1)
            assert not second_store.done()
        finally:
            factory_release.set()

        assert first_store.result(timeout=2) is store
        assert second_store is not None
        assert second_store.result(timeout=2) is store

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
        assert start_release.wait(timeout=2)
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
        assert start_entered.wait(timeout=1)

        stopper = pool.submit(stop_module)
        assert stop_started.wait(timeout=1)

        try:
            assert not store_stop_called.wait(timeout=0.1)
            assert not stopper.done()
        finally:
            start_release.set()

        assert getter.result(timeout=2) is store
        stopper.result(timeout=2)

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
        assert close_release.wait(timeout=2)
        close_rpc()

    monkeypatch.setattr(module, "_close_rpc", blocking_close_rpc)

    def get_store() -> SqliteStore:
        getter_started.set()
        return module.store

    with ThreadPoolExecutor(max_workers=2) as pool:
        stop = pool.submit(module.stop)
        getter = None
        try:
            assert close_started.wait(timeout=1)
            getter = pool.submit(get_store)
            assert getter_started.wait(timeout=1)
            assert not getter.done()
        finally:
            close_release.set()

        stop.result(timeout=2)
        assert getter is not None
        with pytest.raises(RuntimeError, match="stopping or stopped"):
            getter.result(timeout=2)

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
    assert not module._memory_teardown_failed
    with pytest.raises(RuntimeError, match="stopping or stopped"):
        assert module.store is not None

    module.stop()

    assert store.stop.call_count == 2
    store.stop.assert_called_with()
    assert module._store is None
    assert module._memory_stopped.is_set()
    assert not module._memory_teardown_failed


def test_memory_module_logs_store_error_after_cleanup_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cleanup_error = RuntimeError("recorder cleanup failed")
    store_error = OSError("disk gone")
    store = MagicMock(spec=SqliteStore)
    test_logger = MagicMock()
    module = MemoryModule(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    module._store = store
    monkeypatch.setattr(
        module,
        "_before_memory_stop",
        MagicMock(side_effect=cleanup_error),
    )
    monkeypatch.setattr(memory_module, "logger", test_logger)
    store.stop.side_effect = store_error

    with pytest.raises(RuntimeError) as exc_info:
        module.stop()

    assert exc_info.value is cleanup_error
    store.stop.assert_called_once_with()
    test_logger.exception.assert_called_once_with("Memory store shutdown failed during teardown")
    assert module._store is store
    assert not module._memory_stopped.is_set()


def test_memory_module_does_not_close_store_after_generic_teardown_error(
    tmp_path: Path,
) -> None:
    error = RuntimeError("generic teardown failed")
    store = MagicMock(spec=SqliteStore)
    late_cleanup = MagicMock()
    module = MemoryModule(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    module._store = store

    def fail_cleanup() -> None:
        raise error

    module.register_disposable(Disposable(fail_cleanup))
    module.register_disposable(Disposable(late_cleanup))

    try:
        with pytest.raises(RuntimeError) as exc_info:
            module.stop()

        assert exc_info.value is error
        late_cleanup.assert_not_called()
        store.stop.assert_not_called()
        assert module._store is store
        assert not module._memory_stopped.is_set()
        assert module._memory_teardown_failed
        assert not module._memory_stop_active

        with pytest.raises(RuntimeError, match="teardown previously failed"):
            module.stop()

        late_cleanup.assert_not_called()
        store.stop.assert_not_called()
        assert module._store is store
        assert not module._memory_stopped.is_set()
        assert not module._memory_stop_active
    finally:
        module._store = None
        module._close_module()


def test_memory_module_preserves_cleanup_error_and_blocks_retry_after_teardown_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cleanup_error = RuntimeError("recorder cleanup failed")
    teardown_error = RuntimeError("generic teardown failed")
    store = MagicMock(spec=SqliteStore)
    module = MemoryModule(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    module._store = store
    monkeypatch.setattr(
        module,
        "_before_memory_stop",
        MagicMock(side_effect=cleanup_error),
    )
    module.register_disposable(
        Disposable(MagicMock(side_effect=teardown_error)),
    )

    try:
        with pytest.raises(RuntimeError) as exc_info:
            module.stop()

        assert exc_info.value is cleanup_error
        assert module._memory_teardown_failed
        assert not module._memory_stop_active
        store.stop.assert_not_called()

        with pytest.raises(RuntimeError, match="teardown previously failed"):
            module.stop()

        store.stop.assert_not_called()
        assert module._store is store
        assert not module._memory_stopped.is_set()
        assert not module._memory_stop_active
    finally:
        module._store = None
        module._close_module()


def test_recorder_recursive_stop_from_input_cleanup_closes_store_last(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    store = MagicMock(spec=SqliteStore)
    store.stop.side_effect = lambda: events.append("store")
    module = Recorder(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    module._store = store

    def recursive_cleanup() -> None:
        events.append("cleanup-start")
        module.stop()
        events.append("cleanup-finish")

    module._input_cleanups.append(_recorder_cleanup(unsubscribe=recursive_cleanup))

    module.stop()

    assert events == ["cleanup-start", "cleanup-finish", "store"]
    store.stop.assert_called_once_with()
    assert module._store is None
    assert module._memory_stopped.is_set()
    assert not module._memory_stop_active


def test_memory_module_recursive_stop_from_generic_cleanup_closes_store_last(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    store = MagicMock(spec=SqliteStore)
    store.stop.side_effect = lambda: events.append("store")
    module = MemoryModule(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    module._store = store

    def recursive_cleanup() -> None:
        events.append("cleanup-start")
        module.stop()
        events.append("cleanup-finish")

    module.register_disposable(Disposable(recursive_cleanup))

    module.stop()

    assert events == ["cleanup-start", "cleanup-finish", "store"]
    store.stop.assert_called_once_with()
    assert module._store is None
    assert module._memory_stopped.is_set()
    assert not module._memory_stop_active


def test_memory_module_recursive_stop_from_store_close_does_not_reenter(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    store = MagicMock(spec=SqliteStore)
    module = MemoryModule(
        db_path=tmp_path / "recording.db",
        rpc_transport=_TestRPC,
    )
    module._store = store

    def recursive_store_stop() -> None:
        events.append("store-start")
        module.stop()
        events.append("store-finish")

    store.stop.side_effect = recursive_store_stop

    module.stop()

    assert events == ["store-start", "store-finish"]
    store.stop.assert_called_once_with()
    assert module._store is None
    assert module._memory_stopped.is_set()
    assert not module._memory_stop_active


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
    assert "_memory_stop_active" not in state
    assert "_memory_teardown_failed" not in state
    assert "_store" not in state
    assert restored._store is None
    assert not restored._memory_stopping
    assert not restored._memory_stopped.is_set()
    assert not restored._memory_stop_active
    assert not restored._memory_teardown_failed


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
    assert not restored._memory_stop_active
    assert not restored._memory_teardown_failed
    with pytest.raises(RuntimeError, match="stopping or stopped"):
        assert restored.store is not None
