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

import threading
from threading import Lock
from typing import Any, cast

import pytest
from reactivex.disposable import Disposable

from dimos.core.module import ModuleBase
from dimos.core.stream import In, Out, Transport


class _BareModule(ModuleBase):
    input: In[int]
    output: Out[int]


class _StopProbe:
    def __init__(
        self,
        events: list[str],
        name: str,
        error: BaseException | None = None,
    ) -> None:
        self._events = events
        self._name = name
        self._error = error

    def stop(self) -> None:
        self._events.append(self._name)
        if self._error is not None:
            raise self._error


class _LoopProbe:
    def __init__(self, events: list[str], error: BaseException | None = None) -> None:
        self._events = events
        self._error = error

    def stop(self) -> None:
        self._events.append("loop.stop")

    def call_soon_threadsafe(self, callback: Any) -> None:
        self._events.append("loop.schedule_stop")
        if self._error is not None:
            raise self._error
        callback()


class _ThreadProbe:
    def __init__(self, events: list[str], *, stops: bool = True) -> None:
        self._events = events
        self._alive = True
        self._stops = stops

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self._events.append(f"loop.join:{timeout}")
        if self._stops:
            self._alive = False


def _bare_module() -> _BareModule:
    module = object.__new__(_BareModule)
    module._module_closed_lock = threading.RLock()
    module._module_closed = False
    module._module_close_error = None
    module._module_stop_lock = threading.RLock()
    module._module_stop_started = False
    module._module_stop_error = None
    module._tools_lock = Lock()
    module._tools = {}
    module._disposables = None
    module._main_gen = None
    module._loop = None
    module._loop_thread = None
    module._tf = None
    module.rpc = None  # type: ignore[assignment]
    return module


def test_close_module_attempts_all_phases_and_raises_first_error() -> None:
    events: list[str] = []
    rpc_error = RuntimeError("rpc stop failed")
    loop_error = RuntimeError("loop stop failed")
    tf_error = RuntimeError("tf stop failed")
    input_error = RuntimeError("input stop failed")
    module = _bare_module()

    module._tools = {
        "broken": _StopProbe(events, "tool.broken", RuntimeError("tool stop failed")),
        "healthy": _StopProbe(events, "tool.healthy"),
    }
    module.rpc = _StopProbe(events, "rpc", rpc_error)  # type: ignore[assignment]
    module._loop = cast("Any", _LoopProbe(events, loop_error))
    module._loop_thread = cast("Any", _ThreadProbe(events))
    module._loop_thread_timeout = 0.25
    module._tf = cast("Any", _StopProbe(events, "tf", tf_error))

    input_transport = _StopProbe(events, "input", input_error)
    output_transport = _StopProbe(events, "output")
    module.input = In(
        int,
        "input",
        module,
        cast("Transport[int]", input_transport),
    )
    module.output = Out(
        int,
        "output",
        module,
        cast("Transport[int]", output_transport),
    )

    with pytest.raises(RuntimeError) as raised:
        module._close_module()

    assert raised.value is rpc_error
    assert events == [
        "tool.broken",
        "tool.healthy",
        "rpc",
        "loop.schedule_stop",
        "loop.join:0.25",
        "tf",
        "input",
        "output",
    ]
    assert module.rpc is None
    assert module._loop is None
    assert module._loop_thread is None
    assert module._tf is None
    assert module.input.owner is None
    assert module.output.owner is None

    # A repeated close does not run any phase twice and replays the same error.
    with pytest.raises(RuntimeError, match="rpc stop failed") as repeated:
        module._close_module()
    assert repeated.value is rpc_error
    assert len(events) == 8


def test_module_stop_closes_runtime_after_disposable_failure() -> None:
    events: list[str] = []
    disposable_error = RuntimeError("subscription disposal failed")
    module = _bare_module()
    module.rpc = _StopProbe(events, "rpc")  # type: ignore[assignment]

    def fail_disposal() -> None:
        events.append("disposable.broken")
        raise disposable_error

    module.register_disposable(Disposable(fail_disposal))
    module.register_disposable(Disposable(lambda: events.append("disposable.healthy")))

    with pytest.raises(RuntimeError) as raised:
        module.stop()

    assert raised.value is disposable_error
    assert events == ["disposable.broken", "disposable.healthy", "rpc"]
    assert module.rpc is None
    assert module._module_closed is True


def test_close_module_reports_event_loop_join_timeout_after_other_cleanup() -> None:
    events: list[str] = []
    module = _bare_module()
    module._loop = cast("Any", _LoopProbe(events))
    module._loop_thread = cast("Any", _ThreadProbe(events, stops=False))
    module._loop_thread_timeout = 0.1
    module._tf = cast("Any", _StopProbe(events, "tf"))

    with pytest.raises(TimeoutError, match="event-loop thread did not stop"):
        module._close_module()

    assert events == ["loop.schedule_stop", "loop.stop", "loop.join:0.1", "tf"]
    assert module._loop is None
    assert module._loop_thread is None
    assert module._tf is None


def test_close_module_does_not_join_its_current_event_loop_thread() -> None:
    events: list[str] = []
    module = _bare_module()
    module._loop = cast("Any", _LoopProbe(events))
    module._loop_thread = threading.current_thread()

    module._close_module()

    assert events == ["loop.schedule_stop", "loop.stop"]
    assert module._loop is None
    assert module._loop_thread is None


def test_concurrent_module_stop_waits_and_replays_sticky_error() -> None:
    module = _bare_module()
    dispose_entered = threading.Event()
    release_dispose = threading.Event()
    stop_error = RuntimeError("disposal failed")
    errors: list[BaseException] = []

    def block_then_fail() -> None:
        dispose_entered.set()
        assert release_dispose.wait(timeout=2.0)
        raise stop_error

    def stop() -> None:
        try:
            module.stop()
        except BaseException as error:
            errors.append(error)

    module.register_disposable(Disposable(block_then_fail))
    owner = threading.Thread(target=stop)
    waiter = threading.Thread(target=stop)
    owner.start()
    assert dispose_entered.wait(timeout=2.0)
    waiter.start()
    assert waiter.is_alive()

    release_dispose.set()
    owner.join(timeout=2.0)
    waiter.join(timeout=2.0)

    assert not owner.is_alive()
    assert not waiter.is_alive()
    assert len(errors) == 2
    assert all(error is stop_error for error in errors)
    with pytest.raises(RuntimeError, match="disposal failed") as repeated:
        module.stop()
    assert repeated.value is stop_error


def test_concurrent_module_stop_waits_for_successful_owner() -> None:
    module = _bare_module()
    dispose_entered = threading.Event()
    release_dispose = threading.Event()
    owner_done = threading.Event()
    waiter_done = threading.Event()
    dispose_calls = 0

    def block_disposal() -> None:
        nonlocal dispose_calls
        dispose_calls += 1
        dispose_entered.set()
        assert release_dispose.wait(timeout=2.0)

    module.register_disposable(Disposable(block_disposal))
    owner = threading.Thread(target=lambda: (module.stop(), owner_done.set()))
    waiter = threading.Thread(target=lambda: (module.stop(), waiter_done.set()))
    owner.start()
    assert dispose_entered.wait(timeout=2.0)
    waiter.start()
    assert not waiter_done.wait(timeout=0.05)

    release_dispose.set()
    owner.join(timeout=2.0)
    waiter.join(timeout=2.0)

    assert not owner.is_alive()
    assert not waiter.is_alive()
    assert owner_done.is_set()
    assert waiter_done.is_set()
    assert dispose_calls == 1
    module.stop()
    assert dispose_calls == 1


def test_module_state_restore_reinitializes_lifecycle_guards() -> None:
    module = _bare_module()
    module._module_closed = True
    module._module_close_error = RuntimeError("old close error")
    module._module_stop_started = True
    module._module_stop_error = RuntimeError("old stop error")
    module.register_disposable(Disposable(lambda: None))

    state = module.__getstate__()

    for key in (
        "_disposables",
        "_module_closed_lock",
        "_module_close_error",
        "_module_stop_lock",
        "_module_stop_started",
        "_module_stop_error",
        "_module_closed",
    ):
        assert key not in state

    restored = object.__new__(_BareModule)
    restored.__setstate__(state)

    assert restored._disposables is None
    assert restored._module_closed is False
    assert restored._module_close_error is None
    assert restored._module_stop_started is False
    assert restored._module_stop_error is None
    assert restored._loop is None
    assert restored._loop_thread is None
