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

import pytest
from reactivex.disposable import CompositeDisposable, Disposable

from dimos.core.resource import CompositeResource


def test_composite_resource_attempts_every_disposable_and_raises_first_error() -> None:
    resource = CompositeResource()
    events: list[str] = []
    first_error = RuntimeError("first disposal failed")
    second_error = RuntimeError("second disposal failed")

    def fail(name: str, error: RuntimeError) -> None:
        events.append(name)
        raise error

    resource.register_disposable(Disposable(lambda: events.append("before")))
    resource.register_disposable(Disposable(lambda: fail("first", first_error)))
    resource.register_disposable(Disposable(lambda: fail("second", second_error)))
    resource.register_disposable(Disposable(lambda: events.append("after")))

    with pytest.raises(RuntimeError) as raised:
        resource.stop()

    assert raised.value is first_error
    assert events == ["before", "first", "second", "after"]

    # stop() is idempotent, and late registrations are disposed immediately.
    resource.stop()
    resource.register_disposable(Disposable(lambda: events.append("late")))
    assert events == ["before", "first", "second", "after", "late"]


def test_empty_stop_then_register_disposes_immediately() -> None:
    resource = CompositeResource()
    events: list[str] = []

    resource.stop()
    resource.register_disposable(Disposable(lambda: events.append("late")))

    assert events == ["late"]


def test_first_register_racing_stop_disposes_exactly_once() -> None:
    resource = CompositeResource()
    events: list[str] = []
    ready = threading.Barrier(2)

    def register() -> None:
        ready.wait()
        resource.register_disposable(Disposable(lambda: events.append("child")))

    thread = threading.Thread(target=register)
    thread.start()
    ready.wait()
    resource.stop()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert events == ["child"]


def test_nested_composite_attempts_siblings_after_child_failure() -> None:
    resource = CompositeResource()
    events: list[str] = []
    nested_error = RuntimeError("nested child failed")

    def fail() -> None:
        events.append("nested.broken")
        raise nested_error

    resource.register_disposable(
        CompositeDisposable(
            Disposable(fail),
            Disposable(lambda: events.append("nested.healthy")),
        )
    )
    resource.register_disposable(Disposable(lambda: events.append("outer.healthy")))

    with pytest.raises(RuntimeError) as raised:
        resource.stop()

    assert raised.value is nested_error
    assert events == ["nested.broken", "nested.healthy", "outer.healthy"]
