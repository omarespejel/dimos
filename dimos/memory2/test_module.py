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

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest
from reactivex.disposable import Disposable

from dimos.core.module import ModuleConfig
from dimos.core.stream import In, Out
from dimos.memory2 import module as memory_module
from dimos.memory2.module import MemoryModule, StreamModule
from dimos.memory2.store.sqlite import SqliteStore
from dimos.memory2.stream import Stream
from dimos.memory2.transform import Transformer
from dimos.memory2.type.observation import Observation
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
