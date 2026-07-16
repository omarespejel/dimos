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

from queue import Queue
import threading
from typing import TYPE_CHECKING

import pytest
from reactivex.abc import DisposableBase

from dimos.memory2.replay import ReplayStream
from dimos.memory2.store.base import StreamAccessor

if TYPE_CHECKING:
    from dimos.memory2.store.base import Store
    from dimos.memory2.store.memory import MemoryStore
    from dimos.memory2.store.sqlite import SqliteStore


def _populate(store: Store, name: str, timestamps: list[float]) -> None:
    """Append integer payloads at each given ts to a named stream."""
    s = store.stream(name, int)
    for i, ts in enumerate(timestamps):
        s.append(i, ts=ts)


def test_streams_accessor_equivalence(sqlite_store: SqliteStore) -> None:
    _populate(sqlite_store, "lidar", [0.0, 0.1, 0.2])
    replay = sqlite_store.replay()
    assert isinstance(replay.streams, StreamAccessor)
    by_attr = replay.streams.lidar
    by_method: ReplayStream[int] = replay.stream("lidar")
    assert isinstance(by_attr, ReplayStream)
    assert isinstance(by_method, ReplayStream)
    assert by_attr.name == by_method.name == "lidar"


def test_first_ts_across_streams(sqlite_store: SqliteStore) -> None:
    _populate(sqlite_store, "lidar", [10.0, 11.0])
    _populate(sqlite_store, "odom", [5.0, 6.0])
    replay = sqlite_store.replay()
    assert replay.first_ts() == 5.0


def test_first_ts_empty_store(sqlite_store: SqliteStore) -> None:
    replay = sqlite_store.replay()
    assert replay.first_ts() is None


def test_seek_filters_frames_before_offset(sqlite_store: SqliteStore) -> None:
    _populate(sqlite_store, "lidar", [100.0, 100.1, 100.2, 100.3])
    replay = sqlite_store.replay(seek=0.2)
    # seek=0.2 from first_ts=100.0 → window starts at 100.2 inclusive.
    pairs = list(replay.streams.lidar.iterate_ts())
    assert [v for _, v in pairs] == [2, 3]
    assert [ts for ts, _ in pairs] == pytest.approx([100.2, 100.3], abs=1e-6)


def test_seek_anchor_pins_to_offset(sqlite_store: SqliteStore) -> None:
    _populate(sqlite_store, "lidar", [100.0, 100.1, 100.2, 100.3])
    replay = sqlite_store.replay(seek=0.2)
    # Pin the anchor directly via the public path the observable would take.
    first_ts = replay.streams.lidar.first_ts()
    assert first_ts is not None
    assert first_ts == pytest.approx(100.2, abs=1e-6)
    _, replay_t0 = replay._resolve_anchor(first_ts)
    assert replay_t0 == pytest.approx(100.2, abs=1e-6)


def test_duration_bounds_window(sqlite_store: SqliteStore) -> None:
    _populate(sqlite_store, "lidar", [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    replay = sqlite_store.replay(duration=0.12)
    # time_range is inclusive on both sides: 0.0, 0.05, 0.10 are in; 0.15 is past.
    assert [v for _, v in replay.streams.lidar.iterate_ts()] == [0, 1, 2]


def test_from_timestamp_absolute(sqlite_store: SqliteStore) -> None:
    _populate(sqlite_store, "lidar", [100.0, 100.1, 100.2, 100.3])
    replay = sqlite_store.replay(from_timestamp=100.2)
    assert [v for _, v in replay.streams.lidar.iterate_ts()] == [2, 3]


def test_replay_stream_iterate_ts(sqlite_store: SqliteStore) -> None:
    _populate(sqlite_store, "lidar", [10.0, 10.5, 11.0])
    replay = sqlite_store.replay()
    pairs = list(replay.streams.lidar.iterate_ts())
    assert pairs == [(10.0, 0), (10.5, 1), (11.0, 2)]


def test_replay_stream_count_respects_seek(sqlite_store: SqliteStore) -> None:
    _populate(sqlite_store, "lidar", [0.0, 1.0, 2.0, 3.0])
    replay = sqlite_store.replay(seek=1.0)
    assert replay.streams.lidar.count() == 3


def test_anchor_is_shared_across_streams(sqlite_store: SqliteStore) -> None:
    _populate(sqlite_store, "lidar", [0.0, 0.1])
    _populate(sqlite_store, "odom", [0.0, 0.1])
    replay = sqlite_store.replay()
    a1 = replay._resolve_anchor(0.0)
    a2 = replay._resolve_anchor(0.0)
    assert a1 is a2 or a1 == a2


def test_anchor_reset_forgets_pin(sqlite_store: SqliteStore) -> None:
    _populate(sqlite_store, "lidar", [0.0, 0.1])
    replay = sqlite_store.replay()
    replay._resolve_anchor(0.0)
    assert replay._anchor is not None
    replay.reset_anchor()
    assert replay._anchor is None


def test_replay_anchor_thread_safe(sqlite_store: SqliteStore) -> None:
    """Concurrent _resolve_anchor calls return the same anchor — no torn state."""
    _populate(sqlite_store, "lidar", [0.0, 0.1, 0.2])
    _populate(sqlite_store, "odom", [0.0, 0.1, 0.2])
    replay = sqlite_store.replay()

    n_workers = 8
    barrier = threading.Barrier(n_workers)
    anchors: list[tuple[float, float]] = []
    anchors_lock = threading.Lock()

    def race() -> None:
        barrier.wait()  # release all workers at once
        a = replay._resolve_anchor(0.0)
        with anchors_lock:
            anchors.append(a)

    threads = [threading.Thread(target=race) for _ in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=1.0)

    assert len(anchors) == n_workers
    assert all(a == anchors[0] for a in anchors)


def test_observable_dispose_waits_for_inflight_callback(memory_store: MemoryStore) -> None:
    _populate(memory_store, "lidar", [0.0, 0.1])
    replay = memory_store.replay()
    callback_started = threading.Event()
    release_callback = threading.Event()
    second_callback = threading.Event()
    completed = threading.Event()
    dispose_entered = threading.Event()
    dispose_returned = threading.Event()
    errors: Queue[BaseException] = Queue()
    callback_count = 0

    def on_next(_value: int) -> None:
        nonlocal callback_count
        callback_count += 1
        if callback_count > 1:
            second_callback.set()
        callback_started.set()
        if not release_callback.wait(timeout=2.0):
            errors.put(TimeoutError("callback release timed out"))

    subscription = replay.streams.lidar.observable().subscribe(
        on_next,
        errors.put,
        completed.set,
    )

    def dispose() -> None:
        dispose_entered.set()
        try:
            subscription.dispose()
        except BaseException as error:
            errors.put(error)
        finally:
            dispose_returned.set()

    dispose_thread = threading.Thread(target=dispose, daemon=True)
    try:
        assert callback_started.wait(timeout=2.0)
        dispose_thread.start()
        assert dispose_entered.wait(timeout=2.0)
        assert not dispose_returned.wait(timeout=0.1)
    finally:
        release_callback.set()
        subscription.dispose()
        if dispose_thread.ident is not None:
            dispose_thread.join(timeout=2.0)

    assert not dispose_thread.is_alive()
    assert dispose_returned.is_set()
    assert not second_callback.wait(timeout=0.2)
    assert not completed.is_set()
    assert callback_count == 1
    assert errors.empty()


def test_observable_can_dispose_from_its_callback(memory_store: MemoryStore) -> None:
    _populate(memory_store, "lidar", [0.0, 0.1, 0.2])
    replay = memory_store.replay()
    subscription_ready = threading.Event()
    second_seen = threading.Event()
    third_seen = threading.Event()
    completed = threading.Event()
    errors: Queue[Exception] = Queue()
    subscription: DisposableBase | None = None

    def on_next(value: int) -> None:
        if value == 0:
            if not subscription_ready.wait(timeout=2.0):
                errors.put(TimeoutError("subscription assignment timed out"))
        elif value == 1:
            assert subscription is not None
            subscription.dispose()
            second_seen.set()
        else:
            third_seen.set()

    try:
        subscription = replay.streams.lidar.observable().subscribe(
            on_next,
            errors.put,
            completed.set,
        )
        subscription_ready.set()

        assert second_seen.wait(timeout=2.0)
        assert not third_seen.wait(timeout=0.2)
        assert not completed.is_set()
        assert errors.empty()
    finally:
        subscription_ready.set()
        if subscription is not None:
            subscription.dispose()
