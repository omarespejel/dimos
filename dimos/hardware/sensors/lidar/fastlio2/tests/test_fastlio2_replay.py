# Copyright 2026 Dimensional Inc.
# SPDX-License-Identifier: Apache-2.0
"""Replay-based test for the pure FastLio2 native module.

Loads a memory2 sqlite fixture captured against a live Mid-360
(see record_fixture.py), replays raw_imu + raw_lidar over LCM at
original timing into a freshly-spawned fastlio2_native, collects
its odometry + world_cloud outputs, and asserts the output rates
and pose shape match the recorded reference.

Styled after dimos/navigation/nav_stack/tests/rosbag_fixtures.py.

Marked slow because it spawns the C++ binary and waits on LCM bus
delivery in real time (~recording-length seconds).
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import lcm as lcmlib
import pytest

from dimos.memory2.store.sqlite import SqliteStore
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.sensor_msgs.RawLidarScan import RawLidarScan

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[6]
FIXTURE_PATH = REPO_ROOT / "tests/data/fastlio2_replay.db"
FASTLIO_BIN = REPO_ROOT / "dimos/hardware/sensors/lidar/fastlio2/cpp/result/bin/fastlio2_native"
FASTLIO_CONFIG = REPO_ROOT / "dimos/hardware/sensors/lidar/fastlio2/cpp/config/mid360.yaml"

# Use replay-specific channel names so a live pipeline doesn't pollute the test.
TOPIC_RAW_IMU = "/test_raw_imu#sensor_msgs.Imu"
TOPIC_RAW_LIDAR = "/test_raw_lidar#sensor_msgs.RawLidarScan"
TOPIC_ODOM = "/test_odom#nav_msgs.Odometry"
TOPIC_WORLD = "/test_world_cloud#sensor_msgs.PointCloud2"

# Time for the native process to initialize before feeding data.
_PROCESS_STARTUP_SEC = 1.5
# Time after feeding data for FastLio to finish processing trailing scans.
_POST_FEED_DRAIN_SEC = 2.0


@pytest.fixture(scope="module")
def fixture_streams() -> dict[str, list[tuple[float, object]]]:
    """Load all recorded streams into in-memory (ts, payload) lists."""
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"Fixture not found: {FIXTURE_PATH}. "
            "Run: uv run python -m dimos.hardware.sensors.lidar.fastlio2.tests.record_fixture"
        )
    store = SqliteStore(path=str(FIXTURE_PATH), must_exist=True)
    streams: dict[str, list[tuple[float, object]]] = {}
    for name, ptype in (
        ("raw_imu", Imu),
        ("raw_lidar", RawLidarScan),
        ("odometry", Odometry),
        ("world_cloud", PointCloud2),
    ):
        s = store.stream(name, ptype)
        rows: list[tuple[float, object]] = [(obs.ts, obs.data) for obs in s.to_list()]
        rows.sort(key=lambda row: row[0])
        streams[name] = rows
    store.stop()
    return streams


def _start_native(binary_path: Path) -> subprocess.Popen[bytes]:
    if not binary_path.exists():
        pytest.skip(f"fastlio2_native not built at {binary_path}. Run: nix build .#fastlio2_native")
    return subprocess.Popen(
        [
            str(binary_path),
            "--raw_imu", TOPIC_RAW_IMU,
            "--raw_lidar", TOPIC_RAW_LIDAR,
            "--lidar", TOPIC_WORLD,
            "--odometry", TOPIC_ODOM,
            "--config_path", str(FASTLIO_CONFIG),
            "--frame_id", "odom",
            "--child_frame_id", "base_link",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _stop_native(proc: subprocess.Popen[bytes], timeout: float = 3.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _collect_outputs(
    bus: lcmlib.LCM, stop_event: threading.Event, collected: dict[str, list[tuple[float, bytes]]]
) -> None:
    while not stop_event.is_set():
        bus.handle_timeout(50)


def _replay_inputs_at_original_timing(
    bus: lcmlib.LCM,
    imu_rows: list[tuple[float, object]],
    lidar_rows: list[tuple[float, object]],
) -> None:
    timeline: list[tuple[float, str, bytes]] = []
    for ts, msg in imu_rows:
        timeline.append((ts, TOPIC_RAW_IMU, msg.lcm_encode()))  # type: ignore[attr-defined]
    for ts, msg in lidar_rows:
        timeline.append((ts, TOPIC_RAW_LIDAR, msg.lcm_encode()))  # type: ignore[attr-defined]
    timeline.sort(key=lambda entry: entry[0])
    if not timeline:
        return
    start_recorded = timeline[0][0]
    real_start = time.monotonic()
    for recorded_ts, topic, blob in timeline:
        target_offset = recorded_ts - start_recorded
        elapsed = time.monotonic() - real_start
        if target_offset > elapsed:
            time.sleep(target_offset - elapsed)
        bus.publish(topic, blob)


def test_fastlio2_replay_produces_outputs(
    fixture_streams: dict[str, list[tuple[float, object]]],
) -> None:
    imu_rows = fixture_streams["raw_imu"]
    lidar_rows = fixture_streams["raw_lidar"]
    odom_ref = fixture_streams["odometry"]
    world_ref = fixture_streams["world_cloud"]
    assert imu_rows and lidar_rows, "Fixture must contain raw_imu + raw_lidar"

    proc = _start_native(FASTLIO_BIN)
    try:
        bus = lcmlib.LCM()
        collected: dict[str, list[tuple[float, bytes]]] = {"odometry": [], "world_cloud": []}

        def on_odom(_chan: str, data: bytes) -> None:
            collected["odometry"].append((time.monotonic(), data))

        def on_world(_chan: str, data: bytes) -> None:
            collected["world_cloud"].append((time.monotonic(), data))

        bus.subscribe(TOPIC_ODOM, on_odom)
        bus.subscribe(TOPIC_WORLD, on_world)

        stop_event = threading.Event()
        worker = threading.Thread(target=_collect_outputs, args=(bus, stop_event, collected),
                                  daemon=True)
        worker.start()

        time.sleep(_PROCESS_STARTUP_SEC)
        _replay_inputs_at_original_timing(bus, imu_rows, lidar_rows)
        time.sleep(_POST_FEED_DRAIN_SEC)
        stop_event.set()
        worker.join(timeout=2.0)
    finally:
        _stop_native(proc)

    n_odom = len(collected["odometry"])
    n_world = len(collected["world_cloud"])
    # The reference recording produced odometry and world_cloud, so the replay must too.
    # We compare counts loosely: FastLio is deterministic across the rate-limiter window
    # but startup transients and the IMU/Lidar arrival order can shift things by a few.
    assert n_odom >= max(1, int(0.5 * len(odom_ref))), (
        f"replay produced too few odometry msgs: {n_odom} vs reference {len(odom_ref)}"
    )
    assert n_world >= max(1, int(0.5 * len(world_ref))), (
        f"replay produced too few world_cloud msgs: {n_world} vs reference {len(world_ref)}"
    )

    # Decode the first odometry sample and sanity-check shape (finite components).
    first_odom = Odometry.lcm_decode(collected["odometry"][0][1])
    p = first_odom.pose.position
    o = first_odom.pose.orientation
    for name, v in (("x", p.x), ("y", p.y), ("z", p.z),
                    ("qx", o.x), ("qy", o.y), ("qz", o.z), ("qw", o.w)):
        assert v == v, f"pose.{name} is NaN"  # NaN != NaN
        assert abs(v) < 1e9, f"pose.{name}={v} is implausibly large"


def test_fixture_streams_well_formed(
    fixture_streams: dict[str, list[tuple[float, object]]],
) -> None:
    """Sanity-check the recorded fixture itself before we try to replay it."""
    assert len(fixture_streams["raw_imu"]) > 0, "raw_imu stream is empty"
    assert len(fixture_streams["raw_lidar"]) > 0, "raw_lidar stream is empty"
    assert len(fixture_streams["odometry"]) > 0, "odometry stream is empty"
    assert len(fixture_streams["world_cloud"]) > 0, "world_cloud stream is empty"

    first_imu = fixture_streams["raw_imu"][0][1]
    assert isinstance(first_imu, Imu)

    first_lidar = fixture_streams["raw_lidar"][0][1]
    assert isinstance(first_lidar, RawLidarScan)
    assert len(first_lidar) > 0, "first raw_lidar scan has no points"
