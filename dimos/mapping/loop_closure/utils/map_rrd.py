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

"""Dump a recorded dataset to .rrd: lidar point clouds + camera frames.

Lidar clouds are assumed to be in world frame and logged directly under
their entity path (no parent transform). Entities written:

- ``world/lidar``     — primary point cloud (default: Go2 L1, ``--lidar-stream``)
- ``world/livox``     — fastlio_lidar raw cloud (if present)
- ``world/fastlio``   — fastlio_odometry pose axis (if present)
- ``world/camera``    — color_image camera pose (static pinhole + Transform3D)
- ``world/camera/image`` — color_image frames

Usage:
    uv run python -m dimos.mapping.loop_closure.utils.map_rrd mid360 --out map.rrd
    rerun map.rrd
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import time
from typing import Any

import rerun as rr
import typer

from dimos.memory2.store.sqlite import SqliteStore
from dimos.memory2.transform import throttle
from dimos.memory2.type.observation import Observation
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2, register_colormap_annotation
from dimos.robot.unitree.go2.connection import _camera_info_static
from dimos.utils.data import resolve_named_path

TIMELINE = "ts"


def _progress(total: int, label: str) -> Callable[[Observation[Any]], None]:
    """Matches dimos/utils/cli/map.py:progress."""
    seen = 0
    wall_start: float | None = None
    last_wall: float | None = None
    first_ts: float | None = None

    def tick(obs: Observation[Any]) -> None:
        nonlocal seen, wall_start, last_wall, first_ts
        now = time.monotonic()
        if wall_start is None:
            wall_start = now
            first_ts = obs.ts
        assert first_ts is not None
        frame_ms = (now - last_wall) * 1000 if last_wall is not None else 0.0
        last_wall = now
        seen += 1
        pct = 100 * seen // total if total else 100
        wall = now - wall_start
        data = obs.ts - first_ts
        speed = data / wall if wall > 0 else 0.0
        end = "\n" if seen >= total else ""
        print(
            f"\r{label} {pct:>3}% [{seen}/{total}] {data:.1f}s ({speed:.1f} x rt) {frame_ms:.0f}ms/frame",
            end=end,
            flush=True,
        )

    return tick


def main(
    dataset: str = typer.Argument(..., help="Dataset .db: bare name (cwd or data/) or path"),
    out: Path = typer.Option(..., "--out", help="Output .rrd path"),
    lidar_stream: str = typer.Option(
        "lidar", "--lidar-stream", help="Primary point cloud stream (default: Go2 L1)"
    ),
    voxel: float = typer.Option(
        0.05, "--voxel", help="Voxel size hint for the point cloud renderer"
    ),
    point_mode: str = typer.Option(
        "spheres", "--point-mode", help="Render mode: 'spheres', 'boxes', or 'points'"
    ),
    camera_hz: float = typer.Option(
        2.0, "--camera-hz", help="Throttle color_image to at most this rate; 0 disables"
    ),
) -> None:
    db_path = resolve_named_path(dataset, ".db")
    cam_info = _camera_info_static()

    rr.init("dimos map_rrd", recording_id=db_path.stem)
    rr.save(str(out))
    register_colormap_annotation("turbo")

    # Static pinhole on the camera entity; per-frame Transform3D goes on the
    # same entity. Image is the child so it projects through the pinhole.
    pinhole = cam_info.to_rerun()
    assert not isinstance(pinhole, list)
    rr.log("world/camera", pinhole, static=True)

    # Static axis triads as children of each moving Transform3D, so the
    # transforms are actually visible in the 3D view.
    axes = rr.Arrows3D(
        vectors=[[0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    )
    rr.log("world/fastlio/axes", axes, static=True)
    rr.log("world/odom/axes", axes, static=True)

    store = SqliteStore(path=str(db_path))
    with store:
        print(store.summary())

        lidar = store.stream(lidar_stream, PointCloud2)
        color_image = store.stream("color_image", Image)

        n_lidar = lidar.count()
        cb = _progress(n_lidar, f"{lidar_stream:>12s}")
        for lidar_obs in lidar:
            cb(lidar_obs)
            rr.set_time(TIMELINE, timestamp=lidar_obs.ts)
            rr.log(
                "world/lidar",
                lidar_obs.data.to_rerun(voxel_size=voxel, mode=point_mode),
            )

        # ---- livox raw cloud at world/livox (skip if it's the primary) ----
        if "fastlio_lidar" in store.streams and lidar_stream != "fastlio_lidar":
            livox = store.stream("fastlio_lidar", PointCloud2)
            cb = _progress(livox.count(), "fastlio_lidar")
            for obs in livox:
                cb(obs)
                rr.set_time(TIMELINE, timestamp=obs.ts)
                rr.log("world/livox", obs.data.to_rerun(voxel_size=voxel, mode=point_mode))

        # ---- fastlio pose axis from fastlio_odometry stream ----
        if "fastlio_odometry" in store.streams:
            odometry = store.stream("fastlio_odometry", Odometry)
            cb = _progress(odometry.count(), "fastlio_odometry")
            for obs in odometry:
                cb(obs)
                if obs.pose is None:
                    continue
                rr.set_time(TIMELINE, timestamp=obs.ts)
                x, y, z, qx, qy, qz, qw = obs.pose
                rr.log(
                    "world/fastlio",
                    rr.Transform3D(
                        translation=[x, y, z],
                        quaternion=rr.Quaternion(xyzw=[qx, qy, qz, qw]),
                    ),
                )

        # ---- Go2 native odom pose axis ----
        if "odom" in store.streams:
            odom = store.stream("odom", PoseStamped)
            cb = _progress(odom.count(), "        odom")
            for obs in odom:
                cb(obs)
                if obs.pose is None:
                    continue
                rr.set_time(TIMELINE, timestamp=obs.ts)
                x, y, z, qx, qy, qz, qw = obs.pose
                rr.log(
                    "world/odom",
                    rr.Transform3D(
                        translation=[x, y, z],
                        quaternion=rr.Quaternion(xyzw=[qx, qy, qz, qw]),
                    ),
                )

        # ---- pass 2: camera pose + image per color_image ----
        cam_pipeline = (
            color_image.transform(throttle(1.0 / camera_hz)) if camera_hz > 0 else color_image
        )
        n_img = cam_pipeline.count()
        cb = _progress(n_img, "  color_image")
        for img_obs in cam_pipeline:
            cb(img_obs)
            rr.set_time(TIMELINE, timestamp=img_obs.ts)
            if img_obs.pose is not None:
                x, y, z, qx, qy, qz, qw = img_obs.pose
                rr.log(
                    "world/camera",
                    rr.Transform3D(
                        translation=[x, y, z], quaternion=rr.Quaternion(xyzw=[qx, qy, qz, qw])
                    ),
                )
            rr.log("world/camera/image", img_obs.data.to_rerun())

    print(f"wrote {out}")
    print(f"open with: rerun {out}")


if __name__ == "__main__":
    typer.run(main)
