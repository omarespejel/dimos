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

"""PGO drift corrections as composable Stream stages.

Pipeline:

    lidar: Stream[PointCloud2]
        -> pgo_keyframes(...)            -> Stream[Keyframe]
        -> keyframes_to_corrections(...) -> Stream[Transform]   (world_corrected <- world_raw)
        -> apply_corrections(any_stream, corrections) -> Stream[T]   (obs.pose shuffled)

The math: per keyframe, the drift correction is
    R_corr = R_global @ R_local.T
    t_corr = t_global - R_corr @ t_local
and at arbitrary ts we SLERP R between the two bracketing keyframes and linear-lerp t,
clipping out-of-range to endpoints.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from dimos.memory2.store.memory import MemoryStore
from dimos.memory2.stream import Stream
from dimos.memory2.type.observation import Observation
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2

T = TypeVar("T")


@dataclass(frozen=True)
class Keyframe:
    ts: float
    r_local: np.ndarray  # 3x3
    t_local: np.ndarray  # (3,)
    r_global: np.ndarray  # 3x3
    t_global: np.ndarray  # (3,)


def pgo_keyframes(
    stream: Stream[PointCloud2],
    *,
    on_frame: Callable[[Any], None] | None = None,
    **pgo_cfg: Any,
) -> Stream[Keyframe]:
    """Run PGO across a pose-stamped point-cloud stream; return one obs per keyframe."""
    # Imported here to keep pgo_internals.py the only place that imports gtsam at module scope.
    from dimos.mapping.relocalization.pgo_internals import PGOConfig, _SimplePGO

    cfg = PGOConfig(**pgo_cfg)
    pgo = _SimplePGO(cfg)

    for obs in stream:
        if on_frame is not None:
            on_frame(obs)
        if obs.pose is None:
            continue
        x, y, z, qx, qy, qz, qw = obs.pose
        if x == 0 and y == 0 and z == 0:
            continue
        if qx == 0 and qy == 0 and qz == 0 and qw == 0:
            continue
        r = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        t = np.array([x, y, z])
        points, _ = obs.data.as_numpy()
        if len(points) == 0:
            continue
        body_pts = (
            (r.T @ (points[:, :3].T - t[:, None])).T if cfg.unregister_input else points[:, :3]
        )
        if pgo.add_key_pose(r, t, obs.ts, body_pts):
            pgo.search_for_loops()
            pgo.smooth_and_update()

    kps = sorted(pgo._key_poses, key=lambda kp: kp.timestamp)
    kps = [kp for i, kp in enumerate(kps) if i == 0 or kp.timestamp > kps[i - 1].timestamp]

    mem = MemoryStore()
    out: Stream[Keyframe] = mem.stream("keyframes", Keyframe)
    for kp in kps:
        out.append(
            Keyframe(
                ts=kp.timestamp,
                r_local=np.ascontiguousarray(kp.r_local),
                t_local=np.ascontiguousarray(kp.t_local),
                r_global=np.ascontiguousarray(kp.r_global),
                t_global=np.ascontiguousarray(kp.t_global),
            ),
            ts=kp.timestamp,
        )
    return out


def keyframes_to_corrections(keyframes: Stream[Keyframe]) -> Stream[Transform]:
    """Per-keyframe drift correction as Transform(world_corrected <- world_raw)."""
    mem = MemoryStore()
    out: Stream[Transform] = mem.stream("corrections", Transform)
    for obs in keyframes:
        kp = obs.data
        R_corr = kp.r_global @ kp.r_local.T
        t_corr = kp.t_global - R_corr @ kp.t_local
        tf = Transform(
            translation=Vector3(float(t_corr[0]), float(t_corr[1]), float(t_corr[2])),
            rotation=Quaternion.from_rotation_matrix(R_corr),
            frame_id="world_corrected",
            child_frame_id="world_raw",
            ts=kp.ts,
        )
        out.append(tf, ts=kp.ts)
    return out


def make_interpolator(corrections: Stream[Transform]) -> Callable[[float], Transform]:
    """Materialize corrections once; return a fast ts -> Transform lookup."""
    ts_list: list[float] = []
    R_list: list[np.ndarray] = []
    t_list: list[np.ndarray] = []
    for obs in corrections:
        tf = obs.data
        ts_list.append(tf.ts)
        q = tf.rotation
        R_list.append(Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix())
        t_list.append(np.array([tf.translation.x, tf.translation.y, tf.translation.z]))

    if not ts_list:
        raise ValueError("empty corrections stream")

    if len(ts_list) == 1:
        only_ts = ts_list[0]
        only_R = R_list[0]
        only_t = t_list[0]

        def _const(ts: float) -> Transform:
            return Transform(
                translation=Vector3(float(only_t[0]), float(only_t[1]), float(only_t[2])),
                rotation=Quaternion.from_rotation_matrix(only_R),
                frame_id="world_corrected",
                child_frame_id="world_raw",
                ts=ts if ts is not None else only_ts,
            )

        return _const

    ts_arr = np.array(ts_list)
    R_stack = np.stack(R_list)
    t_stack = np.stack(t_list)
    slerp = Slerp(ts_arr, Rotation.from_matrix(R_stack))

    def interp(ts: float) -> Transform:
        ts_clip = float(np.clip(ts, ts_arr[0], ts_arr[-1]))
        R = slerp([ts_clip])[0].as_matrix()
        idx = int(np.searchsorted(ts_arr, ts_clip))
        if idx == 0:
            t = t_stack[0]
        elif idx >= len(ts_arr):
            t = t_stack[-1]
        else:
            t_lo, t_hi = ts_arr[idx - 1], ts_arr[idx]
            alpha = (ts_clip - t_lo) / (t_hi - t_lo) if t_hi > t_lo else 0.0
            t = (1 - alpha) * t_stack[idx - 1] + alpha * t_stack[idx]
        return Transform(
            translation=Vector3(float(t[0]), float(t[1]), float(t[2])),
            rotation=Quaternion.from_rotation_matrix(R),
            frame_id="world_corrected",
            child_frame_id="world_raw",
            ts=float(ts),
        )

    return interp


def correction_at(corrections: Stream[Transform], ts: float) -> Transform:
    """One-off lookup. For hot paths build `make_interpolator` once and reuse."""
    return make_interpolator(corrections)(ts)


def apply_corrections(
    stream: Stream[T],
    corrections: Stream[Transform],
) -> Stream[T]:
    """Shuffle obs.pose on `stream` by the interpolated correction at each obs.ts.

    `obs.data` is untouched. Frames with `obs.pose is None` pass through
    unchanged. Out-of-range `obs.ts` get the endpoint correction (clipped).
    """
    interp = make_interpolator(corrections)

    def xf(upstream: Iterator[Observation[T]]) -> Iterator[Observation[T]]:
        for obs in upstream:
            if obs.pose is None:
                yield obs
                continue
            x, y, z, qx, qy, qz, qw = obs.pose
            tf = interp(obs.ts)
            R_corr = Rotation.from_quat(
                [tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w]
            ).as_matrix()
            t_corr = np.array([tf.translation.x, tf.translation.y, tf.translation.z])
            R_raw = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t_raw = np.array([x, y, z])
            R_new = R_corr @ R_raw
            t_new = R_corr @ t_raw + t_corr
            q_new = Rotation.from_matrix(R_new).as_quat()  # xyzw
            new_pose = (
                float(t_new[0]),
                float(t_new[1]),
                float(t_new[2]),
                float(q_new[0]),
                float(q_new[1]),
                float(q_new[2]),
                float(q_new[3]),
            )
            yield obs.derive(data=obs.data, pose=new_pose)

    return stream.transform(xf)
