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

"""Synthetic-trajectory tests for the scoring library."""

from __future__ import annotations

import math

import pytest

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Path import Path
from dimos.utils.benchmarking.scoring import (
    ExecutedTrajectory,
    TrajectoryTick,
    score_run,
    score_run_with_trajectory,
)
from dimos.utils.characterization.trajectories import circle, straight


def _pose(x: float, y: float, yaw: float = 0.0) -> PoseStamped:
    return PoseStamped(
        position=Vector3(x, y, 0.0),
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, yaw)),
    )


def _straight_path(length: float = 5.0, step: float = 0.1) -> Path:
    n = int(length / step)
    poses = [_pose(i * step, 0.0, 0.0) for i in range(n + 1)]
    return Path(poses=poses)


def _zero_twist() -> Twist:
    return Twist()


def _const_twist(vx: float = 0.5, wz: float = 0.0) -> Twist:
    return Twist(linear=Vector3(vx, 0.0, 0.0), angular=Vector3(0.0, 0.0, wz))


def test_perfect_tracking_zero_error() -> None:
    """Executed pose exactly on the path → CTE and heading error are zero."""
    path = _straight_path(length=5.0, step=0.1)
    ticks = [
        TrajectoryTick(
            t=i * 0.1,
            pose=_pose(i * 0.1, 0.0, 0.0),
            cmd_twist=_const_twist(0.5),
            actual_twist=_const_twist(0.5),
        )
        for i in range(50)
    ]
    result = score_run(path, ExecutedTrajectory(ticks=ticks, arrived=True))

    assert result.cte_rms == pytest.approx(0.0, abs=1e-9)
    assert result.cte_max == pytest.approx(0.0, abs=1e-9)
    assert result.heading_err_rms == pytest.approx(0.0, abs=1e-9)
    assert result.heading_err_max == pytest.approx(0.0, abs=1e-9)
    assert result.arrived is True
    assert result.n_ticks == 50


def test_constant_lateral_offset() -> None:
    """Executed pose offset 0.1 m perpendicular to path → CTE = 0.1."""
    path = _straight_path(length=5.0, step=0.1)
    offset = 0.1
    ticks = [
        TrajectoryTick(
            t=i * 0.1,
            pose=_pose(i * 0.1, offset, 0.0),
            cmd_twist=_const_twist(0.5),
            actual_twist=_const_twist(0.5),
        )
        for i in range(50)
    ]
    result = score_run(path, ExecutedTrajectory(ticks=ticks, arrived=True))

    assert result.cte_rms == pytest.approx(offset, abs=1e-6)
    assert result.cte_max == pytest.approx(offset, abs=1e-6)
    assert result.heading_err_rms == pytest.approx(0.0, abs=1e-9)


def test_constant_heading_offset() -> None:
    """Executed pose on path but yawed 0.2 rad off → heading_err = 0.2."""
    path = _straight_path(length=5.0, step=0.1)
    yaw_off = 0.2
    ticks = [
        TrajectoryTick(
            t=i * 0.1,
            pose=_pose(i * 0.1, 0.0, yaw_off),
            cmd_twist=_const_twist(0.5),
            actual_twist=_const_twist(0.5),
        )
        for i in range(50)
    ]
    result = score_run(path, ExecutedTrajectory(ticks=ticks, arrived=True))

    assert result.heading_err_rms == pytest.approx(yaw_off, abs=1e-6)
    assert result.heading_err_max == pytest.approx(yaw_off, abs=1e-6)
    assert result.cte_rms == pytest.approx(0.0, abs=1e-9)


def test_command_metrics() -> None:
    """RMS speeds and time-to-complete reflect commanded values."""
    path = _straight_path(length=5.0, step=0.1)
    vx, wz = 0.4, 0.3
    n = 50
    dt = 0.1
    ticks = [
        TrajectoryTick(
            t=i * dt,
            pose=_pose(i * 0.1, 0.0, 0.0),
            cmd_twist=_const_twist(vx, wz),
            actual_twist=_const_twist(vx, wz),
        )
        for i in range(n)
    ]
    result = score_run(path, ExecutedTrajectory(ticks=ticks, arrived=True))

    assert result.linear_speed_rms == pytest.approx(vx, abs=1e-9)
    assert result.angular_speed_rms == pytest.approx(wz, abs=1e-9)
    assert result.time_to_complete == pytest.approx((n - 1) * dt, abs=1e-9)
    # constant cmd → cmd_rate_integral = 0
    assert result.cmd_rate_integral == pytest.approx(0.0, abs=1e-9)


def test_cmd_rate_integral_picks_up_jumps() -> None:
    path = _straight_path(length=5.0, step=0.1)
    # Alternating linear-x command at 0.0 and 0.5 each tick → jump magnitude
    # is 0.5 between every adjacent pair; for 5 ticks we get 4 jumps x 0.5 = 2.0.
    vxs = [0.0, 0.5, 0.0, 0.5, 0.0]
    ticks = [
        TrajectoryTick(
            t=i * 0.1,
            pose=_pose(i * 0.1, 0.0, 0.0),
            cmd_twist=_const_twist(vx, 0.0),
            actual_twist=_const_twist(vx, 0.0),
        )
        for i, vx in enumerate(vxs)
    ]
    result = score_run(path, ExecutedTrajectory(ticks=ticks, arrived=True))
    assert result.cmd_rate_integral == pytest.approx(2.0, abs=1e-9)


def test_empty_trajectory_returns_zeros() -> None:
    path = _straight_path()
    result = score_run(path, ExecutedTrajectory(ticks=[], arrived=False))
    assert result.n_ticks == 0
    assert result.cte_rms == 0.0
    assert result.arrived is False


def test_corner_path_segment_choice() -> None:
    """L-shaped path: a pose right at the corner is on both legs; pick whichever."""
    poses = [_pose(0.0, 0.0), _pose(1.0, 0.0), _pose(1.0, 1.0)]
    path = Path(poses=poses)
    ticks = [
        TrajectoryTick(
            t=0.0,
            pose=_pose(1.0, 0.0, 0.0),  # exactly on corner
            cmd_twist=_zero_twist(),
            actual_twist=_zero_twist(),
        ),
    ]
    result = score_run(path, ExecutedTrajectory(ticks=ticks, arrived=False))
    assert result.cte_rms == pytest.approx(0.0, abs=1e-9)


def test_off_axis_perpendicular_to_corner() -> None:
    """Pose 0.3 m above the L-corner: nearest distance is to corner point."""
    poses = [_pose(0.0, 0.0), _pose(1.0, 0.0), _pose(1.0, 1.0)]
    path = Path(poses=poses)
    ticks = [
        TrajectoryTick(
            t=0.0,
            pose=_pose(1.3, 0.0, 0.0),  # past the corner along leg-1's extension
            cmd_twist=_zero_twist(),
            actual_twist=_zero_twist(),
        ),
    ]
    result = score_run(path, ExecutedTrajectory(ticks=ticks, arrived=False))
    # nearest point on either segment is the corner (1.0, 0.0); distance = 0.3
    assert result.cte_rms == pytest.approx(0.3, abs=1e-6)


# ---------------------------------------------------------------------------
# Trajectory-tracking scoring (time-indexed)
# ---------------------------------------------------------------------------


def test_constant_along_track_lag() -> None:
    """Ref straight at 0.5 m/s; executed shifted 0.1 m back at every tick.

    Pins sign convention: ``along_track_lag > 0`` means robot is BEHIND
    the reference. Cross-track should be ~0, heading error ~0.
    """
    traj = straight(v=0.5, duration=4.0)
    dt = 0.05
    n = 80
    ticks = [
        TrajectoryTick(
            t=i * dt,
            pose=_pose(traj.ref_fn(i * dt).x - 0.1, 0.0, 0.0),  # 0.1 m behind in ref-frame x
            cmd_twist=_const_twist(0.5),
            actual_twist=_const_twist(0.5),
        )
        for i in range(n)
    ]
    result = score_run_with_trajectory(
        ExecutedTrajectory(ticks=ticks, arrived=False),
        traj.ref_fn,
        duration_s=traj.duration_s,
    )

    assert result.along_track_lag_rms == pytest.approx(0.1, abs=1e-9)
    assert result.along_track_lag_max == pytest.approx(0.1, abs=1e-9)
    assert result.cross_track_traj_rms == pytest.approx(0.0, abs=1e-9)
    assert result.heading_err_traj_rms == pytest.approx(0.0, abs=1e-9)
    assert result.traj_completed_on_time_pct == pytest.approx(
        (n - 1) * dt / traj.duration_s, abs=1e-9
    )


def test_pure_cross_track_drift() -> None:
    """Ref straight along +x; executed has same x but drifts +y at 0.1 m/s.

    Pins frame rotation: cross-track is computed in ref-yaw frame.
    Since ref.yaw=0, ref-frame y == world y. Cross-track RMS over
    [0, T] of a linear drift is ``v_drift * T / sqrt(3)``.
    """
    traj = straight(v=0.5, duration=4.0)
    dt = 0.05
    n = 80
    v_drift = 0.1
    ticks = [
        TrajectoryTick(
            t=i * dt,
            pose=_pose(traj.ref_fn(i * dt).x, v_drift * (i * dt), 0.0),
            cmd_twist=_const_twist(0.5),
            actual_twist=_const_twist(0.5),
        )
        for i in range(n)
    ]
    result = score_run_with_trajectory(ExecutedTrajectory(ticks=ticks, arrived=False), traj.ref_fn)

    # Discrete RMS for a linear ramp 0..v_drift*T over n samples
    T = (n - 1) * dt
    expected_rms = v_drift * math.sqrt(sum((i * dt) ** 2 for i in range(n)) / n)
    assert result.cross_track_traj_rms == pytest.approx(expected_rms, abs=1e-9)
    # Max cross-track is the final tick's drift
    assert result.cross_track_traj_max == pytest.approx(v_drift * T, abs=1e-9)
    assert result.along_track_lag_rms == pytest.approx(0.0, abs=1e-9)
    assert result.heading_err_traj_rms == pytest.approx(0.0, abs=1e-9)


def test_saturated_circle() -> None:
    """Ref circle at w=1.6 rad/s; executed clipped to w=1.5 rad/s.

    Pins the saturation signature the classifier hunts for: heading
    error grows monotonically with time; cross-track grows too as the
    two circle radii diverge.
    """
    v = 0.5
    w_ref = 1.6
    w_actual = 1.5
    ref_traj = circle(v=v, w=w_ref, duration=1.0)

    dt = 0.02
    n = 50
    ticks = []
    for i in range(n):
        t = i * dt
        yaw = w_actual * t
        inv_w = 1.0 / w_actual
        x = v * inv_w * math.sin(yaw)
        y = v * inv_w * (1.0 - math.cos(yaw))
        ticks.append(
            TrajectoryTick(
                t=t,
                pose=_pose(x, y, yaw),
                cmd_twist=_const_twist(v, w_actual),
                actual_twist=_const_twist(v, w_actual),
            )
        )

    result = score_run_with_trajectory(
        ExecutedTrajectory(ticks=ticks, arrived=False),
        ref_traj.ref_fn,
        duration_s=ref_traj.duration_s,
    )

    # Heading error at the final tick should be (1.6 - 1.5) * t_final ≈ 0.1*0.98 = 0.098 rad.
    # RMS over a linear ramp 0..0.098 over the run is 0.098/sqrt(3) ≈ 0.057.
    expected_he_max = abs(w_ref - w_actual) * (n - 1) * dt
    assert result.heading_err_traj_max == pytest.approx(expected_he_max, abs=1e-6)
    # heading_err and cross_track should both be growing — pin lower bounds:
    assert result.heading_err_traj_rms > 0.02
    assert result.cross_track_traj_rms > 0.0
    # Cross-track should be POSITIVE (the actual circle is to the LEFT of the ref,
    # because actual has larger radius and they share initial heading); enough
    # that we know the sign convention is exercised.
    assert result.cross_track_traj_max > 0.0
