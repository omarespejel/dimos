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

"""Endpoint tests for the time-indexed reference trajectory primitives.

Each test pins the closed-form (or numerically converged) endpoint of one
helper. If integration drifts these will catch it.
"""

from __future__ import annotations

import math

import pytest

from dimos.utils.characterization.trajectories import (
    circle,
    sinusoidal_wz,
    step_vx,
    step_wz,
    straight,
    trajectory_from_spec,
    trapezoidal_vx,
)


def test_straight_endpoint():
    traj = straight(v=0.5, duration=4.0)
    assert traj.recommended_mode == "openloop_ff"

    r0 = traj.ref_fn(0.0)
    assert r0.x == pytest.approx(0.0)
    assert r0.vx == pytest.approx(0.5)
    assert r0.wz == pytest.approx(0.0)

    r_end = traj.ref_fn(4.0)
    assert r_end.x == pytest.approx(2.0)
    assert r_end.y == pytest.approx(0.0)
    assert r_end.yaw == pytest.approx(0.0)


def test_circle_quarter_turn_endpoint():
    """At t=pi/(2w) on a circle the robot has done a quarter turn:
    x = v/w, y = v/w, yaw = pi/2."""
    v, w = 0.5, 0.5  # R = 1.0
    traj = circle(v=v, w=w, duration=4 * math.pi)
    assert traj.recommended_mode == "lowgain_p"

    r = traj.ref_fn(math.pi / (2 * w))
    assert r.x == pytest.approx(v / w, abs=1e-9)
    assert r.y == pytest.approx(v / w, abs=1e-9)
    assert r.yaw == pytest.approx(math.pi / 2, abs=1e-9)
    assert r.vx == pytest.approx(v)
    assert r.wz == pytest.approx(w)


def test_circle_full_revolution_returns_to_origin():
    v, w = 0.4, 1.0
    traj = circle(v=v, w=w, duration=2 * math.pi / w + 1.0)
    r = traj.ref_fn(2 * math.pi / w)
    assert r.x == pytest.approx(0.0, abs=1e-9)
    assert r.y == pytest.approx(0.0, abs=1e-9)
    assert r.yaw == pytest.approx(2 * math.pi, abs=1e-9)


def test_circle_zero_w_falls_back_to_straight():
    """A circle with w=0 has undefined radius; helper degrades to straight."""
    traj = circle(v=0.5, w=0.0, duration=2.0)
    r = traj.ref_fn(2.0)
    assert r.x == pytest.approx(1.0)
    assert r.y == pytest.approx(0.0)
    assert r.wz == pytest.approx(0.0)


def test_step_vx_pre_step_is_idle():
    traj = step_vx(v_target=0.6, duration=4.0, t_step=1.0)
    assert traj.recommended_mode == "openloop_ff"

    r = traj.ref_fn(0.5)
    assert r.x == pytest.approx(0.0)
    assert r.vx == pytest.approx(0.0)


def test_step_vx_post_step_distance():
    traj = step_vx(v_target=0.6, duration=4.0, t_step=1.0)
    r = traj.ref_fn(3.0)
    assert r.x == pytest.approx(0.6 * 2.0, abs=1e-9)
    assert r.vx == pytest.approx(0.6)


def test_step_wz_pre_step_is_straight():
    traj = step_wz(vx=0.4, w_target=0.8, duration=4.0, t_step=1.0)
    assert traj.recommended_mode == "openloop_ff"

    r = traj.ref_fn(0.5)
    assert r.x == pytest.approx(0.4 * 0.5)
    assert r.y == pytest.approx(0.0)
    assert r.yaw == pytest.approx(0.0)


def test_step_wz_half_turn_after_step():
    """After step_wz: at t = t_step + pi/w, robot has done a half turn."""
    vx, w, t_step = 0.4, 0.8, 1.0
    traj = step_wz(vx=vx, w_target=w, duration=10.0, t_step=t_step)

    r = traj.ref_fn(t_step + math.pi / w)
    # x0 = vx*t_step; the half-turn brings x back to x0 (sin(pi)=0)
    assert r.x == pytest.approx(vx * t_step, abs=1e-9)
    # y at half turn = (vx/w)*(1-cos(pi)) = 2*vx/w
    assert r.y == pytest.approx(2.0 * vx / w, abs=1e-9)
    assert r.yaw == pytest.approx(math.pi, abs=1e-9)


def test_trapezoidal_vx_full_distance():
    """Symmetric trapezoid: total distance = v_max*duration - v_max^2/accel."""
    v_max, accel, duration = 0.8, 0.5, 4.0
    traj = trapezoidal_vx(v_max=v_max, accel=accel, duration=duration)
    assert traj.recommended_mode == "openloop_ff"

    r = traj.ref_fn(duration)
    expected = v_max * duration - (v_max * v_max) / accel
    assert r.x == pytest.approx(expected, abs=1e-9)
    assert r.vx == pytest.approx(0.0, abs=1e-9)
    assert r.yaw == pytest.approx(0.0)


def test_trapezoidal_vx_triangular_when_too_short():
    """If duration < 2*v_max/accel, peak doesn't reach v_max; profile is triangular."""
    v_max, accel, duration = 1.0, 0.5, 2.0
    traj = trapezoidal_vx(v_max=v_max, accel=accel, duration=duration)
    # Peak occurs at duration/2 = 1.0; v_peak = accel * t_accel = 0.5 * 1 = 0.5
    r = traj.ref_fn(1.0)
    assert r.vx == pytest.approx(0.5, abs=1e-9)
    r_end = traj.ref_fn(2.0)
    # Triangular total area = 0.5 * v_peak * duration = 0.5 * 0.5 * 2.0 = 0.5
    assert r_end.x == pytest.approx(0.5, abs=1e-9)


def test_sinusoidal_wz_returns_to_zero_yaw_per_period():
    """yaw(t) = (w_amp/omega)(1 - cos(omega*t)); zero at t = 0, 2*pi/omega = 1/freq."""
    traj = sinusoidal_wz(vx=0.3, w_amp=0.5, freq_hz=0.5, duration=4.0)
    assert traj.recommended_mode == "lowgain_p"

    r0 = traj.ref_fn(0.0)
    assert r0.x == pytest.approx(0.0)
    assert r0.y == pytest.approx(0.0)
    assert r0.yaw == pytest.approx(0.0)

    # One period at freq=0.5 Hz is 2.0 s — yaw returns to 0
    r_period = traj.ref_fn(2.0)
    assert r_period.yaw == pytest.approx(0.0, abs=1e-3)
    # Position has drifted along +x by less than vx*period (curvature stole some)
    assert r_period.x < 0.3 * 2.0


def test_spec_round_trip_circle():
    original = circle(v=0.5, w=0.5, duration=4.0)
    rebuilt = trajectory_from_spec(original.spec)
    for t in [0.0, 1.0, 2.0, 3.0, 4.0]:
        r_orig = original.ref_fn(t)
        r_re = rebuilt.ref_fn(t)
        assert r_orig.x == pytest.approx(r_re.x)
        assert r_orig.y == pytest.approx(r_re.y)
        assert r_orig.yaw == pytest.approx(r_re.yaw)
        assert r_orig.vx == pytest.approx(r_re.vx)
        assert r_orig.wz == pytest.approx(r_re.wz)


def test_spec_round_trip_all_primitives():
    """Round-trip the full set; sample one mid-trajectory point for each."""
    primitives = [
        straight(v=0.5, duration=4.0),
        circle(v=0.5, w=0.5, duration=4.0),
        step_vx(v_target=0.6, duration=4.0),
        step_wz(vx=0.4, w_target=0.8, duration=4.0),
        trapezoidal_vx(v_max=0.8, accel=0.5, duration=4.0),
        sinusoidal_wz(vx=0.3, w_amp=0.5, freq_hz=0.5, duration=4.0),
    ]
    for traj in primitives:
        rebuilt = trajectory_from_spec(traj.spec)
        t = traj.duration_s * 0.5
        a, b = traj.ref_fn(t), rebuilt.ref_fn(t)
        assert a.x == pytest.approx(b.x), f"{traj.spec['kind']}: x mismatch"
        assert a.y == pytest.approx(b.y), f"{traj.spec['kind']}: y mismatch"
        assert a.yaw == pytest.approx(b.yaw), f"{traj.spec['kind']}: yaw mismatch"


def test_spec_round_trip_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown trajectory kind"):
        trajectory_from_spec({"kind": "bogus", "v": 1.0, "duration_s": 1.0})


def test_t_clipped_below_zero_and_above_duration():
    traj = straight(v=0.5, duration=4.0)
    # Below 0 clips to 0
    r_neg = traj.ref_fn(-1.0)
    assert r_neg.x == pytest.approx(0.0)
    # Above duration clips to duration (avoids extrapolation past intent)
    r_over = traj.ref_fn(10.0)
    assert r_over.x == pytest.approx(2.0)
