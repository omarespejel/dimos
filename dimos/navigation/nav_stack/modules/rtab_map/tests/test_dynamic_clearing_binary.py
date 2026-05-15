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

"""Verify the persistent global OctoMap actually clears stale obstacles.

User-observed regression: a chair (or person who walked past) shows up in
the OctoMap and never disappears, even when subsequent scans clearly
show empty space at that location. This test reproduces the scenario
synthetically: feed N scans with a "chair" obstacle present, then M
scans where the chair is gone but a wall is visible further back. The
rays from sensor to that wall must pass through where the chair was, so
the chair voxels should clear via log-odds within a few scans.

If this test passes, the clearing math works in principle and any
real-robot failure is a config/geometry issue (drift, RangeMax, etc.).
If it fails, there's a bug in the C++ binary itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from dimos.navigation.nav_stack.modules.rtab_map.tests.conftest import (
    RtabHarness,
    identity_quat,
)

pytestmark = [pytest.mark.self_hosted]

_CHAIR_X = 1.5
_WALL_X = 3.5


def _scan_with_chair() -> np.ndarray:
    """Body-frame scan with floor + side walls + a chair block at x=1.5.

    Synthetic scans have pose.z=0, so cloud z is in body frame ==
    world frame. To get the chair classified as an obstacle (not
    ground), its points must sit above ``Grid/MaxGroundHeight=0.05``.
    Real-robot pose.z is ~1.23 m so this looks different on hardware,
    but the clearing math is identical.
    """
    floor_x = np.linspace(0.2, _WALL_X + 0.5, 24)
    floor_y = np.linspace(-1.4, 1.4, 12)
    xx, yy = np.meshgrid(floor_x, floor_y)
    floor = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)

    wall_x = np.linspace(0.0, _WALL_X + 0.5, 16)
    wall_z = np.linspace(0.0, 1.2, 8)
    xx_w, zz_w = np.meshgrid(wall_x, wall_z)
    left = np.stack([xx_w.ravel(), np.full(xx_w.size, 1.5), zz_w.ravel()], axis=1)
    right = np.stack([xx_w.ravel(), np.full(xx_w.size, -1.5), zz_w.ravel()], axis=1)

    # Chair block at x=_CHAIR_X, y near 0, z above MaxGroundHeight.
    chair_y = np.linspace(-0.3, 0.3, 5)
    chair_z = np.linspace(0.2, 0.8, 6)
    yy_c, zz_c = np.meshgrid(chair_y, chair_z)
    chair = np.stack([np.full(yy_c.size, _CHAIR_X), yy_c.ravel(), zz_c.ravel()], axis=1)

    cloud = np.concatenate([floor, left, right, chair]).astype(np.float32)
    return np.column_stack([cloud, np.ones(len(cloud), dtype=np.float32)])


def _scan_without_chair() -> np.ndarray:
    """Same as above minus the chair. Back wall is now fully visible at
    x=_WALL_X — rays from sensor to wall pass through where the chair
    used to sit, so OctoMap should clear those cells."""
    floor_x = np.linspace(0.2, _WALL_X + 0.5, 24)
    floor_y = np.linspace(-1.4, 1.4, 12)
    xx, yy = np.meshgrid(floor_x, floor_y)
    floor = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)

    wall_x = np.linspace(0.0, _WALL_X + 0.5, 16)
    wall_z = np.linspace(0.0, 1.2, 8)
    xx_w, zz_w = np.meshgrid(wall_x, wall_z)
    left = np.stack([xx_w.ravel(), np.full(xx_w.size, 1.5), zz_w.ravel()], axis=1)
    right = np.stack([xx_w.ravel(), np.full(xx_w.size, -1.5), zz_w.ravel()], axis=1)

    # Back wall vertical span must cover the chair's full height range
    # — rays from a low sensor (z=0) only sweep an angular cone, and
    # chair cells outside that cone never see a clearing ray.
    back_y = np.linspace(-0.4, 0.4, 9)
    back_z = np.linspace(0.05, 1.5, 14)
    yy_b, zz_b = np.meshgrid(back_y, back_z)
    back = np.stack([np.full(yy_b.size, _WALL_X), yy_b.ravel(), zz_b.ravel()], axis=1)

    cloud = np.concatenate([floor, left, right, back]).astype(np.float32)
    return np.column_stack([cloud, np.ones(len(cloud), dtype=np.float32)])


def _count_chair_voxels(pts: np.ndarray) -> int:
    return int(
        np.sum(
            (np.abs(pts[:, 0] - _CHAIR_X) < 0.3)
            & (np.abs(pts[:, 1]) < 0.4)
            & (pts[:, 2] > 0.15)
            & (pts[:, 2] < 0.85)
        )
    )


def test_chair_clears_after_it_moves_away(rtab_harness: RtabHarness) -> None:
    """Phase 1: 15 scans with chair present at x=1.5. Phase 2: 15 scans
    with chair removed (wall now visible at x=3.5). The chair voxels
    should be in phase-1 octomap and gone by end of phase 2."""
    chair = _scan_with_chair()
    no_chair = _scan_without_chair()

    # Phase 1: chair present.
    for i in range(15):
        # Start from a non-zero base — some encoders treat ts=0 as
        # "fill in current wall-clock time," which then trips the
        # binary's monotonic-time gate and stops further publishes.
        ts = 100.0 + float(i) * 0.3
        # Nudge x slightly so rtabmap admits each frame as a keyframe.
        rtab_harness.publish_odom(np.array([-0.05 * i, 0.0, 0.0]), identity_quat(), ts)
        rtab_harness.publish_scan(chair, ts)
        rtab_harness.drain(seconds=0.15)
    rtab_harness.drain(seconds=2.0)

    phase1_msgs = [msg for msg in rtab_harness.octomap.messages if len(msg.as_numpy()[0]) > 0]
    assert phase1_msgs, "no non-empty octomap in phase 1"
    phase1_pts, _ = phase1_msgs[-1].as_numpy()
    chair_voxels_present = _count_chair_voxels(phase1_pts)
    assert chair_voxels_present > 0, (
        f"chair should produce obstacle voxels at x~{_CHAIR_X}; "
        f"got {chair_voxels_present}. pts shape={phase1_pts.shape}"
    )

    # Phase 2: chair gone. Keep moving so keyframes admit and new scans
    # integrate into the OctoMap.
    phase1_count = len(rtab_harness.octomap.messages)
    for i in range(15, 30):
        # Start from a non-zero base — some encoders treat ts=0 as
        # "fill in current wall-clock time," which then trips the
        # binary's monotonic-time gate and stops further publishes.
        ts = 100.0 + float(i) * 0.3
        rtab_harness.publish_odom(np.array([-0.05 * i, 0.0, 0.0]), identity_quat(), ts)
        rtab_harness.publish_scan(no_chair, ts)
        rtab_harness.drain(seconds=0.15)
    rtab_harness.drain(seconds=3.0)

    phase2_msgs = [
        msg for msg in rtab_harness.octomap.messages[phase1_count:] if len(msg.as_numpy()[0]) > 0
    ]
    assert phase2_msgs, "no non-empty octomap in phase 2"
    phase2_pts, _ = phase2_msgs[-1].as_numpy()
    chair_voxels_after = _count_chair_voxels(phase2_pts)

    # Allow some residual voxels at chair-volume cells that no back-wall
    # ray happens to pass through (synthetic-geometry artifact). The
    # primary signal is the ratio: clearing should remove most of the
    # chair, leaving a fraction at the boundary of the ray cone.
    assert chair_voxels_after < chair_voxels_present // 2, (
        f"chair should have cleared significantly after 15 scans of "
        f"empty rays, but {chair_voxels_after} voxels remain (was "
        f"{chair_voxels_present} in phase 1) — log-odds clearing not "
        f"taking effect"
    )
