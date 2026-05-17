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

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.nav_msgs.DynamicCloud import DynamicCloud
from dimos.msgs.nav_msgs.Path import Path
from dimos.navigation.nav_stack.modules.apply_closure.apply_closure import (
    apply_closure_to_cloud,
    compute_node_deltas,
    invert_transforms,
    lbs_warp_positions,
    merge_duplicate_voxels,
    path_to_arrays,
    pose_stamped_to_matrix,
)


def _pose(ts: float, x: float, y: float, z: float, yaw: float = 0.0) -> PoseStamped:
    q = Rotation.from_euler("z", yaw).as_quat()
    return PoseStamped(
        ts=ts,
        frame_id="map",
        position=[x, y, z],
        orientation=[q[0], q[1], q[2], q[3]],
    )


def _path(*poses: PoseStamped) -> Path:
    return Path(ts=0.0, frame_id="map", poses=list(poses))


class TestTransformHelpers:
    def test_invert_round_trip(self) -> None:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.2, 1.1]).as_matrix()
        T[:3, 3] = [1.0, -2.0, 3.0]
        T_batch = T[None, :, :]
        inv = invert_transforms(T_batch)
        np.testing.assert_allclose(T_batch @ inv, np.eye(4)[None, :, :], atol=1e-9)

    def test_pose_stamped_to_matrix_identity(self) -> None:
        pose = PoseStamped(
            ts=0.0,
            frame_id="map",
            position=[0.0, 0.0, 0.0],
            orientation=[0.0, 0.0, 0.0, 1.0],
        )
        np.testing.assert_allclose(pose_stamped_to_matrix(pose), np.eye(4), atol=1e-12)

    def test_compute_node_deltas_identity_when_unchanged(self) -> None:
        prev = np.stack([np.eye(4), np.eye(4)], axis=0)
        nxt = prev.copy()
        deltas = compute_node_deltas(prev, nxt)
        np.testing.assert_allclose(deltas, np.stack([np.eye(4), np.eye(4)]), atol=1e-12)


class TestLBSWarp:
    def test_empty_positions_returns_empty(self) -> None:
        out = lbs_warp_positions(
            np.zeros((0, 3)),
            np.zeros(0),
            np.array([0.0, 1.0]),
            np.stack([np.eye(4), np.eye(4)]),
        )
        assert out.shape == (0, 3)

    def test_no_nodes_passes_through(self) -> None:
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = lbs_warp_positions(positions, np.array([1.0, 2.0]), np.zeros(0), np.zeros((0, 4, 4)))
        np.testing.assert_allclose(out, positions)

    def test_single_node_applies_rigidly(self) -> None:
        delta = np.eye(4)
        delta[:3, 3] = [10.0, 0.0, 0.0]
        positions = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        out = lbs_warp_positions(
            positions, np.array([0.0, 100.0]), np.array([0.0]), delta[None, :, :]
        )
        np.testing.assert_allclose(out, positions + np.array([10.0, 0.0, 0.0]))

    def test_before_range_clips_to_first_node(self) -> None:
        deltas = np.stack([np.eye(4), np.eye(4)])
        deltas[0, :3, 3] = [1.0, 0.0, 0.0]
        deltas[1, :3, 3] = [10.0, 0.0, 0.0]
        positions = np.array([[0.0, 0.0, 0.0]])
        # point time well before node[0] should snap to delta[0]
        out = lbs_warp_positions(positions, np.array([-100.0]), np.array([0.0, 1.0]), deltas)
        np.testing.assert_allclose(out, [[1.0, 0.0, 0.0]])

    def test_after_range_clips_to_last_node(self) -> None:
        deltas = np.stack([np.eye(4), np.eye(4)])
        deltas[0, :3, 3] = [1.0, 0.0, 0.0]
        deltas[1, :3, 3] = [10.0, 0.0, 0.0]
        positions = np.array([[0.0, 0.0, 0.0]])
        out = lbs_warp_positions(positions, np.array([1e9]), np.array([0.0, 1.0]), deltas)
        np.testing.assert_allclose(out, [[10.0, 0.0, 0.0]])

    def test_midpoint_translation_lerps(self) -> None:
        deltas = np.stack([np.eye(4), np.eye(4)])
        deltas[0, :3, 3] = [0.0, 0.0, 0.0]
        deltas[1, :3, 3] = [10.0, 0.0, 0.0]
        positions = np.array([[0.0, 0.0, 0.0]])
        out = lbs_warp_positions(positions, np.array([0.5]), np.array([0.0, 1.0]), deltas)
        np.testing.assert_allclose(out, [[5.0, 0.0, 0.0]])

    def test_midpoint_rotation_slerps(self) -> None:
        deltas = np.stack([np.eye(4), np.eye(4)])
        deltas[1, :3, :3] = Rotation.from_euler("z", math.pi / 2).as_matrix()
        # A point at (1, 0, 0) rotated by 45deg should land at (cos45, sin45, 0)
        out = lbs_warp_positions(
            np.array([[1.0, 0.0, 0.0]]),
            np.array([0.5]),
            np.array([0.0, 1.0]),
            deltas,
        )
        np.testing.assert_allclose(
            out, [[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0]], atol=1e-9
        )


class TestMergeDuplicates:
    def test_no_duplicates_passes_through(self) -> None:
        voxels = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32)
        quantity = np.array([2, 3], dtype=np.uint32)
        events = np.array([0, 1, 1], dtype=np.uint32)
        v, q, e = merge_duplicate_voxels(voxels, quantity, events)
        # np.unique sorts lexicographically — order may differ but contents must match
        assert v.shape == (2, 3)
        assert int(q.sum()) == 5
        # Build old → new index map and verify events were remapped correctly
        old_to_new = {}
        for old_i, original in enumerate(voxels):
            matches = np.where((v == original).all(axis=1))[0]
            assert matches.size == 1
            old_to_new[old_i] = int(matches[0])
        expected_events = np.array([old_to_new[int(idx)] for idx in events], dtype=np.uint32)
        np.testing.assert_array_equal(e, expected_events)

    def test_collision_sums_quantity(self) -> None:
        voxels = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.int32)
        quantity = np.array([5, 7, 9], dtype=np.uint32)
        events = np.array([0, 1, 2], dtype=np.uint32)
        v, q, e = merge_duplicate_voxels(voxels, quantity, events)
        assert v.shape == (2, 3)
        # Find the merged (0,0,0) row
        zero_row = np.where((v == [0, 0, 0]).all(axis=1))[0][0]
        one_row = np.where((v == [1, 0, 0]).all(axis=1))[0][0]
        assert int(q[zero_row]) == 12
        assert int(q[one_row]) == 9
        # The two events that referenced (0,0,0) should now reference zero_row
        assert int(e[0]) == zero_row
        assert int(e[1]) == zero_row
        assert int(e[2]) == one_row

    def test_empty_inputs(self) -> None:
        v, q, e = merge_duplicate_voxels(
            np.zeros((0, 3), dtype=np.int32),
            np.zeros(0, dtype=np.uint32),
            np.zeros(0, dtype=np.uint32),
        )
        assert v.shape == (0, 3)
        assert q.shape == (0,)
        assert e.shape == (0,)


class TestApplyClosureToCloud:
    def test_empty_pose_graph_returns_input(self) -> None:
        cloud = DynamicCloud(
            voxels=np.array([[1, 2, 3]], dtype=np.int32),
            quantity=np.array([1], dtype=np.uint32),
            voxel_size=0.5,
        )
        out = apply_closure_to_cloud(cloud, _path(), _path())
        assert out is cloud

    def test_identity_correction_preserves_voxels(self) -> None:
        cloud = DynamicCloud(
            voxels=np.array([[1, 0, 0], [-2, 3, 1]], dtype=np.int32),
            quantity=np.array([4, 5], dtype=np.uint32),
            voxel_size=0.5,
        )
        prev = _path(_pose(0.0, 0.0, 0.0, 0.0), _pose(10.0, 1.0, 0.0, 0.0))
        nxt = _path(_pose(0.0, 0.0, 0.0, 0.0), _pose(10.0, 1.0, 0.0, 0.0))
        out = apply_closure_to_cloud(cloud, prev, nxt)
        # Sort both for comparison since merge_duplicates may reorder
        np.testing.assert_array_equal(np.sort(out.voxels, axis=0), np.sort(cloud.voxels, axis=0))
        assert int(out.quantity.sum()) == int(cloud.quantity.sum())

    def test_rigid_translation_shift(self) -> None:
        """All nodes shifted by the same vector → entire cloud shifts by it."""
        cloud = DynamicCloud(
            voxels=np.array([[2, 0, 0], [4, 0, 0]], dtype=np.int32),
            quantity=np.array([1, 1], dtype=np.uint32),
            voxel_size=0.5,
        )
        prev = _path(_pose(0.0, 0.0, 0.0, 0.0), _pose(1.0, 0.0, 0.0, 0.0))
        # Both next poses shifted by +1m in x
        nxt = _path(_pose(0.0, 1.0, 0.0, 0.0), _pose(1.0, 1.0, 0.0, 0.0))
        out = apply_closure_to_cloud(cloud, prev, nxt)
        # World positions: (1.0, 0, 0) and (2.0, 0, 0). +1m → (2,0,0), (3,0,0).
        # voxel_size = 0.5, so voxels should be (4, 0, 0) and (6, 0, 0).
        sorted_out = np.sort(out.voxels, axis=0)
        np.testing.assert_array_equal(sorted_out, np.array([[4, 0, 0], [6, 0, 0]]))

    def test_recent_voxel_follows_latest_node_correction(self) -> None:
        """A voxel with a recent event timestamp warps by the late-node delta.

        Older voxel (no event, effective ts=0) clips to the early node which has
        zero correction; recent voxel clips to the late node which has a +5m shift.

        Note: PoseStamped maps ts=0 to time.time() as a "missing" sentinel, so
        we use ts >= 1.0 throughout to keep the pose-graph timeline well-defined.
        """
        cloud = DynamicCloud(
            voxels=np.array([[0, 0, 0], [10, 0, 0]], dtype=np.int32),
            quantity=np.array([1, 1], dtype=np.uint32),
            event_indices=np.array([1], dtype=np.uint32),
            event_timestamps=np.array([100 * 1_000_000_000], dtype=np.uint64),
            voxel_size=1.0,
        )
        prev = _path(_pose(1.0, 0.0, 0.0, 0.0), _pose(100.0, 0.0, 0.0, 0.0))
        # First node unchanged; second node shifted +5m in x.
        nxt = _path(_pose(1.0, 0.0, 0.0, 0.0), _pose(100.0, 5.0, 0.0, 0.0))
        out = apply_closure_to_cloud(cloud, prev, nxt)
        # Voxel 0 (no event → ts=0 → clipped to first node, ts=1 → identity delta): (0,0,0)
        # Voxel 1 (event_ts=100s → clipped to last node → +5m): (10,0,0) → (15,0,0)
        sorted_out = np.sort(out.voxels, axis=0)
        np.testing.assert_array_equal(sorted_out, np.array([[0, 0, 0], [15, 0, 0]]))

    def test_mismatched_length_raises(self) -> None:
        cloud = DynamicCloud(voxel_size=0.5)
        prev = _path(_pose(0.0, 0.0, 0.0, 0.0))
        nxt = _path(_pose(0.0, 0.0, 0.0, 0.0), _pose(1.0, 0.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="length mismatch"):
            apply_closure_to_cloud(cloud, prev, nxt)

    def test_mismatched_timestamps_raises(self) -> None:
        cloud = DynamicCloud(voxel_size=0.5)
        prev = _path(_pose(0.0, 0.0, 0.0, 0.0), _pose(1.0, 0.0, 0.0, 0.0))
        nxt = _path(_pose(0.0, 0.0, 0.0, 0.0), _pose(5.0, 0.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="timestamps do not match"):
            apply_closure_to_cloud(cloud, prev, nxt)

    def test_path_to_arrays_returns_timestamps_and_matrices(self) -> None:
        path = _path(_pose(1.0, 2.0, 3.0, 4.0), _pose(5.0, 6.0, 7.0, 8.0))
        ts, T = path_to_arrays(path)
        np.testing.assert_array_equal(ts, [1.0, 5.0])
        assert T.shape == (2, 4, 4)
        np.testing.assert_allclose(T[0, :3, 3], [2.0, 3.0, 4.0])
        np.testing.assert_allclose(T[1, :3, 3], [6.0, 7.0, 8.0])
