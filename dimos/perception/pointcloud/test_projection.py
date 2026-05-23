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

"""Invariants for the projection module.

Each test pins down a property of project/unproject. The fixtures cover the
three distortion paths (pinhole, plumb_bob, equidistant). Where a test only
makes sense for one model it's not parametrized.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dimos.msgs.geometry_msgs.Point import Point
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.perception.pointcloud.projection import Camera, Ray

W, H = 640, 480
FX = FY = 500.0
CX = W / 2.0
CY = H / 2.0


def make_info(
    model: str = "plumb_bob",
    D: list[float] | None = None,
    fx: float = FX,
    fy: float = FY,
    cx: float = CX,
    cy: float = CY,
) -> CameraInfo:
    if D is None:
        D = [0.0, 0.0, 0.0, 0.0, 0.0] if model == "plumb_bob" else [0.0, 0.0, 0.0, 0.0]
    return CameraInfo(
        height=H,
        width=W,
        distortion_model=model,
        D=D,
        K=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
    )


def identity_pose() -> Pose:
    return Pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def posed(x: float, y: float, z: float, rx: float, ry: float, rz: float) -> Pose:
    """Build a Pose at (x,y,z) with XYZ-intrinsic Euler rotation."""
    q = Rotation.from_euler("xyz", [rx, ry, rz]).as_quat()  # (qx, qy, qz, qw)
    return Pose(x, y, z, q[0], q[1], q[2], q[3])


# --- fixtures: three distortion variants -------------------------------------


@pytest.fixture(
    params=[
        ("pinhole", "plumb_bob", [0.0] * 5),
        ("radtan", "plumb_bob", [-0.30, 0.10, 0.001, -0.001, 0.0]),
        ("fisheye", "equidistant", [-0.05, 0.01, -0.005, 0.001]),
    ],
    ids=["pinhole", "radtan", "fisheye"],
)
def cam(request) -> Camera:
    _, model, D = request.param
    return Camera(make_info(model, D), identity_pose())


# --- 1-3. Round-trip identity across models ----------------------------------


def _frustum_points(n: int, max_off_axis_deg: float, rng) -> np.ndarray:
    """N random points in front of the optical axis, within a cone."""
    depths = rng.uniform(0.5, 10.0, size=n)
    az = rng.uniform(-1, 1, size=n) * np.radians(max_off_axis_deg)
    el = rng.uniform(-1, 1, size=n) * np.radians(max_off_axis_deg)
    x = depths * np.tan(az)
    y = depths * np.tan(el)
    z = depths
    return np.stack([x, y, z], axis=1)


def _recover_via_ray(cam: Camera, pixels: np.ndarray, originals: np.ndarray) -> np.ndarray:
    """For each pixel, unproject to a ray and pull the original point's depth back."""
    origin, dirs = cam.unproject(pixels)
    # t = dot(p - origin, dir) — the parameter along the ray at which p sits.
    t = np.einsum("ij,ij->i", originals - origin, dirs)
    return origin[None, :] + t[:, None] * dirs


def test_roundtrip_pinhole():
    cam = Camera(make_info("plumb_bob", [0.0] * 5), identity_pose())
    rng = np.random.default_rng(0)
    pts = _frustum_points(100, 25.0, rng)
    pixels, valid = cam.project(pts)
    assert valid.all()
    recovered = _recover_via_ray(cam, pixels[valid], pts[valid])
    assert np.allclose(recovered, pts[valid], atol=1e-6)


def test_roundtrip_radtan():
    cam = Camera(make_info("plumb_bob", [-0.30, 0.10, 0.001, -0.001, 0.0]), identity_pose())
    rng = np.random.default_rng(1)
    pts = _frustum_points(100, 25.0, rng)
    pixels, valid = cam.project(pts)
    assert valid.sum() >= 80  # most should survive
    recovered = _recover_via_ray(cam, pixels[valid], pts[valid])
    assert np.allclose(recovered, pts[valid], atol=1e-4)


def test_roundtrip_fisheye():
    # Wider-FoV intrinsics so 50°-off-axis points actually land in the frame.
    info = make_info("equidistant", [-0.05, 0.01, -0.005, 0.001], fx=200.0, fy=200.0)
    cam = Camera(info, identity_pose())
    rng = np.random.default_rng(2)
    pts = _frustum_points(100, 50.0, rng)  # fisheye handles wider FoV
    pixels, valid = cam.project(pts)
    assert valid.sum() >= 80
    recovered = _recover_via_ray(cam, pixels[valid], pts[valid])
    assert np.allclose(recovered, pts[valid], atol=1e-4)


# --- 4. Center pixel sanity --------------------------------------------------


def test_center_pixel_pinhole():
    cam = Camera(make_info("plumb_bob", [0.0] * 5), identity_pose())
    # Point straight ahead at varying depth — must project to (cx, cy).
    for d in (0.5, 1.0, 5.0, 100.0):
        pixels, valid = cam.project(np.array([[0.0, 0.0, d]]))
        assert valid[0]
        np.testing.assert_allclose(pixels[0], [CX, CY], atol=1e-9)


# --- 5. Pinhole closed-form match -------------------------------------------


def test_pinhole_closed_form():
    cam = Camera(make_info("plumb_bob", [0.0] * 5), identity_pose())
    pts = np.array([[0.3, -0.2, 2.0], [1.0, 0.5, 5.0], [-0.4, 0.1, 1.0]])
    pixels, valid = cam.project(pts)
    assert valid.all()
    expected = np.stack([FX * pts[:, 0] / pts[:, 2] + CX, FY * pts[:, 1] / pts[:, 2] + CY], axis=1)
    np.testing.assert_allclose(pixels, expected, atol=1e-9)


# --- 6. Behind-camera rejection ---------------------------------------------


def test_behind_camera_rejected(cam: Camera):
    pts = np.array([[0.0, 0.0, -1.0], [0.1, 0.1, -5.0], [0.0, 0.0, 0.0]])
    _, valid = cam.project(pts)
    assert not valid.any()


# --- 7. Outside-image rejection ---------------------------------------------


def test_outside_image_rejected_pinhole():
    cam = Camera(make_info("plumb_bob", [0.0] * 5), identity_pose())
    # An off-axis point that pinhole-projects well outside the frame.
    pts = np.array([[100.0, 0.0, 1.0]])
    pixels, valid = cam.project(pts)
    assert not valid[0]
    # Pixel value should still be a finite number (not NaN/inf).
    assert np.isfinite(pixels[0]).all()


# --- 8. CameraInfo respect (the "must change projection" invariant) ---------


def test_camera_info_change_changes_projection():
    pose = identity_pose()
    cam_a = Camera(make_info(fx=FX, fy=FY), pose)
    cam_b = Camera(make_info(fx=FX * 2, fy=FY), pose)
    p = np.array([[0.3, 0.0, 2.0]])
    pix_a, _ = cam_a.project(p)
    pix_b, _ = cam_b.project(p)
    # Doubling fx halves the deviation from cx (projection scales linearly).
    du_a = pix_a[0, 0] - CX
    du_b = pix_b[0, 0] - CX
    np.testing.assert_allclose(du_b, 2 * du_a, atol=1e-9)


# --- 9. Pose respect --------------------------------------------------------


def test_pose_translation_shifts_pixel():
    """Translating the camera +x by dx should shift a point's pixel -dx*fx/Z."""
    cam_at_origin = Camera(make_info("plumb_bob", [0.0] * 5), identity_pose())
    cam_translated = Camera(make_info("plumb_bob", [0.0] * 5), posed(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    p_world = np.array([[0.0, 0.0, 5.0]])  # point on optical axis from origin
    pix_a, va = cam_at_origin.project(p_world)
    pix_b, vb = cam_translated.project(p_world)
    assert va[0] and vb[0]
    # From the translated camera, point is at (-1, 0, 5) — should land left of center.
    expected_du = -FX * 1.0 / 5.0  # u shift in pixels
    np.testing.assert_allclose(pix_b[0, 0] - pix_a[0, 0], expected_du, atol=1e-9)


def test_pose_rotation_shifts_pixel():
    """Yaw the camera; a point that was on-axis ends up off-axis predictably."""
    yaw = np.radians(5.0)
    cam_yawed = Camera(make_info("plumb_bob", [0.0] * 5), posed(0.0, 0.0, 0.0, 0.0, yaw, 0.0))
    # Point straight ahead in world. Camera yawed +y => point now appears at -yaw angle.
    p_world = np.array([[0.0, 0.0, 5.0]])
    pix, valid = cam_yawed.project(p_world)
    assert valid[0]
    expected_u = CX + FX * np.tan(-yaw)
    np.testing.assert_allclose(pix[0, 0], expected_u, atol=1e-6)
    np.testing.assert_allclose(pix[0, 1], CY, atol=1e-6)


# --- 10. Cross-check vs raw cv2 ---------------------------------------------


def test_cv2_cross_check_radtan():
    D = [-0.30, 0.10, 0.001, -0.001, 0.0]
    cam = Camera(make_info("plumb_bob", D), identity_pose())
    pts = np.array([[0.3, -0.2, 2.0], [1.0, 0.5, 5.0], [-0.4, 0.1, 1.0]])
    pixels, _ = cam.project(pts)
    expected, _ = cv2.projectPoints(
        pts.reshape(-1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        cam._K,
        cam._D,
    )
    np.testing.assert_allclose(pixels, expected.reshape(-1, 2), atol=1e-9)


# --- 11. Distortion is actually applied -------------------------------------


def test_distortion_changes_pixel():
    pose = identity_pose()
    cam_pin = Camera(make_info("plumb_bob", [0.0] * 5), pose)
    cam_dist = Camera(make_info("plumb_bob", [-0.30, 0.10, 0.0, 0.0, 0.0]), pose)
    # Off-axis point — radial distortion should warp it visibly.
    p = np.array([[0.4, 0.4, 1.0]])
    pix_pin, _ = cam_pin.project(p)
    pix_dist, _ = cam_dist.project(p)
    pixel_shift = np.linalg.norm(pix_pin[0] - pix_dist[0])
    assert pixel_shift > 1.0, f"distortion barely moved pixel: {pixel_shift}"


# --- 12. Unit-norm direction -------------------------------------------------


def test_ray_direction_is_unit(cam: Camera):
    ray = cam.ray(CX + 30, CY - 50)
    assert isinstance(ray, Ray)
    np.testing.assert_allclose(np.linalg.norm(ray.direction), 1.0, atol=1e-12)


def test_unproject_directions_are_unit(cam: Camera):
    rng = np.random.default_rng(3)
    pixels = rng.uniform([0, 0], [W, H], size=(50, 2))
    _, dirs = cam.unproject(pixels)
    norms = np.linalg.norm(dirs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


# --- bonus: project_point convenience ----------------------------------------


def test_project_point_valid_and_invalid():
    cam = Camera(make_info("plumb_bob", [0.0] * 5), identity_pose())
    # Valid: on-axis at depth 2.
    got = cam.project_point(Point(0.0, 0.0, 2.0))
    assert got is not None
    np.testing.assert_allclose(got, (CX, CY), atol=1e-9)
    # Invalid: behind camera.
    assert cam.project_point(Point(0.0, 0.0, -2.0)) is None
