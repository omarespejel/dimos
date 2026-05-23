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

"""3D <-> pixel projection for a posed camera.

Pure geometry: a :class:`Camera` holds a :class:`CameraInfo` and a
:class:`Pose` (camera optical frame, in the same world frame as the points
you pass in) and projects 3D points to pixels and unprojects pixels to rays.

Frame conventions are the caller's job. ``pose`` is interpreted as the
camera *optical* frame pose in the world frame (OpenCV convention: z
forward along the optical axis, x right, y down).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from dimos.msgs.geometry_msgs.Point import Point
    from dimos.msgs.geometry_msgs.Pose import Pose
    from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo


@dataclass(frozen=True)
class Ray:
    """An infinite ray in 3D. ``direction`` is a unit vector.

    Local lightweight type for projection results. No standard
    ``geometry_msgs/Ray`` exists in ROS or dimos. If a future use case
    needs rays in memory2 streams or over LCM, promote this to a real
    message under ``dimos/msgs/`` — the ``origin`` / ``direction`` shape
    won't change.
    """

    origin: np.ndarray
    direction: np.ndarray


class Camera:
    """A camera with a known optical-frame pose, ready to project / unproject.

    The pose's orientation must rotate the camera optical frame (z forward,
    x right, y down) into the same world frame as the points/rays the
    caller passes. We do not translate between sensor / body / optical
    frames — that's the caller's responsibility.

    Supported distortion models (``CameraInfo.distortion_model``):

    * ``""`` or ``"plumb_bob"`` — Brown-Conrady radial-tangential
      (``D = [k1, k2, p1, p2, k3]``). Empty ``D`` or zero ``D`` collapses
      to pure pinhole.
    * ``"equidistant"`` — fisheye Kannala-Brandt (``D = [k1, k2, k3, k4]``).
    """

    def __init__(self, info: CameraInfo, pose: Pose) -> None:
        self._info = info
        self._pose = pose

        K = info.get_K_matrix()
        D = info.get_D_coeffs()
        model = info.distortion_model or "plumb_bob"
        if model not in ("plumb_bob", "equidistant"):
            raise ValueError(
                f"Unsupported distortion model {model!r}. "
                f"Supported: 'plumb_bob' (radtan), 'equidistant' (fisheye)."
            )

        self._K = np.ascontiguousarray(K, dtype=np.float64)
        if model == "plumb_bob":
            # plumb_bob accepts 4/5/8/12 coeffs; empty D collapses to pinhole.
            self._D = np.ascontiguousarray(D if D.size else np.zeros(5), dtype=np.float64).reshape(
                -1, 1
            )
        else:
            if D.size != 4:
                raise ValueError(
                    f"equidistant model requires 4 distortion coefficients, got {D.size}"
                )
            self._D = np.ascontiguousarray(D, dtype=np.float64).reshape(-1, 1)
        self._model = model

        # SE(3): camera-to-world and the closed-form inverse.
        # Pose orientation is (qx, qy, qz, qw); Rotation.from_quat uses scalar-last.
        q = pose.orientation
        t = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

        T_world_cam = np.eye(4, dtype=np.float64)
        T_world_cam[:3, :3] = R
        T_world_cam[:3, 3] = t
        self._T_world_cam = T_world_cam

        T_cam_world = np.eye(4, dtype=np.float64)
        T_cam_world[:3, :3] = R.T
        T_cam_world[:3, 3] = -R.T @ t
        self._T_cam_world = T_cam_world

    @property
    def info(self) -> CameraInfo:
        return self._info

    @property
    def pose(self) -> Pose:
        return self._pose

    @property
    def position(self) -> np.ndarray:
        return self._T_world_cam[:3, 3].copy()

    @property
    def T_world_cam(self) -> np.ndarray:
        return self._T_world_cam.copy()

    @property
    def T_cam_world(self) -> np.ndarray:
        return self._T_cam_world.copy()

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N,3) world points -> (N,2) float pixels + (N,) bool valid mask.

        Invalid = behind camera (z <= 0 after world->cam) OR projected pixel
        outside [0, W) x [0, H). Invalid rows hold NaN.
        """
        pts = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
        n = pts.shape[0]

        pts_h = np.concatenate([pts, np.ones((n, 1))], axis=1)
        p_cam = (self._T_cam_world @ pts_h.T).T[:, :3]
        valid_z = p_cam[:, 2] > 0

        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.zeros(3, dtype=np.float64)

        pixels = np.full((n, 2), np.nan, dtype=np.float64)
        if valid_z.any():
            p_valid = p_cam[valid_z].reshape(-1, 1, 3)
            if self._model == "plumb_bob":
                proj, _ = cv2.projectPoints(p_valid, rvec, tvec, self._K, self._D)
            else:
                proj, _ = cv2.fisheye.projectPoints(p_valid, rvec, tvec, self._K, self._D)
            pixels[valid_z] = proj.reshape(-1, 2)

        W, H = self._info.width, self._info.height
        u, v = pixels[:, 0], pixels[:, 1]
        in_img = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        valid = valid_z & in_img
        return pixels, valid

    def unproject(self, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N,2) pixels -> (origin: (3,), directions: (N,3) unit) in world frame.

        All rays share the camera's position. Pixels outside the image are
        accepted; the ray is still well-defined.
        """
        pix = np.ascontiguousarray(pixels, dtype=np.float64).reshape(-1, 1, 2)

        if self._model == "plumb_bob":
            normalized = cv2.undistortPoints(pix, self._K, self._D)
        else:
            normalized = cv2.fisheye.undistortPoints(pix, self._K, self._D)
        xy = normalized.reshape(-1, 2)

        dirs_cam = np.concatenate([xy, np.ones((xy.shape[0], 1))], axis=1)
        dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)

        R_world_cam = self._T_world_cam[:3, :3]
        dirs_world = (R_world_cam @ dirs_cam.T).T

        return self.position, dirs_world

    def project_point(self, point: Point) -> tuple[float, float] | None:
        """Project a single :class:`Point`. Returns ``None`` if invalid."""
        pts = np.array([[point.x, point.y, point.z]], dtype=np.float64)
        pixels, valid = self.project(pts)
        if not valid[0]:
            return None
        return float(pixels[0, 0]), float(pixels[0, 1])

    def ray(self, u: float, v: float) -> Ray:
        """Unproject a single pixel into a :class:`Ray` in world frame."""
        origin, dirs = self.unproject(np.array([[u, v]], dtype=np.float64))
        return Ray(origin=origin, direction=dirs[0])
