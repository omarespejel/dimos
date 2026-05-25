#!/usr/bin/env python3

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

"""Minimal X2 Ultra stack: vis module only. Base for larger blueprints."""

import math
from typing import Any

from dimos.core.global_config import global_config
from dimos.visualization.vis_module import vis_module


def _convert_camera_info(camera_info: Any) -> Any:
    return camera_info.to_rerun(
        image_topic="/world/color_image",
        optical_frame="rgbd_head_front",
    )


def _convert_color_image(image: Any) -> list[tuple[str, Any]]:
    import cv2
    import rerun as rr

    _, buf = cv2.imencode(".jpg", image.data, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return [
        ("world/color_image", rr.EncodedImage(contents=bytes(buf.tobytes()), media_type="image/jpeg")),
        ("world/color_image", rr.Transform3D(parent_frame="tf#/rgbd_head_front")),
    ]


def _convert_rear_image(image: Any) -> list[tuple[str, Any]]:
    import cv2
    import rerun as rr

    _, buf = cv2.imencode(".jpg", image.data, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return [
        ("world/rear_image", rr.EncodedImage(contents=bytes(buf.tobytes()), media_type="image/jpeg")),
        ("world/rear_image", rr.Transform3D(parent_frame="tf#/rgb_head_rear")),
    ]


def _convert_depth_pointcloud(pointcloud: Any) -> list[tuple[str, Any]]:
    import rerun as rr

    return [
        ("world/pointcloud", pointcloud.to_rerun()),
        ("world/pointcloud", rr.Transform3D(parent_frame="tf#/rgbd_head_front")),
    ]


def _convert_lidar(pointcloud: Any) -> list[tuple[str, Any]]:
    import rerun as rr

    # X2Connection._on_lidar already rotates and translates the cloud into
    # base_link frame, so attach directly to base_link with no further TF.
    return [
        ("world/lidar", pointcloud.to_rerun()),
        ("world/lidar", rr.Transform3D(parent_frame="tf#/base_link")),
    ]


def _quat_xyzw_from_rpy(roll: float, pitch: float, yaw: float) -> list[float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def _static_world_coords(rr: Any) -> Any:
    return rr.ViewCoordinates.RIGHT_HAND_Z_UP


def _static_base_link(rr: Any) -> list[Any]:
    # X2 Ultra is ~1.65 m tall, ~0.45 m wide, ~0.3 m deep
    return [
        rr.Boxes3D(
            half_sizes=[0.225, 0.15, 0.825],
            colors=[(0, 200, 255)],
            fill_mode="MajorWireframe",
        ),
        rr.Transform3D(parent_frame="tf#/base_link"),
    ]


def _static_torso_link(rr: Any) -> Any:
    return rr.Transform3D(
        translation=[0.0, 0.0, 0.1550706468356796],
        parent_frame="tf#/base_link",
        child_frame="tf#/torso_link",
    )


def _static_head_yaw_link(rr: Any) -> Any:
    return rr.Transform3D(
        translation=[0.00834772789983268, 0.0, 0.309397077276121],
        parent_frame="tf#/torso_link",
        child_frame="tf#/head_yaw_link",
    )


def _static_head_pitch_link(rr: Any) -> Any:
    return rr.Transform3D(
        translation=[0.0, 0.0, 0.0889],
        parent_frame="tf#/head_yaw_link",
        child_frame="tf#/head_pitch_link",
    )


def _static_rgbd_head_front(rr: Any) -> Any:
    return rr.Transform3D(
        translation=[0.05761, -0.011183, -0.04837],
        rotation=rr.Quaternion(xyzw=_quat_xyzw_from_rpy(2.2689, 0.0, 1.5708)),
        parent_frame="tf#/head_pitch_link",
        child_frame="tf#/rgbd_head_front",
    )


def _static_rgb_head_rear(rr: Any) -> Any:
    return rr.Transform3D(
        translation=[-0.0834, 0.00026495, 0.0],
        rotation=rr.Quaternion(xyzw=_quat_xyzw_from_rpy(-1.5708, 0.0, 1.5676)),
        parent_frame="tf#/head_pitch_link",
        child_frame="tf#/rgb_head_rear",
    )


def _x2_rerun_blueprint() -> Any:
    """Split layout: stacked front+rear cameras on left, 3D world view on right."""
    import rerun as rr
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="world/color_image", name="Front"),
                rrb.Spatial2DView(origin="world/rear_image", name="Rear"),
            ),
            rrb.Spatial3DView(
                origin="world",
                name="3D",
                background=rrb.Background(kind="SolidColor", color=[0, 0, 0]),
                line_grid=rrb.LineGrid3D(
                    plane=rr.components.Plane3D.XY.with_distance(0.5),
                ),
            ),
            column_shares=[1, 2],
        ),
    )


rerun_config = {
    "blueprint": _x2_rerun_blueprint,
    "max_hz": {
        "world/color_image": 3.0,
        "world/rear_image": 2.0,
        "world/lidar": 3.0,
        "world/pointcloud": 3.0,
    },
    "visual_override": {
        "world/color_image": _convert_color_image,
        "world/rear_image": _convert_rear_image,
        "world/camera_info": _convert_camera_info,
        "world/pointcloud": _convert_depth_pointcloud,
        "world/lidar": _convert_lidar,
    },
    "static": {
        "world": _static_world_coords,
        "world/tf/base_link": _static_base_link,
        "world/tf/torso_link": _static_torso_link,
        "world/tf/head_yaw_link": _static_head_yaw_link,
        "world/tf/head_pitch_link": _static_head_pitch_link,
        "world/tf/rgbd_head_front": _static_rgbd_head_front,
        "world/tf/rgb_head_rear": _static_rgb_head_rear,
    },
}

agibot_x2_primitive = vis_module(
    viewer_backend=global_config.viewer,
    rerun_config=rerun_config,
)

__all__ = ["agibot_x2_primitive"]
