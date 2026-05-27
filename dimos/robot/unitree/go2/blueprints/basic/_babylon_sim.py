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

"""Babylon / pimsim viewer composition for the Go2.

Mirrors the G1 ``_babylon_blueprint`` helper in shape: the viewer
integrates ``/cmd_vel`` locally, publishes ``sim_odom`` as the
canonical ``/odom`` for downstream consumers, and (when a scene
package is configured) overlays the cooked visual GLB + browser
collision mesh. Compared to G1 we use a small box proxy MJCF — Go2's
real-hardware low-level loop runs on Unitree firmware via SPORT_MOD,
so the simulator only needs a kinematic base to track.
"""

from __future__ import annotations

from importlib import resources
import os
from pathlib import Path
from typing import Any

from dimos.core.coordination.blueprints import Blueprint
from dimos.core.global_config import global_config
from dimos.core.transport import LCMTransport
from dimos.experimental.pimsim.entity import EntityStateBatch
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.nav_msgs.Path import Path as PathMsg
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_GO2_PROXY_MJCF = resources.files("dimos.robot.unitree.go2").joinpath("go2_proxy.xml")
_DEFAULT_BABYLON_PORT = 8091


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _babylon_enabled() -> bool:
    return _env_bool("DIMOS_ENABLE_BABYLON", True)


def _scene_package() -> Any:
    """Load the cooked scene package if one is configured, else ``None``."""
    scene = getattr(global_config, "scene", "")
    if not scene:
        return None
    from dimos.simulation.scene_assets.spec import load_scene_package

    candidates = [
        Path.home() / ".cache" / "dimos" / "scene_packages" / scene / "scene.meta.json",
    ]
    for path in candidates:
        if path.exists():
            return load_scene_package(path)
    logger.info("Go2 babylon: no cooked scene package found for scene=%r", scene)
    return None


def go2_babylon_blueprint() -> Blueprint | None:
    """Build the BabylonSceneViewerModule blueprint for the Go2 in pimsim mode.

    Returns ``None`` when ``DIMOS_ENABLE_BABYLON=0``.  Caller is
    responsible for only invoking this when
    ``global_config.simulation in ("babylon", "pimsim")``.
    """
    if not _babylon_enabled():
        return None

    from dimos.experimental.pimsim.module import BabylonSceneViewerModule

    with resources.as_file(_GO2_PROXY_MJCF) as proxy_path:
        viewer_mjcf_path = str(proxy_path)

    kwargs: dict[str, Any] = dict(
        mjcf_path=viewer_mjcf_path,
        port=_env_int("DIMOS_BABYLON_PORT", _DEFAULT_BABYLON_PORT),
        # Babylon integrates cmd_vel directly — no MuJoCo at runtime.
        enable_sim=True,
        sim_rate=_env_float("DIMOS_BABYLON_SIM_RATE_HZ", 100.0),
        vehicle_height=_env_float("DIMOS_GO2_VEHICLE_HEIGHT", 0.40),
        step_offset=_env_float("DIMOS_BABYLON_STEP_OFFSET", 0.10),
        support_floor=_env_bool("DIMOS_BABYLON_SUPPORT_FLOOR", True),
        support_floor_z=_env_float("DIMOS_SCENE_SUPPORT_FLOOR_Z", 0.0),
        support_floor_size=_env_float("DIMOS_SCENE_SUPPORT_FLOOR_SIZE", 0.0),
        lock_z=True,
        pointcloud_hz=_env_float("DIMOS_BABYLON_POINTCLOUD_HZ", 2.0),
        pointcloud_max_points=_env_int("DIMOS_BABYLON_POINTCLOUD_MAX_POINTS", 70000),
    )

    scene_package = _scene_package()
    if scene_package is not None and scene_package.visual_path is not None:
        browser_collision_path = (
            str(scene_package.browser_collision_path)
            if scene_package.browser_collision_path is not None
            else None
        )
        kwargs.update(
            scene_path=str(scene_package.visual_path),
            scene_scale=scene_package.alignment.scale,
            scene_translation=scene_package.alignment.translation,
            scene_rotation_zyx_deg=scene_package.alignment.rotation_zyx_deg,
            scene_y_up=scene_package.alignment.y_up,
            browser_collision_path=browser_collision_path,
            initial_entities=scene_package.entities,
        )

    return BabylonSceneViewerModule.blueprint(**kwargs).transports(
        {
            ("joint_state", JointState): LCMTransport("/coordinator/joint_state", JointState),
            ("odom", PoseStamped): LCMTransport("/odom", PoseStamped),
            # sim_odom IS the canonical /odom — the connection's TF
            # republisher and every downstream consumer (mapping, planner)
            # read from this same LCM topic.
            ("sim_odom", PoseStamped): LCMTransport("/odom", PoseStamped),
            ("path", PathMsg): LCMTransport("/nav_path", PathMsg),
            ("pointcloud_overlay", PointCloud2): LCMTransport("/global_map", PointCloud2),
            ("cmd_vel", Twist): LCMTransport("/cmd_vel", Twist),
            ("clicked_point", PointStamped): LCMTransport("/clicked_point", PointStamped),
            ("point_goal", PointStamped): LCMTransport("/point_goal", PointStamped),
            ("workspace_image", Image): LCMTransport("/workspace_image", Image),
            ("entity_state_batch", EntityStateBatch): LCMTransport(
                "/entity_state_batch", EntityStateBatch
            ),
        }
    )


__all__ = ["go2_babylon_blueprint"]
