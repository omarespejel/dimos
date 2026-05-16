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

"""G1 GR00T whole-body-control blueprint.

``dimos --simulation mujoco run g1-groot-wbc`` uses MuJoCo as the whole-body
backend. ``dimos run g1-groot-wbc`` uses the real G1 DDS connection.
The coordinator/task stack is shared; only the hardware adapter and the
sim-only navigation/visualization modules are gated by configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys
from typing import Any

from dimos_lcm.std_msgs import Bool

from dimos.control.components import HardwareComponent, HardwareType
from dimos.control.coordinator import ControlCoordinator, TaskConfig
from dimos.control.tasks.g1_groot_wbc_task import (
    ARM_DEFAULT_POSE,
    G1_GROOT_KD,
    G1_GROOT_KP,
    g1_arms,
    g1_joints,
    g1_legs_waist,
)
from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.core.global_config import global_config
from dimos.core.transport import LCMTransport
from dimos.hardware.whole_body.spec import WholeBodyConfig
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.msgs.nav_msgs.Path import Path as PathMsg
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.robot.unitree.g1.wholebody_connection import G1WholeBodyConnection
from dimos.utils.data import LfsPath
from dimos.web.websocket_vis.websocket_vis_module import WebsocketVisModule

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[6]
_GROOT_MODEL_DIR = LfsPath("groot")
_MJCF_PATH = LfsPath("mujoco_sim/g1_gear_wbc.xml")
_G1_MESH_DIR = _REPO_ROOT / "data/g1_urdf/meshes"

_SIM_DOF = 29
_SIM_TICK_RATE_HZ = 50.0
_REAL_TICK_RATE_HZ = 500.0
_REAL_ARM_RAMP_SECONDS = 10.0
_SIM_POLICY_DECIMATION = 1
_DEFAULT_COMMAND_CENTER_PORT = 7779
_DEFAULT_VISER_PORT = 8082
_DEFAULT_BABYLON_PORT = 8091
_DEFAULT_POINTCLOUD_FPS = 2.0
_DEFAULT_LIDAR_VOXEL_SIZE_M = 0.05
_DEFAULT_LIDAR_CAMERA_WIDTH = 640
_DEFAULT_LIDAR_CAMERA_HEIGHT = 360
_DEFAULT_GLOBAL_MAP_VOXEL_SIZE_M = 0.05
_DEFAULT_CUSTOM_SCENE_SCALE = 0.05
_DEFAULT_OFFICE_MESH_SCALE = 2.0
_DEFAULT_G1_SPAWN_Z_M = 0.793


@dataclass(frozen=True)
class _BackendSelection:
    blueprint: Blueprint
    adapter_type: str
    adapter_address: str | Path
    viewer_mjcf_path: str | Path
    tick_rate: float
    auto_arm: bool
    auto_dry_run: bool
    default_ramp_seconds: float
    decimation: int | None
    arm_holder: TaskConfig | None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def _env_xyz(name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    values = tuple(float(part.strip()) for part in raw.split(","))
    if len(values) != 3:
        raise ValueError(f"{name} must be formatted as 'x,y,z'")
    return values


def _scene_mesh_config() -> tuple[
    str | None, float, tuple[float, float, float], tuple[float, float, float], bool
]:
    scene_mesh_override = os.environ.get("DIMOS_SCENE_MESH_PATH") or None
    default_scene_mesh = _REPO_ROOT / "data/dimos_office_mesh/dimos_office_mesh.glb"
    scene_mesh_path = scene_mesh_override or (
        str(default_scene_mesh) if default_scene_mesh.exists() else None
    )

    default_scale = (
        _DEFAULT_CUSTOM_SCENE_SCALE if scene_mesh_override else _DEFAULT_OFFICE_MESH_SCALE
    )
    scene_mesh_scale = _env_float("DIMOS_SCENE_MESH_SCALE", default_scale)
    scene_mesh_translation = _env_xyz("DIMOS_SCENE_MESH_TRANSLATION", (0.0, 0.0, 0.0))
    scene_mesh_rotation = _env_xyz("DIMOS_SCENE_MESH_ROTATION_ZYX_DEG", (0.0, 0.0, 0.0))
    scene_mesh_y_up = _env_bool("DIMOS_SCENE_MESH_Y_UP", scene_mesh_override is not None)
    return (
        scene_mesh_path,
        scene_mesh_scale,
        scene_mesh_translation,
        scene_mesh_rotation,
        scene_mesh_y_up,
    )


def _scene_backed_mjcf(
    scene_mesh_path: str | None,
    scene_mesh_scale: float,
    scene_mesh_translation: tuple[float, float, float],
    scene_mesh_rotation: tuple[float, float, float],
    scene_mesh_y_up: bool,
) -> tuple[str | Path, str | Path]:
    if not scene_mesh_path or not _env_bool("DIMOS_SCENE_MESH_COLLISION", True):
        return _MJCF_PATH, _MJCF_PATH

    from dimos.simulation.mujoco.mesh_scene import SceneMeshAlignment
    from dimos.simulation.mujoco.scene_mesh_to_mjcf import load_or_bake

    _, wrapper = load_or_bake(
        scene_mesh_path=scene_mesh_path,
        robot_mjcf_path=_MJCF_PATH,
        alignment=SceneMeshAlignment(
            scale=scene_mesh_scale,
            translation=scene_mesh_translation,
            rotation_zyx_deg=scene_mesh_rotation,
            y_up=scene_mesh_y_up,
        ),
        meshdir=_G1_MESH_DIR,
        include_visual_mesh=_env_bool("DIMOS_SCENE_MESH_VISUAL", False),
        rebake=_env_bool("DIMOS_SCENE_MESH_BAKE", False),
    )
    compiled = wrapper.with_name("compiled.mjb")
    return (compiled if compiled.exists() else wrapper), wrapper


def _select_backend() -> _BackendSelection:
    if not global_config.simulation:
        arm_holder = TaskConfig(
            name="servo_arms",
            type="servo",
            joint_names=g1_arms,
            priority=10,
            default_positions=ARM_DEFAULT_POSE,
            auto_start=True,
        )
        return _BackendSelection(
            blueprint=G1WholeBodyConnection.blueprint(release_sport_mode=True),
            adapter_type="transport_lcm",
            adapter_address="",
            viewer_mjcf_path=_MJCF_PATH,
            tick_rate=_REAL_TICK_RATE_HZ,
            auto_arm=False,
            auto_dry_run=True,
            default_ramp_seconds=_REAL_ARM_RAMP_SECONDS,
            decimation=None,
            arm_holder=arm_holder,
        )

    (
        scene_mesh_path,
        scene_mesh_scale,
        scene_mesh_translation,
        scene_mesh_rotation,
        scene_mesh_y_up,
    ) = _scene_mesh_config()
    sim_mjcf_path, viewer_mjcf_path = _scene_backed_mjcf(
        scene_mesh_path,
        scene_mesh_scale,
        scene_mesh_translation,
        scene_mesh_rotation,
        scene_mesh_y_up,
    )
    lidar_disabled = _env_bool("DIMOS_DISABLE_LIDAR", False)
    depth_cloud_enabled = _env_bool("DIMOS_ENABLE_DEPTH_CLOUD", False)

    from dimos.simulation.engines.mujoco_sim_module import MujocoSimModule

    backend = MujocoSimModule.blueprint(
        address=sim_mjcf_path,
        meshdir=str(_G1_MESH_DIR),
        headless=_env_bool("DIMOS_MUJOCO_HEADLESS", True),
        dof=_SIM_DOF,
        camera_name=os.environ.get("DIMOS_MUJOCO_CAMERA", "head_color"),
        enable_color=False,
        enable_depth=depth_cloud_enabled,
        enable_pointcloud=(not lidar_disabled) or depth_cloud_enabled,
        pointcloud_fps=_env_float("DIMOS_POINTCLOUD_FPS", _DEFAULT_POINTCLOUD_FPS),
        lidar_camera_names=(
            []
            if lidar_disabled
            else ["lidar_front_camera", "lidar_left_camera", "lidar_right_camera"]
        ),
        lidar_camera_width=_env_int("DIMOS_LIDAR_CAMERA_WIDTH", _DEFAULT_LIDAR_CAMERA_WIDTH),
        lidar_camera_height=_env_int("DIMOS_LIDAR_CAMERA_HEIGHT", _DEFAULT_LIDAR_CAMERA_HEIGHT),
        lidar_voxel_size=_env_float("DIMOS_LIDAR_VOXEL_SIZE", _DEFAULT_LIDAR_VOXEL_SIZE_M),
        enable_kinematic_base_control=_env_bool("DIMOS_KINEMATIC_BASE_CONTROL", False),
        enable_kinematic_joint_hold=_env_bool("DIMOS_MUJOCO_KINEMATIC_JOINT_HOLD", False),
        inject_legacy_assets=True,
        spawn_xy=global_config.mujoco_start_pos_float,
        spawn_z=_env_float("DIMOS_MUJOCO_START_Z", _DEFAULT_G1_SPAWN_Z_M),
    )
    return _BackendSelection(
        blueprint=backend,
        adapter_type="sim_mujoco_g1",
        adapter_address=sim_mjcf_path,
        viewer_mjcf_path=viewer_mjcf_path,
        tick_rate=_SIM_TICK_RATE_HZ,
        auto_arm=True,
        auto_dry_run=False,
        default_ramp_seconds=0.0,
        decimation=_SIM_POLICY_DECIMATION,
        arm_holder=None,
    )


def _coordinator_blueprint(selection: _BackendSelection) -> tuple[Blueprint, str]:
    cmd_vel_topic = "/cmd_vel" if global_config.simulation else "/g1/cmd_vel"
    task_configs = [
        TaskConfig(
            name="groot_wbc",
            type="g1_groot_wbc",
            joint_names=g1_legs_waist,
            priority=50,
            model_path=_GROOT_MODEL_DIR,
            hardware_id="g1",
            auto_start=True,
            auto_arm=selection.auto_arm,
            auto_dry_run=selection.auto_dry_run,
            default_ramp_seconds=selection.default_ramp_seconds,
            decimation=selection.decimation,
        ),
        *([selection.arm_holder] if selection.arm_holder is not None else []),
    ]

    coordinator = ControlCoordinator.blueprint(
        tick_rate=selection.tick_rate,
        publish_joint_state=True,
        joint_state_frame_id="coordinator",
        hardware=[
            HardwareComponent(
                hardware_id="g1",
                hardware_type=HardwareType.WHOLE_BODY,
                joints=g1_joints,
                adapter_type=selection.adapter_type,
                address=selection.adapter_address,
                auto_enable=True,
                wb_config=WholeBodyConfig(kp=tuple(G1_GROOT_KP), kd=tuple(G1_GROOT_KD)),
            ),
        ],
        tasks=task_configs,
    ).transports(
        {
            ("joint_state", JointState): LCMTransport("/coordinator/joint_state", JointState),
            ("joint_command", JointState): LCMTransport("/g1/joint_command", JointState),
            ("twist_command", Twist): LCMTransport(cmd_vel_topic, Twist),
            ("motor_states", JointState): LCMTransport("/g1/motor_states", JointState),
            ("imu", Imu): LCMTransport("/g1/imu", Imu),
            ("motor_command", MotorCommandArray): LCMTransport(
                "/g1/motor_command", MotorCommandArray
            ),
        }
    )
    return coordinator, cmd_vel_topic


def _websocket_blueprint(cmd_vel_topic: str) -> Blueprint:
    return WebsocketVisModule.blueprint(
        port=_env_int("DIMOS_COMMAND_CENTER_PORT", _DEFAULT_COMMAND_CENTER_PORT)
    ).transports(
        {
            ("tele_cmd_vel", Twist): LCMTransport(cmd_vel_topic, Twist),
        }
    )


def _sim_support_blueprints(mujoco_mjcf_path: str | Path) -> tuple[Blueprint, ...]:
    if not global_config.simulation:
        return ()

    from dimos.mapping.costmapper import CostMapper
    from dimos.mapping.static_costmap import StaticCostmapModule
    from dimos.mapping.voxels import VoxelGridMapper
    from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner

    lidar_disabled = _env_bool("DIMOS_DISABLE_LIDAR", False)
    (
        scene_mesh_path,
        scene_mesh_scale,
        scene_mesh_translation,
        scene_mesh_rotation,
        scene_mesh_y_up,
    ) = _scene_mesh_config()

    mapping_stack: tuple[Blueprint, ...] = (
        (StaticCostmapModule.blueprint(),)
        if lidar_disabled or not scene_mesh_path or sys.platform == "darwin"
        else (
            VoxelGridMapper.blueprint(
                voxel_size=_env_float(
                    "DIMOS_GLOBAL_MAP_VOXEL_SIZE", _DEFAULT_GLOBAL_MAP_VOXEL_SIZE_M
                )
            ).transports({("lidar", PointCloud2): LCMTransport("/lidar", PointCloud2)}),
            CostMapper.blueprint(),
        )
    )

    viser_stack: tuple[Blueprint, ...] = ()
    if _env_bool("DIMOS_ENABLE_VISER", True):
        try:
            from dimos.visualization.viser import ViserRenderModule

            viser_stack = (
                ViserRenderModule.blueprint(
                    splat_path=None,
                    mjcf_path=mujoco_mjcf_path,
                    port=_env_int(
                        "DIMOS_VISER_PORT", global_config.viser_port or _DEFAULT_VISER_PORT
                    ),
                    scene_mesh_path=scene_mesh_path,
                    scene_mesh_scale=scene_mesh_scale,
                    scene_mesh_translation=scene_mesh_translation,
                    scene_mesh_rotation_zyx_deg=scene_mesh_rotation,
                    scene_mesh_y_up=scene_mesh_y_up,
                ).transports(
                    {
                        ("joint_state", JointState): LCMTransport(
                            "/coordinator/joint_state", JointState
                        ),
                        ("odom", PoseStamped): LCMTransport("/odom", PoseStamped),
                        ("path", PathMsg): LCMTransport("/nav_path", PathMsg),
                        ("clicked_point", PointStamped): LCMTransport(
                            "/clicked_point", PointStamped
                        ),
                        ("pointcloud_overlay", PointCloud2): LCMTransport(
                            "/global_map", PointCloud2
                        ),
                    }
                ),
            )
        except ModuleNotFoundError as exc:
            logger.warning("Viser disabled because optional dependency is missing: %s", exc.name)

    return (
        *mapping_stack,
        ReplanningAStarPlanner.blueprint(),
        *viser_stack,
    )


def _babylon_blueprint(viewer_mjcf_path: str | Path, cmd_vel_topic: str) -> Blueprint | None:
    """Build the BabylonSceneViewerModule blueprint for either backend.

    Sim mode optionally overlays a scene-mesh visual; real mode uses the bare
    G1 MJCF. Returns ``None`` if ``DIMOS_ENABLE_BABYLON`` is unset.
    """
    if not _env_bool("DIMOS_ENABLE_BABYLON", False):
        return None

    from dimos.simulation.mujoco.model import get_assets
    from dimos.visualization.babylon_scene_viewer import BabylonSceneViewerModule

    kwargs: dict[str, Any] = dict(
        mjcf_path=viewer_mjcf_path,
        assets=get_assets(),
        port=_env_int("DIMOS_BABYLON_PORT", _DEFAULT_BABYLON_PORT),
    )
    if global_config.simulation:
        (
            scene_mesh_path,
            scene_mesh_scale,
            scene_mesh_translation,
            scene_mesh_rotation,
            scene_mesh_y_up,
        ) = _scene_mesh_config()
        scene_visual_override = os.environ.get("DIMOS_SCENE_VISUAL_PATH") or None
        scene_visual_path = scene_visual_override or scene_mesh_path
        kwargs.update(
            scene_path=scene_visual_path,
            scene_scale=_env_float("DIMOS_SCENE_VISUAL_SCALE", scene_mesh_scale),
            scene_translation=_env_xyz(
                "DIMOS_SCENE_VISUAL_TRANSLATION", scene_mesh_translation
            ),
            scene_rotation_zyx_deg=_env_xyz(
                "DIMOS_SCENE_VISUAL_ROTATION_ZYX_DEG", scene_mesh_rotation
            ),
            scene_y_up=_env_bool("DIMOS_SCENE_VISUAL_Y_UP", scene_mesh_y_up),
            pointcloud_hz=_env_float("DIMOS_BABYLON_POINTCLOUD_HZ", 2.0),
            pointcloud_max_points=_env_int("DIMOS_BABYLON_POINTCLOUD_MAX_POINTS", 70000),
        )

    return BabylonSceneViewerModule.blueprint(**kwargs).transports(
        {
            ("joint_state", JointState): LCMTransport(
                "/coordinator/joint_state", JointState
            ),
            ("odom", PoseStamped): LCMTransport("/odom", PoseStamped),
            ("path", PathMsg): LCMTransport("/nav_path", PathMsg),
            ("pointcloud_overlay", PointCloud2): LCMTransport("/global_map", PointCloud2),
            ("cmd_vel", Twist): LCMTransport(cmd_vel_topic, Twist),
            ("clicked_point", PointStamped): LCMTransport("/clicked_point", PointStamped),
            ("point_goal", PointStamped): LCMTransport("/point_goal", PointStamped),
        }
    )


def _arm_teleop_blueprint() -> Blueprint | None:
    """Bridge the babylon slider HUD to the coordinator's ``servo_arms`` task.

    Implements ``HumanoidControlSpec``; the viewer auto-wires it. Only worth
    starting when Babylon is enabled (nothing else consumes the spec).
    """
    if not _env_bool("DIMOS_ENABLE_BABYLON", False):
        return None
    from dimos.robot.unitree.g1.arm_teleop import G1ArmTeleop

    return G1ArmTeleop.blueprint().transports(
        {
            ("joint_command", JointState): LCMTransport("/g1/joint_command", JointState),
        }
    )


_backend_selection = _select_backend()
_coordinator, _cmd_vel_topic = _coordinator_blueprint(_backend_selection)
_babylon = _babylon_blueprint(_backend_selection.viewer_mjcf_path, _cmd_vel_topic)
_teleop = _arm_teleop_blueprint()
_optional = tuple(bp for bp in (_babylon, _teleop) if bp is not None)

g1_groot_wbc = autoconnect(
    _backend_selection.blueprint,
    _coordinator,
    _websocket_blueprint(_cmd_vel_topic),
    *_sim_support_blueprints(_backend_selection.viewer_mjcf_path),
    *_optional,
).transports(
    {
        ("odom", PoseStamped): LCMTransport("/odom", PoseStamped),
        ("cmd_vel", Twist): LCMTransport(_cmd_vel_topic, Twist),
        ("nav_cmd_vel", Twist): LCMTransport(_cmd_vel_topic, Twist),
        ("pointcloud", PointCloud2): LCMTransport("/lidar", PointCloud2),
        ("global_map", PointCloud2): LCMTransport("/global_map", PointCloud2),
        ("global_costmap", OccupancyGrid): LCMTransport("/global_costmap", OccupancyGrid),
        ("path", PathMsg): LCMTransport("/nav_path", PathMsg),
        ("clicked_point", PointStamped): LCMTransport("/clicked_point", PointStamped),
        ("point_goal", PointStamped): LCMTransport("/point_goal", PointStamped),
        ("goal_request", PoseStamped): LCMTransport("/goal_request", PoseStamped),
        ("stop_movement", Bool): LCMTransport("/stop_movement", Bool),
    }
)

__all__ = ["g1_groot_wbc"]
