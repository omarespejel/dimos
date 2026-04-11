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

"""Galaxea R1 Pro robot configurations."""

from __future__ import annotations

from typing import Any

from dimos.control.components import HardwareComponent, HardwareType, make_twist_base_joints
from dimos.robot.config import GripperConfig, RobotConfig
from dimos.utils.data import LfsPath

R1PRO_MODEL_PATH = LfsPath("r1_pro_description") / "urdf" / "r1_pro.urdf"

# Collision exclusion pairs — structural mesh overlaps in the full-body URDF
# plus gripper parallel-linkage exclusions.
R1PRO_COLLISION_EXCLUSIONS: list[tuple[str, str]] = [
    # Chassis ↔ wheels (mesh overlap at zero pose)
    ("base_link", "wheel_motor_link1"),
    ("base_link", "wheel_motor_link2"),
    ("base_link", "wheel_motor_link3"),
    ("base_link", "steer_motor_link1"),
    ("base_link", "steer_motor_link2"),
    ("base_link", "steer_motor_link3"),
    # Torso ↔ arm shoulders (tight mesh fit)
    ("torso_link4", "left_arm_link1"),
    ("torso_link4", "right_arm_link1"),
    ("torso_link4", "left_arm_base_link"),
    ("torso_link4", "right_arm_base_link"),
    # Non-adjacent arm links that overlap at zero pose (link5 ↔ link7)
    ("left_arm_link5", "left_arm_link7"),
    ("right_arm_link5", "right_arm_link7"),
    # Left gripper
    ("left_arm_link7", "left_gripper_link"),
    ("left_gripper_link", "left_gripper_finger_link1"),
    ("left_gripper_link", "left_gripper_finger_link2"),
    ("left_gripper_finger_link1", "left_gripper_finger_link2"),
    ("left_gripper_link", "left_D405_link"),
    ("left_arm_link7", "left_D405_link"),
    # Right gripper
    ("right_arm_link7", "right_gripper_link"),
    ("right_gripper_link", "right_gripper_finger_link1"),
    ("right_gripper_link", "right_gripper_finger_link2"),
    ("right_gripper_finger_link1", "right_gripper_finger_link2"),
    ("right_gripper_link", "right_D405_link"),
    ("right_arm_link7", "right_D405_link"),
]


def r1pro_arm(
    side: str = "left",
    name: str | None = None,
    *,
    adapter_type: str = "mock",
    address: str | None = None,
    add_gripper: bool = False,
    **overrides: Any,
) -> RobotConfig:
    """Create an R1 Pro arm configuration for one side.

    Both sides point to the same full-body URDF. Each selects only its
    arm's 7 revolute joints. Drake's SetAutoRenaming handles duplicate
    model names when both arms are loaded into the same planning world.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    resolved_name = name or f"{side}_arm"
    joint_names = [f"{side}_arm_joint{i}" for i in range(1, 8)]
    ee_link = f"{side}_arm_link7"

    gripper = None
    if add_gripper:
        gripper = GripperConfig(
            type="r1pro",
            joints=[f"{side}_gripper_finger_joint1"],
            collision_exclusions=[
                pair
                for pair in R1PRO_COLLISION_EXCLUSIONS
                if side in pair[0] and side in pair[1]
            ],
            open_position=0.04,
            close_position=0.0,
        )

    defaults: dict[str, Any] = {
        "name": resolved_name,
        "model_path": R1PRO_MODEL_PATH,
        "end_effector_link": ee_link,
        "adapter_type": adapter_type,
        "address": address,
        "joint_names": joint_names,
        "base_link": "base_link",
        "home_joints": [0.0] * 7,
        "base_pose": [0, 0, 0, 0, 0, 0, 1],
        "package_paths": {
            "r1_pro_description": LfsPath("r1_pro_description"),
            "mobiman": LfsPath("r1_pro_description"),
        },
        "auto_convert_meshes": True,
        "collision_exclusion_pairs": R1PRO_COLLISION_EXCLUSIONS,
        "max_velocity": 0.5,
        "max_acceleration": 1.0,
        "gripper": gripper,
        "adapter_kwargs": {"side": side},
    }
    defaults.update(overrides)
    return RobotConfig(**defaults)


def r1pro_torso(
    hardware_id: str = "torso",
    adapter_type: str = "r1pro_torso",
    address: str | None = None,
) -> HardwareComponent:
    """Create an R1 Pro torso HardwareComponent (4-DOF).

    The torso stack has four revolute joints that pitch and yaw the upper
    body relative to the chassis.  It sits between the chassis and both
    arm shoulders and must be enabled before commanding either arm when
    the robot is in an unusual posture.

    Joint order exposed to the ControlCoordinator::

        [{hardware_id}/joint1, …, {hardware_id}/joint4]

    Args:
        hardware_id: Coordinator hardware ID — used as joint name prefix
            (e.g., ``"torso"`` → ``"torso/joint1"``).
        adapter_type: Adapter registry key (default: ``"r1pro_torso"``).
        address: Unused; kept for API consistency.
    """
    return HardwareComponent(
        hardware_id=hardware_id,
        hardware_type=HardwareType.MANIPULATOR,
        joints=[f"{hardware_id}/joint{i}" for i in range(1, 5)],
        adapter_type=adapter_type,
        address=address,
        adapter_kwargs={},
    )


def r1pro_whole_body(
    hardware_id: str = "r1pro",
    adapter_type: str = "r1pro_whole_body",
    address: str | None = None,
) -> HardwareComponent:
    """Create an R1 Pro whole-body HardwareComponent (18-DOF, WholeBodyAdapter).

    Uses the :class:`~dimos.hardware.whole_body.r1pro.adapter.R1ProWholeBodyAdapter`
    (``HardwareType.WHOLE_BODY``) which exposes the full upper body through the
    :class:`~dimos.hardware.whole_body.spec.WholeBodyAdapter` protocol.

    **Joint layout** (18 motors):

    ========== ======================== =======================================
    motor0–3   torso                   torso_joint1–4
    motor4–10  left arm                left_arm_joint1–7
    motor11–17 right arm               right_arm_joint1–7
    ========== ======================== =======================================

    Args:
        hardware_id: Coordinator hardware ID (e.g., ``"r1pro"``).
        adapter_type: Adapter registry key (default: ``"r1pro_whole_body"``).
        address: Unused; kept for API consistency.
    """
    from dimos.control.components import make_humanoid_joints

    return HardwareComponent(
        hardware_id=hardware_id,
        hardware_type=HardwareType.WHOLE_BODY,
        joints=make_humanoid_joints(hardware_id, dof=18),
        adapter_type=adapter_type,
        address=address,
        adapter_kwargs={},
    )


def r1pro_upper_body(
    hardware_id: str = "upper_body",
    adapter_type: str = "r1pro_upper_body",
    address: str | None = None,
) -> HardwareComponent:
    """Create an R1 Pro upper-body HardwareComponent (18-DOF composite).

    Exposes torso (4) + left arm (7) + right arm (7) as a single flat
    joint array.  Internally the adapter creates and manages individual
    :class:`R1ProTorsoAdapter` and :class:`R1ProArmAdapter` instances.
    Use this when an external policy or planner needs to command the full
    upper body without knowing sub-adapter boundaries.

    **Joint layout** (18 DOF total):

    ========== ========= ===================================================
    Indices    Segment   Description
    ========== ========= ===================================================
    0 – 3      torso     torso_joint1–4 (pitch, pitch, pitch, yaw)
    4 – 10     left      left_arm_joint1–7
    11 – 17    right     right_arm_joint1–7
    ========== ========= ===================================================

    Args:
        hardware_id: Coordinator hardware ID (e.g., ``"upper_body"``).
        adapter_type: Adapter registry key (default: ``"r1pro_upper_body"``).
        address: Unused; kept for API consistency.
    """
    joints = (
        [f"{hardware_id}/torso{i}" for i in range(1, 5)]
        + [f"{hardware_id}/left{i}" for i in range(1, 8)]
        + [f"{hardware_id}/right{i}" for i in range(1, 8)]
    )
    return HardwareComponent(
        hardware_id=hardware_id,
        hardware_type=HardwareType.MANIPULATOR,
        joints=joints,
        adapter_type=adapter_type,
        address=address,
        adapter_kwargs={},
    )


def r1pro_chassis(
    hardware_id: str = "chassis",
    adapter_type: str = "r1pro_chassis",
    address: str | None = None,
) -> HardwareComponent:
    """Create an R1 Pro chassis HardwareComponent.

    The chassis is a 3-DOF holonomic swerve drive (vx, vy, wz).
    On connect, the adapter opens Gate 1 (IK subscriber) and publishes
    all chassis-mounted sensors to independent LCM transports under
    ``/r1pro/{hardware_id}/``.

    Args:
        hardware_id: Coordinator hardware ID — also used to name sensor
            transport topics (e.g., ``"chassis"``).
        adapter_type: Adapter registry key (default: ``"r1pro_chassis"``).
        address: Unused; kept for consistency with arm factory.
    """
    return HardwareComponent(
        hardware_id=hardware_id,
        hardware_type=HardwareType.BASE,
        joints=make_twist_base_joints(hardware_id),
        adapter_type=adapter_type,
        address=address,
        adapter_kwargs={},
    )


__all__ = [
    "R1PRO_COLLISION_EXCLUSIONS",
    "R1PRO_MODEL_PATH",
    "r1pro_arm",
    "r1pro_chassis",
    "r1pro_torso",
    "r1pro_upper_body",
    "r1pro_whole_body",
]
