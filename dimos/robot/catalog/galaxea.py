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
        "package_paths": {"r1_pro_description": LfsPath("r1_pro_description")},
        "auto_convert_meshes": True,
        "collision_exclusion_pairs": R1PRO_COLLISION_EXCLUSIONS,
        "max_velocity": 0.5,
        "max_acceleration": 1.0,
        "gripper": gripper,
        "adapter_kwargs": {"side": side},
    }
    defaults.update(overrides)
    return RobotConfig(**defaults)


__all__ = [
    "R1PRO_COLLISION_EXCLUSIONS",
    "R1PRO_MODEL_PATH",
    "r1pro_arm",
]
