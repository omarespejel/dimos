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

"""G1 arm teleop adapter.

Bridges the babylon viewer's ``HumanoidControlSpec`` to the
``ControlCoordinator``'s ``servo_arms`` task. The viewer's per-joint slider
HUD calls ``set_arm_joint(name, position)`` via RPC; this module publishes a
matching ``JointState`` on the ``joint_command`` stream (transport-mapped to
``/g1/joint_command``). The coordinator routes it to the priority-10
``servo_arms`` task that owns the 14 arm joints, while the priority-50
``groot_wbc`` task continues to own legs + waist for balance.

Joint names exposed to the viewer are the short form ("left_shoulder_pitch"),
matching the convention the babylon slider HUD assumes. We add the ``g1/``
hardware-id prefix internally when publishing.

Limits come straight from the G1 URDF, parsed once at module construction so
this module owns the source of truth (mismatched gains in older configs
have caused us trouble before).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import Out
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_HW_ID = "g1"
_URDF_PATH = Path(__file__).resolve().parent / "g1.urdf"

# Canonical arm-joint short names, left arm then right arm (matches
# ``make_humanoid_joints("g1")[15:]`` ordering).
_ARM_JOINT_NAMES: tuple[str, ...] = (
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
)


def _load_arm_limits() -> dict[str, tuple[float, float]]:
    """Parse arm joint limits from the G1 URDF. Maps short name → (lo, hi)."""
    tree = ET.parse(_URDF_PATH)
    root = tree.getroot()
    limits: dict[str, tuple[float, float]] = {}
    for short in _ARM_JOINT_NAMES:
        full = short + "_joint"
        for joint in root.iter("joint"):
            if joint.get("name") != full:
                continue
            limit_el = joint.find("limit")
            if limit_el is None:
                raise RuntimeError(f"G1 URDF: {full} has no <limit>")
            lo = float(limit_el.get("lower", "0"))
            hi = float(limit_el.get("upper", "0"))
            limits[short] = (lo, hi)
            break
        else:
            raise RuntimeError(f"G1 URDF: joint {full} not found")
    return limits


class G1ArmTeleop(Module):
    """Publishes arm joint targets to the coordinator on behalf of the viewer.

    Ports:
        joint_command (Out[JointState]): target positions routed by the
            coordinator to the ``servo_arms`` task.
    """

    joint_command: Out[JointState]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._limits = _load_arm_limits()

    @rpc
    def start(self) -> None:
        super().start()
        logger.info(
            "G1ArmTeleop ready (%d arm joints, limits from URDF)", len(self._limits)
        )

    @rpc
    def stop(self) -> None:
        super().stop()

    @rpc
    def arm_joint_limits(self) -> list[tuple[str, float, float]]:
        """(short_name, lower_rad, upper_rad) for each of the 14 arm joints,
        in left-then-right URDF order."""
        return [(name, *self._limits[name]) for name in _ARM_JOINT_NAMES]

    @rpc
    def set_arm_joint(self, name: str, position: float) -> bool:
        """Drive one arm joint. ``name`` is the short form (e.g.,
        ``left_shoulder_pitch``); we add the ``g1/`` prefix the coordinator
        expects. Position is clamped to URDF limits."""
        limit = self._limits.get(name)
        if limit is None:
            logger.warning("G1ArmTeleop: unknown arm joint %r", name)
            return False
        lo, hi = limit
        clamped = max(lo, min(hi, float(position)))
        msg = JointState(name=[f"{_HW_ID}/{name}"], position=[clamped])
        self.joint_command.publish(msg)
        return True

    @rpc
    def release_arms(self) -> bool:
        """Send all 14 arm joints back to neutral (zero pose). The servo task
        will hold them there until the next ``set_arm_joint`` overrides."""
        names = [f"{_HW_ID}/{n}" for n in _ARM_JOINT_NAMES]
        positions = [0.0] * len(_ARM_JOINT_NAMES)
        self.joint_command.publish(JointState(name=names, position=positions))
        return True


g1_arm_teleop = G1ArmTeleop.blueprint

__all__ = ["G1ArmTeleop", "g1_arm_teleop"]
