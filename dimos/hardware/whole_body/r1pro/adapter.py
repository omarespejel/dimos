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

"""Galaxea R1 Pro whole-body adapter — implements WholeBodyAdapter via ROS 2.

Exposes the R1 Pro upper body (torso + left arm + right arm) through the
:class:`~dimos.hardware.whole_body.spec.WholeBodyAdapter` protocol.

**Joint order** (18 motors total):

.. list-table::
   :header-rows: 1

   * - Index range
     - Segment
     - Description
   * - 0 – 3
     - torso
     - torso_joint1–4 (pitch, pitch, pitch, yaw)
   * - 4 – 10
     - left arm
     - left_arm_joint1–7
   * - 11 – 17
     - right arm
     - right_arm_joint1–7

**Implementation note**: This adapter uses the high-level joint tracker
topics (``/motion_target/target_joint_state_*``) for safe, smooth control.
When a :class:`~dimos.hardware.whole_body.spec.MotorCommand` is received,
``cmd.q`` is used as the target position and ``cmd.dq`` (if non-sentinel) is
used as the tracking speed.  The ``kp``, ``kd``, and ``tau`` fields are
currently ignored because the joint tracker handles gains internally.

Future enhancement: add a ``use_low_level`` flag that routes to
``/motion_control/control_*`` (direct CAN) for full PD gain control.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dimos.hardware.whole_body.registry import WholeBodyAdapterRegistry

from dimos.hardware.whole_body.spec import (
    IMUState,
    MotorCommand,
    MotorState,
    POS_STOP,
    VEL_STOP,
)

log = logging.getLogger(__name__)

# Joint layout slices in the flat 18-element motor array
_TORSO_SLICE = slice(0, 4)
_LEFT_SLICE = slice(4, 11)
_RIGHT_SLICE = slice(11, 18)
_NUM_MOTORS = 18

# Default tracking speed (rad/s) when cmd.dq is the sentinel VEL_STOP.
_DEFAULT_TRACKING_SPEED = 0.5


class R1ProWholeBodyAdapter:
    """Galaxea R1 Pro whole-body adapter (18-DOF upper body).

    Implements the :class:`~dimos.hardware.whole_body.spec.WholeBodyAdapter`
    protocol by delegating to three sub-adapters:
    :class:`~dimos.hardware.manipulators.r1pro.adapter.R1ProTorsoAdapter` and
    two :class:`~dimos.hardware.manipulators.r1pro.adapter.R1ProArmAdapter`
    instances.

    The coordinator's tick loop calls ``write_motor_commands()`` on every
    tick with a list of 18 :class:`~dimos.hardware.whole_body.spec.MotorCommand`
    objects (one per motor), enabling smooth whole-body position control.

    Args:
        hardware_id: Coordinator hardware ID (e.g., ``"r1pro"``).
        network_interface: Unused; kept for registry interface compatibility.
    """

    def __init__(
        self,
        hardware_id: str = "r1pro",
        network_interface: int | str = 0,
        **_: object,
    ) -> None:
        from dimos.hardware.manipulators.r1pro.adapter import R1ProArmAdapter, R1ProTorsoAdapter

        self._hardware_id = hardware_id
        self._torso = R1ProTorsoAdapter(hardware_id=f"{hardware_id}_torso")
        self._left = R1ProArmAdapter(side="left", hardware_id=f"{hardware_id}_left")
        self._right = R1ProArmAdapter(side="right", hardware_id=f"{hardware_id}_right")

    # =========================================================================
    # Connection
    # =========================================================================

    def connect(self) -> bool:
        """Connect all three sub-adapters.  Rolls back on partial failure."""
        connected: list[Any] = []
        for sub in (self._torso, self._left, self._right):
            if sub.connect():
                connected.append(sub)
            else:
                log.error(
                    "R1Pro whole-body: sub-adapter %s failed to connect — rolling back",
                    sub.__class__.__name__,
                )
                for already in connected:
                    already.disconnect()
                return False
        log.info("R1Pro whole-body adapter connected (%d motors)", _NUM_MOTORS)
        return True

    def disconnect(self) -> None:
        for sub in (self._torso, self._left, self._right):
            sub.disconnect()
        log.info("R1Pro whole-body adapter disconnected")

    def is_connected(self) -> bool:
        return (
            self._torso.is_connected()
            and self._left.is_connected()
            and self._right.is_connected()
        )

    # =========================================================================
    # State Reading
    # =========================================================================

    def read_motor_states(self) -> list[MotorState]:
        """Read all 18 motor states as ``[MotorState(q, dq, tau), …]``."""
        torso_pos = self._torso.read_joint_positions()
        torso_vel = self._torso.read_joint_velocities()
        torso_eff = self._torso.read_joint_efforts()

        left_pos = self._left.read_joint_positions()
        left_vel = self._left.read_joint_velocities()
        left_eff = self._left.read_joint_efforts()

        right_pos = self._right.read_joint_positions()
        right_vel = self._right.read_joint_velocities()
        right_eff = self._right.read_joint_efforts()

        states: list[MotorState] = []
        for pos, vel, eff in zip(
            torso_pos + left_pos + right_pos,
            torso_vel + left_vel + right_vel,
            torso_eff + left_eff + right_eff,
        ):
            states.append(MotorState(q=pos, dq=vel, tau=eff))
        return states

    def read_imu(self) -> IMUState:
        """IMU state — torso IMU is published by the chassis adapter to LCM."""
        return IMUState()

    # =========================================================================
    # Control
    # =========================================================================

    def write_motor_commands(self, commands: list[MotorCommand]) -> bool:
        """Write per-motor commands to all 18 joints.

        ``cmd.q`` is used as target position.  ``cmd.dq`` is used as the
        tracking speed (rad/s) unless it equals the sentinel ``VEL_STOP``,
        in which case ``_DEFAULT_TRACKING_SPEED`` is used.  ``kp``, ``kd``,
        and ``tau`` are currently ignored (the joint tracker manages gains).

        Returns True only if all three sub-adapters accept their commands.
        """
        if len(commands) != _NUM_MOTORS:
            log.error(
                "R1Pro whole-body: expected %d commands, got %d",
                _NUM_MOTORS, len(commands),
            )
            return False

        def _tracking_speed(cmd: MotorCommand) -> float:
            if cmd.dq == VEL_STOP or cmd.dq == 0.0:
                return _DEFAULT_TRACKING_SPEED
            return abs(cmd.dq)

        torso_cmds = commands[_TORSO_SLICE]
        left_cmds = commands[_LEFT_SLICE]
        right_cmds = commands[_RIGHT_SLICE]

        # Skip writing if commanded position is the sentinel (no-op)
        def _positions_if_valid(cmds: list[MotorCommand]) -> list[float] | None:
            if all(c.q == POS_STOP for c in cmds):
                return None  # nothing to do
            return [c.q for c in cmds]

        torso_pos = _positions_if_valid(torso_cmds)
        left_pos = _positions_if_valid(left_cmds)
        right_pos = _positions_if_valid(right_cmds)

        ok_t = (
            self._torso.write_joint_positions(
                torso_pos,
                velocity=_tracking_speed(torso_cmds[0]),
            )
            if torso_pos is not None
            else True
        )
        ok_l = (
            self._left.write_joint_positions(
                left_pos,
                velocity=_tracking_speed(left_cmds[0]),
            )
            if left_pos is not None
            else True
        )
        ok_r = (
            self._right.write_joint_positions(
                right_pos,
                velocity=_tracking_speed(right_cmds[0]),
            )
            if right_pos is not None
            else True
        )
        return ok_t and ok_l and ok_r


def register(registry: "WholeBodyAdapterRegistry") -> None:
    """Register R1Pro whole-body adapter with the whole-body registry."""
    registry.register("r1pro_whole_body", R1ProWholeBodyAdapter)


__all__ = ["R1ProWholeBodyAdapter"]
