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

"""Damped-least-squares IK for the X2 Ultra arms (Pinocchio-based).

We solve for a target end-effector pose (in the robot's pelvis frame) by
moving only the seven joints of the selected arm, leaving legs/waist/head
where they are. The X2 URDF has a fixed pelvis root and is purely revolute,
so njoints==nq==nv==31 and joint indices map directly into q.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin

_LEFT_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
)
_RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)
_LEFT_EE_FRAME = "left_wrist_roll_link"
_RIGHT_EE_FRAME = "right_wrist_roll_link"


@dataclass
class ArmChain:
    """Joint-index + end-effector-frame info for one arm."""

    qpos_indices: list[int]  # indices into the pinocchio q vector
    nv_indices: list[int]  # indices into the pinocchio v / Jacobian column space
    ee_frame_id: int
    joint_names: tuple[str, ...]


class X2ArmIK:
    """Pinocchio FK + damped-least-squares IK on the X2 Ultra arms."""

    def __init__(self, urdf_path: str | Path) -> None:
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()

        def _make_chain(joint_names: tuple[str, ...], ee_frame: str) -> ArmChain:
            qpos, nv = [], []
            for name in joint_names:
                if not self.model.existJointName(name):
                    raise RuntimeError(f"X2ArmIK: joint not found in URDF: {name}")
                jid = self.model.getJointId(name)
                j = self.model.joints[jid]
                # All arm joints are 1-DoF revolute → one slot each.
                qpos.append(int(j.idx_q))
                nv.append(int(j.idx_v))
            if not self.model.existFrame(ee_frame):
                raise RuntimeError(f"X2ArmIK: frame not found in URDF: {ee_frame}")
            return ArmChain(
                qpos_indices=qpos,
                nv_indices=nv,
                ee_frame_id=int(self.model.getFrameId(ee_frame)),
                joint_names=joint_names,
            )

        self.left = _make_chain(_LEFT_ARM_JOINT_NAMES, _LEFT_EE_FRAME)
        self.right = _make_chain(_RIGHT_ARM_JOINT_NAMES, _RIGHT_EE_FRAME)

    # ---- FK / state ----

    def fk_pose(self, q: np.ndarray, chain: ArmChain) -> pin.SE3:
        """End-effector pose in the pelvis frame for joint state q."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return pin.SE3(self.data.oMf[chain.ee_frame_id])

    def home_q(self) -> np.ndarray:
        """All-zero joint state (a plausible upright pose for X2)."""
        return pin.neutral(self.model)

    # ---- IK ----

    def solve(
        self,
        target_pose: pin.SE3,
        q_seed: np.ndarray,
        chain: ArmChain,
        *,
        max_iter: int = 80,
        eps: float = 1e-3,
        step: float = 0.5,
        damping: float = 1e-3,
        max_step_per_iter: float = 0.2,
    ) -> tuple[np.ndarray, bool, float]:
        """Damped-least-squares IK on the chain's seven joints only.

        Returns (q_solution, converged, final_error_norm). q_solution has the
        same shape as q_seed; only the chain's qpos slots are modified.
        """
        q = q_seed.copy()
        err_norm = float("inf")
        for _ in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            current = self.data.oMf[chain.ee_frame_id]
            err6 = pin.log6(current.actInv(target_pose)).vector  # 6-vec twist
            err_norm = float(np.linalg.norm(err6))
            if err_norm < eps:
                return q, True, err_norm
            # Frame Jacobian in the EE-local frame matches log6's convention.
            J = pin.computeFrameJacobian(
                self.model, self.data, q, chain.ee_frame_id, pin.ReferenceFrame.LOCAL
            )
            J_arm = J[:, chain.nv_indices]  # 6 × 7
            # DLS: dq = J^T (J J^T + λ²I)^-1 err
            JJt = J_arm @ J_arm.T
            dq_arm = J_arm.T @ np.linalg.solve(JJt + (damping**2) * np.eye(6), err6)
            # Clamp per-step magnitude so we don't whip the arm on big targets.
            scale = step
            mag = float(np.linalg.norm(dq_arm))
            if mag * scale > max_step_per_iter:
                scale = max_step_per_iter / mag
            for slot, jv in zip(chain.qpos_indices, range(len(chain.nv_indices)), strict=True):
                q[slot] += scale * dq_arm[jv]
        return q, False, err_norm
