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

"""Sweep the low-gain-P controller's yaw gain in sim before hardware.

The hardware diagnostic's closed-loop trials (circle, sinusoidal_wz) blew
up under ``k_yaw=0.15`` — that gain is too sluggish to track sustained
curvature or oscillating heading. Before spending robot time we sweep
``k_yaw`` against the known FOPDT plant and find the smallest gain that
brings the error down to the plant floor.

The key question this answers: as ``k_yaw`` rises, does the along-track
lag converge toward the FOPDT prediction ``(tau+L)*v`` (→ the closed-loop
blowup was pure controller tuning, no plant ceiling) or does it plateau
well above it (→ a real plant limit at sustained curvature, which would
genuinely motivate MPC)?

Usage::

    python -m dimos.utils.characterization.scripts.sweep_controller_gain
    python -m dimos.utils.characterization.scripts.sweep_controller_gain \\
        --k-yaw 0.15 0.3 0.5 0.8 1.2 2.0
"""

from __future__ import annotations

import argparse
import logging

from dimos.control.tasks.feedforward_gain_compensator import (
    FeedforwardGainCompensator,
    FeedforwardGainConfig,
)
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.utils.benchmarking.plant import Go2PlantSim
from dimos.utils.benchmarking.plant_models import GO2_PLANT_FITTED
from dimos.utils.benchmarking.scoring import (
    ExecutedTrajectory,
    ScoreResult,
    TrajectoryTick,
    score_run_with_trajectory,
)
from dimos.utils.characterization.controllers import lowgain_p_controller
from dimos.utils.characterization.trajectories import (
    Trajectory,
    circle,
    sinusoidal_wz,
)

logger = logging.getLogger(__name__)

DEFAULT_K_YAW = [0.15, 0.3, 0.5, 0.8, 1.2, 2.0]


def _pose(x: float, y: float, yaw: float) -> PoseStamped:
    return PoseStamped(
        position=Vector3(x, y, 0.0),
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, yaw)),
    )


def _simulate(traj: Trajectory, k_yaw: float, sample_rate_hz: float = 50.0) -> ScoreResult:
    """Run ``traj`` in sim under lowgain_p with the given k_yaw; score it.

    Mirrors the diagnostic's active-window loop: record the post-step pose
    at time ``t`` and score it against ``ref(t)``. The reference is in the
    local frame and the sim plant starts at the origin, so no anchoring is
    needed here (frames coincide).
    """
    ff = FeedforwardGainCompensator(
        FeedforwardGainConfig(K_vx=GO2_PLANT_FITTED.vx.K, K_wz=GO2_PLANT_FITTED.wz.K)
    )
    controller = lowgain_p_controller(ff, k_pos=0.0, k_yaw=k_yaw)

    plant = Go2PlantSim(GO2_PLANT_FITTED)
    dt = 1.0 / sample_rate_hz
    plant.reset(0.0, 0.0, 0.0, dt)

    n = int(traj.duration_s / dt)
    ticks: list[TrajectoryTick] = []
    for k in range(n):
        t = k * dt
        ref = traj.ref_fn(t)
        pose = _pose(plant.x, plant.y, plant.yaw)
        cmd_vx, cmd_vy, cmd_wz = controller(t, pose, ref)
        plant.step(cmd_vx, cmd_vy, cmd_wz, dt)
        ticks.append(
            TrajectoryTick(
                t=t,
                pose=_pose(plant.x, plant.y, plant.yaw),
                cmd_twist=Twist(
                    linear=Vector3(cmd_vx, cmd_vy, 0.0),
                    angular=Vector3(0.0, 0.0, cmd_wz),
                ),
                actual_twist=Twist(
                    linear=Vector3(plant.vx, plant.vy, 0.0),
                    angular=Vector3(0.0, 0.0, plant.wz),
                ),
            )
        )
    return score_run_with_trajectory(
        ExecutedTrajectory(ticks=ticks, arrived=True),
        traj.ref_fn,
        duration_s=traj.duration_s,
    )


def _expected_lag_m(traj: Trajectory) -> float:
    """FOPDT lag floor (tau+L)*|v| — the best any gain can achieve."""
    tau_l = max(
        GO2_PLANT_FITTED.vx.tau + GO2_PLANT_FITTED.vx.L,
        GO2_PLANT_FITTED.wz.tau + GO2_PLANT_FITTED.wz.L,
    )
    spec = traj.spec
    v = float(spec.get("vx", spec.get("v", 0.0)))
    return tau_l * abs(v)


def sweep(traj: Trajectory, k_yaws: list[float]) -> list[tuple[float, ScoreResult]]:
    return [(k, _simulate(traj, k)) for k in k_yaws]


def _print_sweep(name: str, traj: Trajectory, results: list[tuple[float, ScoreResult]]) -> None:
    floor = _expected_lag_m(traj)
    print(f"\n=== {name}  (FOPDT lag floor ≈ {floor:.3f} m) ===")
    print(f"{'k_yaw':>7} | {'lag_rms':>9} | {'lag/floor':>9} | {'cross_rms':>9} | {'head_rms':>9}")
    print("-" * 56)
    for k, s in results:
        ratio = s.along_track_lag_rms / floor if floor > 1e-9 else float("inf")
        print(
            f"{k:>7.2f} | {s.along_track_lag_rms:>9.4f} | {ratio:>9.2f} | "
            f"{s.cross_track_traj_rms:>9.4f} | {s.heading_err_traj_rms:>9.4f}"
        )

    # Knee heuristic: smallest k_yaw whose lag is within 1.5x of the best
    # achieved in the sweep (i.e. diminishing returns past it).
    best_lag = min(s.along_track_lag_rms for _, s in results)
    knee = None
    for k, s in results:
        if s.along_track_lag_rms <= 1.5 * best_lag:
            knee = (k, s)
            break
    if knee is not None:
        k, s = knee
        plateau = s.along_track_lag_rms / floor if floor > 1e-9 else float("inf")
        print(
            f"\n  knee: k_yaw={k:.2f} → lag_rms={s.along_track_lag_rms:.3f} m "
            f"({plateau:.2f}x the FOPDT floor)"
        )
        if plateau <= 2.0:
            print(
                "  → converges to the plant floor: the closed-loop blowup was "
                "controller tuning, NOT a plant ceiling."
            )
        else:
            print(
                "  → plateaus well above the plant floor even at high gain: "
                "evidence of a real limit at this regime (motivates MPC)."
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Sweep lowgain_p k_yaw against the FOPDT sim plant."
    )
    parser.add_argument(
        "--k-yaw",
        type=float,
        nargs="+",
        default=DEFAULT_K_YAW,
        help=f"k_yaw values to sweep (default: {DEFAULT_K_YAW})",
    )
    args = parser.parse_args()

    trajectories = {
        "circle_R1.0 (v=0.5, w=0.5, 25s)": circle(v=0.5, w=0.5, duration=25.0),
        "circle_R0.5 (v=0.5, w=1.0, 25s)": circle(v=0.5, w=1.0, duration=25.0),
        "sinusoidal_wz (vx=0.4, amp=0.6, 0.25Hz, 20s)": sinusoidal_wz(
            vx=0.4, w_amp=0.6, freq_hz=0.25, duration=20.0
        ),
    }
    for name, traj in trajectories.items():
        results = sweep(traj, args.k_yaw)
        _print_sweep(name, traj, results)


if __name__ == "__main__":
    main()
