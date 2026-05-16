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

"""Pure-Python sim driver for the trajectory-tracking diagnostic.

Does not touch LCM, the coordinator, or the WebRTC stack. Instantiates
:class:`Go2PlantSim` (FOPDT velocity dynamics + unicycle kinematics) and
ticks it at 50 Hz against the same 6-trial battery the hardware path
runs.

The output session-dir layout matches the hardware path so
``diagnose.py`` can consume either source uniformly. Each tick row in
``cmd_monotonic.jsonl`` carries both the reference state and the
plant's measured pose — sim has no separate ``recording.db`` here.

Usage::

    python -m dimos.utils.characterization.scripts.sim_trajectory_diagnostic
    python -m dimos.utils.characterization.scripts.sim_trajectory_diagnostic \\
        --output-root ~/char_data/sim_diagnostic
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any

from dimos.control.tasks.feedforward_gain_compensator import (
    FeedforwardGainCompensator,
    FeedforwardGainConfig,
)
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.utils.benchmarking.plant import Go2PlantSim
from dimos.utils.benchmarking.plant_models import GO2_PLANT_FITTED
from dimos.utils.characterization.controllers import (
    ControllerFn,
    lowgain_p_controller,
    openloop_ff_controller,
)
from dimos.utils.characterization.trajectories import (
    ControllerMode,
    Trajectory,
    circle,
    sinusoidal_wz,
    step_vx,
    step_wz,
    trapezoidal_vx,
)

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE_HZ = 50.0
DEFAULT_PRE_ROLL_S = 0.5
DEFAULT_POST_ROLL_S = 1.0


# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatteryEntry:
    label: str
    trajectory: Trajectory


def build_battery() -> list[BatteryEntry]:
    """The 6-trial diagnostic battery (matches the hardware driver)."""
    return [
        BatteryEntry("step_vx_0.6", step_vx(v_target=0.6, duration=4.0)),
        BatteryEntry("step_wz_0.8", step_wz(vx=0.4, w_target=0.8, duration=4.0)),
        BatteryEntry("circle_R0.5", circle(v=0.5, w=1.0, duration=25.0)),
        BatteryEntry("circle_R1.0", circle(v=0.5, w=0.5, duration=25.0)),
        BatteryEntry("trapezoidal_vx_0.8", trapezoidal_vx(v_max=0.8, accel=0.5, duration=6.0)),
        BatteryEntry(
            "sinusoidal_wz_0.6",
            sinusoidal_wz(vx=0.4, w_amp=0.6, freq_hz=0.25, duration=20.0),
        ),
    ]


# ---------------------------------------------------------------------------


def _pose(x: float, y: float, yaw: float) -> PoseStamped:
    return PoseStamped(
        position=Vector3(x, y, 0.0),
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, yaw)),
    )


def _make_controller(mode: ControllerMode, ff: FeedforwardGainCompensator) -> ControllerFn:
    if mode == "openloop_ff":
        return openloop_ff_controller(ff)
    return lowgain_p_controller(ff, k_pos=0.0, k_yaw=0.15)


def _make_ff_compensator() -> FeedforwardGainCompensator:
    """FF compensator using the vendored plant gains (matches what sim assumes)."""
    return FeedforwardGainCompensator(
        FeedforwardGainConfig(
            K_vx=GO2_PLANT_FITTED.vx.K,
            K_wz=GO2_PLANT_FITTED.wz.K,
        )
    )


# ---------------------------------------------------------------------------


def run_trial(
    entry: BatteryEntry,
    run_dir: Path,
    *,
    session_id: str,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    pre_roll_s: float = DEFAULT_PRE_ROLL_S,
    post_roll_s: float = DEFAULT_POST_ROLL_S,
) -> dict[str, Any]:
    """Run one trial in sim, write per-tick JSONL + run.json. Returns run summary."""
    run_dir.mkdir(parents=True, exist_ok=False)

    plant = Go2PlantSim(GO2_PLANT_FITTED)
    dt = 1.0 / sample_rate_hz
    plant.reset(0.0, 0.0, 0.0, dt)

    ff = _make_ff_compensator()
    mode = entry.trajectory.recommended_mode
    controller = _make_controller(mode, ff)

    cmd_jsonl = run_dir / "cmd_monotonic.jsonl"
    run_json = run_dir / "run.json"

    t_wall_start = time.time()
    t_mono_start = time.monotonic()

    duration = entry.trajectory.duration_s
    pre_roll_s + duration + post_roll_s
    n_pre = int(pre_roll_s / dt)
    n_active = int(duration / dt)
    n_post = int(post_roll_s / dt)
    n_commanded = 0

    with cmd_jsonl.open("w") as fh:
        seq = 0

        # Pre-roll: zero command, sim still ticks for transient settling.
        for _k in range(n_pre):
            plant.step(0.0, 0.0, 0.0, dt)
            t_mono = t_mono_start + seq * dt
            fh.write(
                json.dumps(
                    {
                        "seq": seq,
                        "tx_mono": t_mono,
                        "tx_wall": t_wall_start + seq * dt,
                        "phase": "pre_roll",
                        "vx": 0.0,
                        "vy": 0.0,
                        "wz": 0.0,
                        "ref_x": 0.0,
                        "ref_y": 0.0,
                        "ref_yaw": 0.0,
                        "ref_vx": 0.0,
                        "ref_wz": 0.0,
                        "pose_x": plant.x,
                        "pose_y": plant.y,
                        "pose_yaw": plant.yaw,
                        "measured_vx": plant.vx,
                        "measured_wz": plant.wz,
                    }
                )
                + "\n"
            )
            seq += 1
            n_commanded += 1

        # Active window
        for k in range(n_active):
            t_active = k * dt
            ref = entry.trajectory.ref_fn(t_active)
            pose = _pose(plant.x, plant.y, plant.yaw)
            cmd_vx, cmd_vy, cmd_wz = controller(t_active, pose, ref)
            plant.step(cmd_vx, cmd_vy, cmd_wz, dt)

            t_mono = t_mono_start + seq * dt
            fh.write(
                json.dumps(
                    {
                        "seq": seq,
                        "tx_mono": t_mono,
                        "tx_wall": t_wall_start + seq * dt,
                        "phase": "active",
                        "vx": cmd_vx,
                        "vy": cmd_vy,
                        "wz": cmd_wz,
                        "ref_x": ref.x,
                        "ref_y": ref.y,
                        "ref_yaw": ref.yaw,
                        "ref_vx": ref.vx,
                        "ref_wz": ref.wz,
                        "pose_x": plant.x,
                        "pose_y": plant.y,
                        "pose_yaw": plant.yaw,
                        "measured_vx": plant.vx,
                        "measured_wz": plant.wz,
                    }
                )
                + "\n"
            )
            seq += 1
            n_commanded += 1

        # Post-roll: zero command again.
        for _k in range(n_post):
            plant.step(0.0, 0.0, 0.0, dt)
            t_mono = t_mono_start + seq * dt
            fh.write(
                json.dumps(
                    {
                        "seq": seq,
                        "tx_mono": t_mono,
                        "tx_wall": t_wall_start + seq * dt,
                        "phase": "post_roll",
                        "vx": 0.0,
                        "vy": 0.0,
                        "wz": 0.0,
                        "ref_x": 0.0,
                        "ref_y": 0.0,
                        "ref_yaw": 0.0,
                        "ref_vx": 0.0,
                        "ref_wz": 0.0,
                        "pose_x": plant.x,
                        "pose_y": plant.y,
                        "pose_yaw": plant.yaw,
                        "measured_vx": plant.vx,
                        "measured_wz": plant.wz,
                    }
                )
                + "\n"
            )
            seq += 1
            n_commanded += 1

    t_wall_end = time.time()

    run_metadata: dict[str, Any] = {
        "run_id": entry.label,
        "session_id": session_id,
        "recipe": {
            "name": entry.label,
            "test_type": "trajectory",
            "duration_s": duration,
            "sample_rate_hz": sample_rate_hz,
            "pre_roll_s": pre_roll_s,
            "post_roll_s": post_roll_s,
            "metadata": {
                "trajectory_spec": entry.trajectory.spec,
                "controller_mode": mode,
            },
        },
        "blueprint": "sim_go2_plant",
        "simulation": True,
        "started_at_wall": t_wall_start,
        "started_at_monotonic": t_mono_start,
        "clock_anchor": {"monotonic": t_mono_start, "wall": t_wall_start},
        "completed_at_wall": t_wall_end,
        "completed_at_monotonic": time.monotonic(),
        "exit_reason": "ok",
        "n_commanded": n_commanded,
        "ts_window_wall": {
            "start": t_wall_start - 0.2,
            "end": t_wall_end + 0.2,
        },
        "cmd_monotonic_jsonl": cmd_jsonl.name,
        "ff_gains_used": {
            "K_vx": ff.cfg.K_vx,
            "K_wz": ff.cfg.K_wz,
        },
        "plant_fopdt_used": {
            "vx": {
                "K": GO2_PLANT_FITTED.vx.K,
                "tau": GO2_PLANT_FITTED.vx.tau,
                "L": GO2_PLANT_FITTED.vx.L,
            },
            "wz": {
                "K": GO2_PLANT_FITTED.wz.K,
                "tau": GO2_PLANT_FITTED.wz.tau,
                "L": GO2_PLANT_FITTED.wz.L,
            },
        },
    }
    with run_json.open("w") as fh:
        json.dump(run_metadata, fh, indent=2, default=str)
        fh.write("\n")

    return {
        "run_id": entry.label,
        "run_dir": str(run_dir),
        "n_commanded": n_commanded,
        "exit_reason": "ok",
    }


def run_battery(output_root: Path) -> Path:
    """Run the full 6-trial sim battery; return the session directory."""
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    session_id = f"session_sim_{time.strftime('%Y%m%d-%H%M%S')}"
    session_dir = output_root / session_id
    session_dir.mkdir(parents=True, exist_ok=False)

    battery = build_battery()
    summaries: list[dict[str, Any]] = []
    for i, entry in enumerate(battery):
        run_dir = session_dir / f"{i:03d}_{entry.label}"
        logger.info("sim trial %d/%d: %s", i + 1, len(battery), entry.label)
        summary = run_trial(entry, run_dir, session_id=session_id)
        summaries.append(summary)

    session_json = session_dir / "session.json"
    with session_json.open("w") as fh:
        json.dump(
            {
                "session_id": session_id,
                "session_dir": str(session_dir),
                "backend": "sim",
                "simulation": True,
                "status": "closed",
                "plan": [
                    {
                        "label": e.label,
                        "recipe": {
                            "name": e.label,
                            "metadata": {
                                "trajectory_spec": e.trajectory.spec,
                                "controller_mode": e.trajectory.recommended_mode,
                            },
                        },
                    }
                    for e in battery
                ],
                "runs": summaries,
                "aborted": False,
            },
            fh,
            indent=2,
            default=str,
        )
        fh.write("\n")

    return session_dir


# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Run the sim trajectory-tracking diagnostic battery."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("~/char_data/sim_diagnostic").expanduser(),
        help="Directory under which a new session dir will be created.",
    )
    args = parser.parse_args()

    session_dir = run_battery(args.output_root)
    print(f"sim session: {session_dir}")
    print(f"next: python -m dimos.utils.characterization.processing.diagnose {session_dir}")


if __name__ == "__main__":
    main()
