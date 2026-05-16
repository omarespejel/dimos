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

"""Hardware (or mock-backend) driver for the trajectory-tracking diagnostic.

Brings up a ``ControlCoordinator`` for the chosen backend, optionally with
teleop for operator-driven repositioning between trials, and runs the
6-trial battery — each trial as a ``TrajectoryRecipe`` whose controller
mode is picked by the trajectory primitive's :attr:`recommended_mode`.

Typical hardware session::

    python -m dimos.utils.characterization.scripts.run_trajectory_diagnostic \\
        --out-dir ~/char_data/diagnostic \\
        --surface concrete --notes "first diagnostic run"

Dry-run on mock backend (no robot, no LCM external publishers needed)::

    python -m dimos.utils.characterization.scripts.run_trajectory_diagnostic \\
        --backend mock --auto --out-dir /tmp/diag_mock_dryrun
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from dimos.control.tasks.feedforward_gain_compensator import (
    FeedforwardGainCompensator,
    FeedforwardGainConfig,
)
from dimos.utils.benchmarking.plant_models import GO2_PLANT_FITTED
from dimos.utils.characterization.controllers import (
    ControllerFn,
    lowgain_p_controller,
    openloop_ff_controller,
)
from dimos.utils.characterization.recipes import TrajectoryRecipe
from dimos.utils.characterization.scripts.sim_trajectory_diagnostic import (
    build_battery,
)
from dimos.utils.characterization.session import (
    OperatorMetadata,
    SessionManager,
)
from dimos.utils.characterization.trajectories import ControllerMode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------


def _make_ff() -> FeedforwardGainCompensator:
    return FeedforwardGainCompensator(
        FeedforwardGainConfig(
            K_vx=GO2_PLANT_FITTED.vx.K,
            K_wz=GO2_PLANT_FITTED.wz.K,
        )
    )


def _make_controller_fn(mode: ControllerMode, ff: FeedforwardGainCompensator) -> ControllerFn:
    if mode == "openloop_ff":
        return openloop_ff_controller(ff)
    return lowgain_p_controller(ff, k_pos=0.0, k_yaw=0.15)


def battery_as_recipes() -> list[TrajectoryRecipe]:
    ff = _make_ff()
    recipes: list[TrajectoryRecipe] = []
    for entry in build_battery():
        mode = entry.trajectory.recommended_mode
        recipes.append(
            TrajectoryRecipe(
                name=entry.label,
                trajectory=entry.trajectory,
                controller_fn=_make_controller_fn(mode, ff),
            )
        )
    return recipes


def _prompt(msg: str) -> str:
    try:
        return input(msg).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "q"


def _print_plan(recipes: list[TrajectoryRecipe]) -> None:
    print(f"trajectory diagnostic plan ({len(recipes)} trials):")
    for i, r in enumerate(recipes):
        meta = r.trajectory.spec
        mode = r.trajectory.recommended_mode
        print(
            f"  [{i + 1:2d}] {r.name:<26} kind={meta.get('kind'):<16} "
            f"dur={r.duration_s:.1f}s  mode={mode}"
        )


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the trajectory-tracking diagnostic battery on a Go2 base."
    )
    parser.add_argument(
        "--out-dir",
        default="/tmp/char_runs_diagnostic",
        help="Parent directory for the session subdir (default /tmp/char_runs_diagnostic)",
    )
    parser.add_argument(
        "--backend",
        choices=("go2", "mock"),
        default="go2",
        help="go2 (real/mujoco) or mock (no-robot plumbing dry-run)",
    )
    parser.add_argument("--simulation", action="store_true", help="Launch mujoco sim for go2")
    parser.add_argument(
        "--rage",
        action="store_true",
        help="Real-Go2 only: enable rage mode at connection start.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan then exit")
    parser.add_argument(
        "--no-teleop",
        action="store_true",
        help="Do not add the keyboard teleop task (recommended for --backend mock).",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Skip the operator prompt between trials (useful in sim / dry-run).",
    )
    parser.add_argument("--warmup-s", type=float, default=4.0)
    parser.add_argument("--surface", default=None)
    parser.add_argument("--payload-kg", type=float, default=None)
    parser.add_argument("--gait-mode", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    recipes = battery_as_recipes()
    _print_plan(recipes)
    if args.dry_run:
        return 0

    operator = OperatorMetadata(
        surface=args.surface,
        payload_kg=args.payload_kg,
        gait_mode=args.gait_mode,
        notes=args.notes,
    )

    # SessionManager.build takes the older PlannedRun list. We don't need it
    # for trajectory recipes (which the manager doesn't iterate over), so pass
    # an empty plan and drive the manager manually.
    with SessionManager.build(
        plan=[],
        output_root=args.out_dir,
        backend=args.backend,
        simulation=args.simulation,
        include_teleop=(not args.no_teleop) and args.backend == "go2",
        warmup_s=args.warmup_s,
        operator=operator,
        rage=args.rage,
    ) as mgr:
        try:
            mgr.start_coordinator()
        except Exception:
            logger.exception("coordinator failed to start; aborting session")
            mgr.mark_aborted()
            return 3
        print(f"session: {mgr.session_id}  dir: {mgr.session_dir}")
        if mgr.include_teleop:
            print("  (hold WASD/QE for keyboard teleop during pauses; release to resume)")

        if not args.auto and mgr.include_teleop:
            print()
            print("Position robot at the FIRST trial's start pose, then press ENTER.")
            print("Each trial starts at the robot's current pose — reposition between trials.")
            ans = _prompt("  [ENTER]=start / q=quit > ")
            if ans == "q":
                mgr.mark_aborted()
                print("session aborted before first trial.")
                return 0

        skipped = 0
        last_idx: int | None = None
        i = 0
        while i < len(recipes):
            recipe = recipes[i]
            mode = recipe.trajectory.recommended_mode
            print()
            print(
                f"--- [{i + 1}/{len(recipes)}] {recipe.name}  "
                f"dur={recipe.duration_s:.1f}s  mode={mode} ---"
            )
            if not args.auto:
                ans = _prompt("  [ENTER]=run / s=skip / r=repeat-last / q=quit > ")
                if ans == "q":
                    mgr.mark_aborted()
                    print("session aborted by operator.")
                    break
                if ans == "s":
                    print(f"  skipping {recipe.name}")
                    skipped += 1
                    i += 1
                    continue
                if ans == "r":
                    if last_idx is None:
                        print("  no previous trial to repeat; running this one.")
                    else:
                        recipe = recipes[last_idx]
                        print(f"  repeating previous: {recipe.name}")

            t_start = time.time()
            try:
                result = mgr.run_trajectory(recipe, run_index=i)
            except Exception:
                logger.exception("trial %d (%s) raised unhandled exception", i, recipe.name)
                skipped += 1
                i += 1
                continue
            dt = time.time() - t_start
            print(
                f"  -> {result.exit_reason}  samples={result.n_commanded}  "
                f"dir={result.run_dir.name}  elapsed={dt:.1f}s"
            )
            last_idx = i
            i += 1

        print()
        print(
            f"diagnostic session done. trials={len(mgr._runs)} skipped={skipped} "
            f"session_dir={mgr.session_dir}"
        )
        print()
        print("next:")
        print(f"  python -m dimos.utils.characterization.processing.diagnose {mgr.session_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
