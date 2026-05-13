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

"""Session manager — one coordinator, many recipes, one shared DB.

Owns the full lifecycle of a characterization session: bring up the
coordinator (with recorder + optional teleop), run each ``TestRecipe``
in turn, write artifacts to disk.

Per-run output:

    {run_dir}/
        run.json              — run metadata (clock anchor, recipe, BMS)
        cmd_monotonic.jsonl   — one line per commanded sample, monotonic-clock timed

Per-session output:

    session_<ts>/
        session.json          — plan + completion state
        recording.db          — memory2 SQLite; sliced per run via ts_window_wall
        000_<recipe>_r1of3/   — one dir per planned run

The ``CharacterizationSession`` class is the per-recipe runner; it owns
LCM publishing and the busy-wait timing loop. ``SessionManager`` wraps
it across many recipes under one coordinator boot.

Typical flow (used by ``python -m dimos.utils.characterization.scripts.run_session``):

    with SessionManager.build(plan, output_root=...) as mgr:
        mgr.start_coordinator()
        for entry in mgr.plan:
            mgr.prompt_operator(entry)   # ENTER / s / r / q — teleop in between
            mgr.run(entry)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.core.global_config import global_config
from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.utils.characterization.recipes import TestRecipe
from dimos.utils.characterization.recorder import BmsLogger, CharacterizationRecorder

if TYPE_CHECKING:
    from dimos.core.coordination.blueprints import Blueprint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- helpers


def _session_id() -> str:
    return f"session_{time.strftime('%Y%m%d-%H%M%S')}"


def _git_sha(repo_root: Path | None = None) -> str | None:
    """Return the current commit SHA, or ``None`` if git isn't available."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
            timeout=2.0,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _generate_run_id(recipe_name: str) -> str:
    """Timestamp-prefixed, recipe-name-suffixed id. Matches dimos run_registry style."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in recipe_name)
    return f"{stamp}-{safe}"


def _dimos_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("dimos")
    except Exception:
        return None


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(data, fh, indent=2, default=str)
        fh.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------- dataclasses


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    n_commanded: int
    exit_reason: str
    run_json: Path
    recording_db: Path
    cmd_monotonic_jsonl: Path


@dataclass
class OperatorMetadata:
    """Operator-supplied run context. All fields optional; all stored in run.json."""

    surface: str | None = None
    payload_kg: float | None = None
    gait_mode: str | None = None
    notes: str | None = None
    ground_truth_source: str = "go2_onboard_odom"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "surface": self.surface,
            "payload_kg": self.payload_kg,
            "gait_mode": self.gait_mode,
            "notes": self.notes,
            "ground_truth_source": self.ground_truth_source,
        }
        out.update(dict(self.extra))
        return out


@dataclass(frozen=True)
class PlannedRun:
    """One (recipe, repeat_index) entry in a session plan."""

    recipe: TestRecipe
    repeat_index: int  # 1-based within the repeats of this recipe
    repeat_total: int  # total repeats requested for this recipe

    @property
    def label(self) -> str:
        """Filesystem-safe label used for the run dir name."""
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in self.recipe.name)
        return f"{safe}_r{self.repeat_index}of{self.repeat_total}"


@dataclass
class SessionResult:
    session_id: str
    session_dir: Path
    session_db: Path
    session_json: Path
    runs: list[RunResult] = field(default_factory=list)
    aborted: bool = False


def expand_plan(
    entries: Iterable[tuple[TestRecipe, int]],
    *,
    randomize: bool = False,
    rng_seed: int | None = None,
) -> list[PlannedRun]:
    """Turn ``[(recipe, repeats), ...]`` into a flat list of ``PlannedRun``.

    ``randomize`` shuffles the expanded list; pass ``rng_seed`` for
    reproducible sessions (e.g. for A/B tests). Randomization runs after
    expansion so each repeat is an independent slot in the permutation.
    """
    expanded: list[PlannedRun] = []
    for recipe, repeats in entries:
        if repeats <= 0:
            continue
        for i in range(1, repeats + 1):
            expanded.append(PlannedRun(recipe=recipe, repeat_index=i, repeat_total=repeats))
    if randomize:
        import random

        r = random.Random(rng_seed)
        r.shuffle(expanded)
    return expanded


def build_session_blueprint(
    db_path: Path,
    *,
    backend: str = "go2",
    include_teleop: bool = True,
    rage: bool = False,
) -> Blueprint:
    """Compose the session blueprint: coordinator + recorder (+ optional teleop).

    Returns a Blueprint with the Recorder pointed at ``db_path``; the
    caller builds a ``ModuleCoordinator`` from it.

    When ``include_teleop`` is True, we add the standard
    ``KeyboardTeleop`` module (runs in its own worker process, so
    pygame rendering doesn't contend with the control tick loop). It's
    configured with ``publish_only_when_active=True`` so its output
    stream is silent when no motion key is held - otherwise its 50Hz
    zero-Twist stream would fight with the recipe runner's commands on
    ``/cmd_vel``.

    When ``rage`` is True (go2 backend only), patches the GO2Connection
    blueprint atom to ``mode="rage"`` so the connection's start path
    runs StandUp -> BalanceStand -> enable_rage_mode after connect, and
    bumps the teleop linear/angular speeds to match.
    """
    if backend == "go2":
        from dimos.robot.unitree.go2.blueprints.basic.unitree_go2_coordinator import (
            unitree_go2_coordinator as base,
        )
    elif backend == "mock":
        if rage:
            raise ValueError("--rage is only valid with --backend go2")
        from dimos.control.blueprints.mobile import coordinator_mock_twist_base as base
    else:
        raise ValueError(f"unknown backend: {backend!r}")

    if rage:
        base = _patch_go2_mode(base, mode="rage")

    transports: dict[tuple[str, type], Any] = {
        ("commanded", Twist): LCMTransport("/cmd_vel", Twist),
    }
    if backend == "go2":
        transports[("measured", PoseStamped)] = LCMTransport("/go2/odom", PoseStamped)

    atoms = [CharacterizationRecorder.blueprint(db_path=str(db_path))]

    if include_teleop:
        from dimos.robot.unitree.keyboard_teleop import KeyboardTeleop

        teleop_kwargs: dict[str, Any] = {"publish_only_when_active": True}
        if rage:
            teleop_kwargs["linear_speed"] = 1.25
            teleop_kwargs["angular_speed"] = 1.2
        atoms.append(KeyboardTeleop.blueprint(**teleop_kwargs))

    return autoconnect(base, *atoms).transports(transports)


def _patch_go2_mode(bp: Blueprint, *, mode: str) -> Blueprint:
    """Return a copy of ``bp`` with the GO2Connection atom's kwargs updated
    to include ``mode=<mode>`` (e.g. "rage").

    The stock ``unitree_go2_coordinator`` calls ``GO2Connection.blueprint()``
    with no kwargs (mode defaults to DEFAULT). We need rage without
    duplicating the whole blueprint, so we mutate the atom's kwargs.
    """
    from dataclasses import replace

    from dimos.robot.unitree.go2.connection import GO2Connection

    new_atoms = []
    touched = False
    for atom in bp.blueprints:
        if atom.module is GO2Connection:
            new_kwargs = dict(atom.kwargs)
            new_kwargs["mode"] = mode
            new_atoms.append(replace(atom, kwargs=new_kwargs))
            touched = True
        else:
            new_atoms.append(atom)
    if not touched:
        logger.warning(
            "Mode patch: no GO2Connection atom found in blueprint, skipping (mode=%s)", mode
        )
        return bp
    return replace(bp, blueprints=tuple(new_atoms))


# ---------------------------------------------------------------------- per-recipe runner


class CharacterizationSession:
    """Run a ``TestRecipe`` once. Publishes Twists, writes artifacts, returns a ``RunResult``.

    Construct once, call :meth:`run` once per recipe. The LCM transport
    is started lazily on the first publish, so constructing a session is
    cheap and side-effect-free.
    """

    def __init__(
        self,
        *,
        cmd_vel_topic: str = "/cmd_vel",
        output_root: Path | str,
        bms: BmsLogger | None = None,
    ) -> None:
        self._cmd_vel = LCMTransport(cmd_vel_topic, Twist)
        self._cmd_vel_topic = cmd_vel_topic
        self._output_root = Path(output_root).expanduser().resolve()
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._bms = bms
        self._closed = False

    def close(self) -> None:
        """Stop the LCM transport. Safe to call multiple times."""
        if self._closed:
            return
        try:
            self._cmd_vel.stop()
        except Exception:  # pragma: no cover
            logger.exception("CharacterizationSession: transport stop failed")
        self._closed = True

    def __enter__(self) -> CharacterizationSession:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def run(
        self,
        recipe: TestRecipe,
        *,
        blueprint_name: str = "unitree_go2_characterization",
        simulation: bool = False,
        operator: OperatorMetadata | None = None,
        run_dir: Path | None = None,
        session_db_path: Path | None = None,
        session_id: str | None = None,
    ) -> RunResult:
        """If ``run_dir`` is given, use it (must already exist); else create one.

        ``session_db_path`` points at a session-level memory2 DB shared by
        all runs in a session. When provided, the run's ``run.json`` stores
        the relative path to that DB and a ``ts_window_wall`` so analysis
        can slice the shared DB to this run's data.
        """
        operator = operator or OperatorMetadata()
        if run_dir is not None:
            run_dir = Path(run_dir).expanduser().resolve()
            run_id = run_dir.name
        else:
            run_id = _generate_run_id(recipe.name)
            run_dir = self._output_root / run_id
            run_dir.mkdir(parents=True, exist_ok=False)

        run_json = run_dir / "run.json"
        cmd_jsonl = run_dir / "cmd_monotonic.jsonl"
        recording_db = run_dir / "recording.db"

        # Clock alignment anchor written before the first command.
        t_mono_start = time.monotonic()
        t_wall_start = time.time()

        bms_start = self._bms.snapshot() if self._bms is not None else None

        # Session-DB plumbing: store the relative path so the run dir is
        # relocatable as long as the session root is kept together.
        session_db_rel: str | None = None
        if session_db_path is not None:
            try:
                session_db_rel = str(
                    Path(session_db_path).resolve().relative_to(run_dir.resolve(), walk_up=True)
                )
            except (ValueError, TypeError):
                session_db_rel = str(Path(session_db_path).resolve())

        metadata_head = {
            "run_id": run_id,
            "session_id": session_id,
            "recipe": recipe.serialize(),
            "blueprint": blueprint_name,
            "simulation": simulation,
            "cmd_vel_topic": self._cmd_vel_topic,
            "started_at_wall": t_wall_start,
            "started_at_monotonic": t_mono_start,
            "clock_anchor": {"monotonic": t_mono_start, "wall": t_wall_start},
            "operator": operator.as_dict(),
            "git_sha": _git_sha(),
            "python_version": sys.version.split()[0],
            "dimos_version": _dimos_version(),
            "bms_start": bms_start,
            # One of these will be the measured-data source at analysis time.
            "recording_db": recording_db.name if session_db_rel is None else None,
            "session_db_path": session_db_rel,
            "cmd_monotonic_jsonl": cmd_jsonl.name,
            "bms_samples": [],  # appended below; filled in at finalize time
        }
        _write_json(run_json, metadata_head)

        exit_reason = "ok"
        n_commanded = 0
        bms_samples: list[dict[str, Any]] = []
        last_bms_mono: float = -1.0
        try:
            with cmd_jsonl.open("w") as fh:
                total = recipe.pre_roll_s + recipe.duration_s + recipe.post_roll_s
                dt = 1.0 / recipe.sample_rate_hz
                seq = 0
                t_start = time.monotonic()

                while True:
                    t_mono = time.monotonic()
                    t_rel = t_mono - t_start
                    if t_rel >= total:
                        break

                    # Phase: pre-roll [0, pre_roll_s); active [pre, pre+dur); post-roll after.
                    if t_rel < recipe.pre_roll_s:
                        vx, vy, wz = 0.0, 0.0, 0.0
                        phase = "pre_roll"
                    elif t_rel < recipe.pre_roll_s + recipe.duration_s:
                        t_active = t_rel - recipe.pre_roll_s
                        vx, vy, wz = recipe.signal_fn(t_active)
                        phase = "active"
                    else:
                        vx, vy, wz = 0.0, 0.0, 0.0
                        phase = "post_roll"

                    twist = Twist(Vector3(vx, vy, 0.0), Vector3(0.0, 0.0, wz))
                    self._cmd_vel.publish(twist)
                    t_wall = time.time()

                    fh.write(
                        json.dumps(
                            {
                                "seq": seq,
                                "tx_mono": t_mono,
                                "tx_wall": t_wall,
                                "phase": phase,
                                "vx": vx,
                                "vy": vy,
                                "wz": wz,
                            }
                        )
                        + "\n"
                    )
                    n_commanded += 1
                    seq += 1

                    # BMS at ~1 Hz
                    if self._bms is not None and self._bms.available:
                        if t_mono - last_bms_mono >= 1.0:
                            snap = self._bms.snapshot()
                            snap["t_mono"] = t_mono
                            snap["t_wall"] = t_wall
                            bms_samples.append(snap)
                            last_bms_mono = t_mono

                    next_t = t_start + (seq * dt)
                    sleep_s = next_t - time.monotonic()
                    if sleep_s > 0:
                        time.sleep(sleep_s)

                # One last zero-twist kick to guarantee the plant sees 0 on shutdown.
                self._cmd_vel.publish(Twist(Vector3(0, 0, 0), Vector3(0, 0, 0)))

        except KeyboardInterrupt:
            exit_reason = "interrupted"
            logger.warning("run %s interrupted by user", run_id)
        except Exception as e:
            exit_reason = f"exception:{type(e).__name__}:{e}"
            logger.exception("run %s failed", run_id)
        finally:
            bms_end = self._bms.snapshot() if self._bms is not None else None
            t_wall_end = time.time()
            metadata_head["completed_at_wall"] = t_wall_end
            metadata_head["completed_at_monotonic"] = time.monotonic()
            metadata_head["exit_reason"] = exit_reason
            metadata_head["n_commanded"] = n_commanded
            metadata_head["bms_end"] = bms_end
            metadata_head["bms_samples"] = bms_samples
            # Wall-clock window for session-DB slicing at analysis time.
            # Pad by 200 ms on each side so we don't clip samples arriving
            # right at the edge due to transport/callback delay.
            metadata_head["ts_window_wall"] = {
                "start": t_wall_start - 0.2,
                "end": t_wall_end + 0.2,
            }
            _write_json(run_json, metadata_head)

        return RunResult(
            run_id=run_id,
            run_dir=run_dir,
            n_commanded=n_commanded,
            exit_reason=exit_reason,
            run_json=run_json,
            recording_db=recording_db,
            cmd_monotonic_jsonl=cmd_jsonl,
        )


# ---------------------------------------------------------------------- session


class SessionManager:
    """Own the coordinator, recorder, and session-level artifacts across many recipes."""

    def __init__(
        self,
        *,
        session_id: str,
        session_dir: Path,
        plan: list[PlannedRun],
        backend: str,
        simulation: bool,
        include_teleop: bool,
        warmup_s: float,
        operator: OperatorMetadata,
        rage: bool = False,
    ) -> None:
        self.session_id = session_id
        self.session_dir = session_dir
        self.session_db = session_dir / "recording.db"
        self.session_json = session_dir / "session.json"
        self.plan = plan
        self.backend = backend
        self.simulation = simulation
        self.include_teleop = include_teleop
        self.warmup_s = warmup_s
        self.operator = operator
        self.rage = rage

        self._coord: ModuleCoordinator | None = None
        self._recipe_session: CharacterizationSession | None = None
        self._bms: BmsLogger | None = None
        self._closed = False
        self._runs: list[RunResult] = []
        self._aborted = False

    @classmethod
    def build(
        cls,
        plan: list[PlannedRun],
        *,
        output_root: Path | str,
        backend: str = "go2",
        simulation: bool = False,
        include_teleop: bool = True,
        warmup_s: float = 4.0,
        operator: OperatorMetadata | None = None,
        session_id: str | None = None,
        rage: bool = False,
    ) -> SessionManager:
        sid = session_id or _session_id()
        out_root = Path(output_root).expanduser().resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        sdir = out_root / sid
        sdir.mkdir(parents=True, exist_ok=False)
        return cls(
            session_id=sid,
            session_dir=sdir,
            plan=plan,
            backend=backend,
            simulation=simulation,
            include_teleop=include_teleop,
            warmup_s=warmup_s,
            operator=operator or OperatorMetadata(),
            rage=rage,
        )

    def __enter__(self) -> SessionManager:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # -------------------------------------------------------------------- lifecycle

    def start_coordinator(self) -> None:
        """Spin up the blueprint. Blocks for ``warmup_s`` before returning."""
        if self._coord is not None:
            return
        if self.simulation:
            global_config.update(simulation=True)

        bp = build_session_blueprint(
            self.session_db,
            backend=self.backend,
            include_teleop=self.include_teleop,
            rage=self.rage,
        )
        self._write_session_head(status="booting")
        logger.info(
            "session %s: building blueprint (%s%s)",
            self.session_id,
            self.backend,
            " [sim]" if self.simulation else "",
        )
        self._coord = ModuleCoordinator.build(bp)
        time.sleep(self.warmup_s)

        self._recipe_session = CharacterizationSession(
            cmd_vel_topic="/cmd_vel",
            output_root=self.session_dir,
            bms=self._try_make_bms_logger(),
        )
        self._write_session_head(status="ready")

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._recipe_session is not None:
                self._recipe_session.close()
        except Exception:  # pragma: no cover
            logger.exception("session %s: recipe session close failed", self.session_id)

        try:
            if self._coord is not None:
                logger.info("session %s: stopping coordinator...", self.session_id)
                self._coord.stop()
        except Exception:  # pragma: no cover
            logger.exception("session %s: coordinator stop failed", self.session_id)

        try:
            self._write_session_head(status="closed")
        except Exception:  # pragma: no cover
            logger.exception("session %s: final session.json write failed", self.session_id)

        self._closed = True

    # -------------------------------------------------------------------- recipe

    def run(self, entry: PlannedRun, *, run_index: int) -> RunResult:
        if self._recipe_session is None:
            raise RuntimeError("SessionManager.run() called before start_coordinator()")

        run_dir = self.session_dir / f"{run_index:03d}_{entry.label}"
        run_dir.mkdir(parents=True, exist_ok=False)
        result = self._recipe_session.run(
            entry.recipe,
            blueprint_name=f"{self.backend}_characterization",
            simulation=self.simulation,
            operator=self.operator,
            run_dir=run_dir,
            session_db_path=self.session_db,
            session_id=self.session_id,
        )
        self._runs.append(result)
        self._write_session_head(status="running")
        return result

    # -------------------------------------------------------------------- helpers

    def _try_make_bms_logger(self) -> BmsLogger | None:
        if self._coord is None:
            return None
        try:
            from dimos.robot.unitree.go2.connection import GO2Connection
        except Exception:
            return None
        try:
            go2 = self._coord.get_module(GO2Connection)
        except Exception:
            return None
        inner = getattr(go2, "connection", None)
        if inner is None or not hasattr(inner, "lowstate_stream"):
            return None
        return BmsLogger(inner)

    def _write_session_head(self, *, status: str) -> None:
        data: dict[str, Any] = {
            "session_id": self.session_id,
            "session_dir": str(self.session_dir),
            "backend": self.backend,
            "simulation": self.simulation,
            "rage": self.rage,
            "warmup_s": self.warmup_s,
            "operator": self.operator.as_dict(),
            "status": status,
            "plan": [
                {
                    "label": p.label,
                    "recipe": p.recipe.serialize(),
                    "repeat_index": p.repeat_index,
                    "repeat_total": p.repeat_total,
                }
                for p in self.plan
            ],
            "runs": [
                {
                    "run_id": r.run_id,
                    "run_dir": str(r.run_dir),
                    "exit_reason": r.exit_reason,
                    "n_commanded": r.n_commanded,
                }
                for r in self._runs
            ],
            "aborted": self._aborted,
            "updated_at_wall": time.time(),
        }
        tmp = self.session_json.with_suffix(".json.tmp")
        with tmp.open("w") as fh:
            json.dump(data, fh, indent=2, default=str)
            fh.write("\n")
        os.replace(tmp, self.session_json)

    def to_result(self) -> SessionResult:
        return SessionResult(
            session_id=self.session_id,
            session_dir=self.session_dir,
            session_db=self.session_db,
            session_json=self.session_json,
            runs=list(self._runs),
            aborted=self._aborted,
        )

    def mark_aborted(self) -> None:
        self._aborted = True
        self._write_session_head(status="aborted")


__all__ = [
    "CharacterizationSession",
    "OperatorMetadata",
    "PlannedRun",
    "RunResult",
    "SessionManager",
    "SessionResult",
    "build_session_blueprint",
    "expand_plan",
]
