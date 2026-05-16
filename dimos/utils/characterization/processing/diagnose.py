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

"""Trajectory-tracking diagnostic — classifier and report generator.

Reads a session produced by either the sim driver
(``sim_trajectory_diagnostic.py``) or the hardware driver
(``run_trajectory_diagnostic.py``). For each trial:

  1. Reconstructs the reference ``r(t)`` from ``run.json`` via
     :func:`trajectory_from_spec`.
  2. Builds an :class:`ExecutedTrajectory` from the per-tick rows in
     ``cmd_monotonic.jsonl``.
  3. Scores via :func:`score_run_with_trajectory`.
  4. Runs a deliberately-dumb classifier over each diagnostic
     signature (saturation, estimation noise, jitter, odom bias,
     deadtime+lag).
  5. Emits per-trial plots and a session-level markdown report.

The classifier is a hint, not a verdict — the report includes the
full numeric breakdown alongside the top pick.

Usage::

    python -m dimos.utils.characterization.processing.diagnose <session_dir>
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.utils.benchmarking.plant_models import GO2_PLANT_FITTED
from dimos.utils.benchmarking.scoring import (
    ExecutedTrajectory,
    ScoreResult,
    TrajectoryTick,
    score_run_with_trajectory,
)
from dimos.utils.characterization.trajectories import (
    RefFn,
    Trajectory,
    anchor_trajectory,
    trajectory_from_spec,
)

logger = logging.getLogger(__name__)

# Tunables — keep loose; tighten after first hardware session.
SATURATION_WZ_THRESHOLD = 1.49  # rad/s, just under the 1.5 firmware envelope
SATURATION_VX_THRESHOLD = 0.99  # m/s, just under the ~1.0 envelope
SATURATION_FRAC_THRESHOLD = 0.30  # 30% of active window
NOISE_FLOOR_RATIO = 0.5  # pre-roll cross-track RMS > 0.5*active → estimation_noise
JITTER_DT_STD_THRESHOLD = 0.005  # 5 ms intersample std → jitter
ODOM_BIAS_DRIFT_THRESHOLD = 0.02  # m or rad over pre-roll → bias
# Lag-vs-FOPDT-floor bands. ratio = measured_lag / (tau+L)*v.
# <= CLEAN: tracking at/under the plant's physical floor (healthy).
# >= DEADTIME: lag clearly exceeds the floor (genuine deadtime bottleneck).
# Between: marginal.
CLEAN_RATIO_MAX = 1.5
DEADTIME_RATIO_MIN = 2.5


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class LoadedRun:
    run_dir: Path
    run_id: str
    run_json: dict[str, Any]
    rows: list[dict[str, Any]]  # parsed cmd_monotonic.jsonl
    trajectory: Trajectory
    _anchored_ref: RefFn | None = field(default=None, init=False, repr=False)

    @property
    def controller_mode(self) -> str:
        return self.run_json.get("recipe", {}).get("metadata", {}).get("controller_mode", "unknown")

    @property
    def start_pose(self) -> tuple[float, float, float]:
        """(x, y, yaw) the reference is anchored to.

        Prefers the explicit ``trajectory_start_pose`` written by the
        fixed ``run_trajectory`` (new recordings). Falls back to the
        recorded pose at the pre-roll → active boundary, which recovers
        the anchor for older recordings made before the fix (the robot
        is commanded zero during pre-roll, so the last pre-roll pose is
        a clean snapshot of where it actually started).
        """
        sp = self.run_json.get("trajectory_start_pose")
        if sp and sp.get("x") is not None:
            return float(sp["x"]), float(sp["y"]), float(sp["yaw"])
        pre = _rows_with_pose([r for r in self.rows if r.get("phase") == "pre_roll"])
        if pre:
            r = pre[-1]
            return float(r["pose_x"]), float(r["pose_y"]), float(r["pose_yaw"])
        act = _rows_with_pose([r for r in self.rows if r.get("phase") == "active"])
        if act:
            r = act[0]
            return float(r["pose_x"]), float(r["pose_y"]), float(r["pose_yaw"])
        return 0.0, 0.0, 0.0

    @property
    def anchored_ref_fn(self) -> RefFn:
        """``trajectory.ref_fn`` transformed into the world (odom) frame.

        This is what should be compared against the recorded poses —
        the raw ``trajectory.ref_fn`` is in the local (origin, yaw=0)
        frame and comparing it to world-frame poses measures the fixed
        start-pose offset, not plant behavior.
        """
        if self._anchored_ref is None:
            sx, sy, syaw = self.start_pose
            self._anchored_ref = anchor_trajectory(self.trajectory.ref_fn, sx, sy, syaw)
        return self._anchored_ref

    @property
    def duration_s(self) -> float:
        return float(self.run_json.get("recipe", {}).get("duration_s", 0.0))

    @property
    def sample_rate_hz(self) -> float:
        return float(self.run_json.get("recipe", {}).get("sample_rate_hz", 50.0))

    @property
    def pre_roll_s(self) -> float:
        return float(self.run_json.get("recipe", {}).get("pre_roll_s", 0.5))


def load_run(run_dir: Path) -> LoadedRun | None:
    run_json_path = run_dir / "run.json"
    jsonl_path = run_dir / "cmd_monotonic.jsonl"
    if not run_json_path.exists() or not jsonl_path.exists():
        return None
    with run_json_path.open() as fh:
        meta = json.load(fh)
    spec = meta.get("recipe", {}).get("metadata", {}).get("trajectory_spec")
    if not spec:
        logger.warning("%s: no trajectory_spec — skipping (not a trajectory run)", run_dir.name)
        return None
    trajectory = trajectory_from_spec(spec)
    rows: list[dict[str, Any]] = []
    with jsonl_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return LoadedRun(
        run_dir=run_dir,
        run_id=meta.get("run_id", run_dir.name),
        run_json=meta,
        rows=rows,
        trajectory=trajectory,
    )


def load_session(session_dir: Path) -> list[LoadedRun]:
    runs: list[LoadedRun] = []
    for run_dir in sorted(p for p in session_dir.iterdir() if p.is_dir()):
        run = load_run(run_dir)
        if run is not None:
            runs.append(run)
    return runs


# ---------------------------------------------------------------------------
# Active-window slicing + ExecutedTrajectory construction
# ---------------------------------------------------------------------------


def _active_rows(run: LoadedRun) -> list[dict[str, Any]]:
    return [r for r in run.rows if r.get("phase") == "active"]


def _preroll_rows(run: LoadedRun) -> list[dict[str, Any]]:
    return [r for r in run.rows if r.get("phase") == "pre_roll"]


def _rows_with_pose(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("pose_x") is not None and r.get("pose_y") is not None]


def _row_to_tick(row: dict[str, Any], t_active: float) -> TrajectoryTick | None:
    """Build a :class:`TrajectoryTick` from one JSONL row, or ``None`` if pose is missing."""
    px = row.get("pose_x")
    py = row.get("pose_y")
    pyaw = row.get("pose_yaw")
    if px is None or py is None or pyaw is None:
        return None
    pose = PoseStamped(
        position=Vector3(float(px), float(py), 0.0),
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, float(pyaw))),
    )
    cmd_twist = Twist(
        linear=Vector3(float(row.get("vx", 0.0)), float(row.get("vy", 0.0)), 0.0),
        angular=Vector3(0.0, 0.0, float(row.get("wz", 0.0))),
    )
    actual_twist = Twist(
        linear=Vector3(float(row.get("measured_vx") or 0.0), 0.0, 0.0),
        angular=Vector3(0.0, 0.0, float(row.get("measured_wz") or 0.0)),
    )
    return TrajectoryTick(t=t_active, pose=pose, cmd_twist=cmd_twist, actual_twist=actual_twist)


def build_executed_trajectory(run: LoadedRun) -> ExecutedTrajectory:
    """Convert the active-window rows into an :class:`ExecutedTrajectory`.

    Rows without recorded pose (e.g. mock backend with no odom publisher,
    or hardware run before odom warmup) are dropped. ``t`` is rebased to
    start at 0 at the first kept row, matching ``ref_fn(t)``'s domain.
    """
    rows = _active_rows(run)
    if not rows:
        return ExecutedTrajectory(ticks=[], arrived=False)
    t0_mono = float(rows[0]["tx_mono"])
    ticks: list[TrajectoryTick] = []
    for r in rows:
        tick = _row_to_tick(r, float(r["tx_mono"]) - t0_mono)
        if tick is not None:
            ticks.append(tick)
    return ExecutedTrajectory(ticks=ticks, arrived=bool(ticks))


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


@dataclass
class ClassifierSignal:
    name: str  # one of: saturation, estimation_noise, jitter, odom_bias, deadtime_lag
    score: float  # higher = more confident
    detail: str  # one-line human-readable reason


@dataclass
class TrialDiagnosis:
    run: LoadedRun
    score: ScoreResult
    signals: list[ClassifierSignal] = field(default_factory=list)
    classification: str = "unknown"
    classification_detail: str = ""

    @property
    def expected_lag_s(self) -> float:
        plant = self.run.run_json.get("plant_fopdt_used", None)
        if plant:
            tau_vx = float(plant["vx"]["tau"])
            L_vx = float(plant["vx"]["L"])
            tau_wz = float(plant["wz"]["tau"])
            L_wz = float(plant["wz"]["L"])
        else:
            tau_vx, L_vx = GO2_PLANT_FITTED.vx.tau, GO2_PLANT_FITTED.vx.L
            tau_wz, L_wz = GO2_PLANT_FITTED.wz.tau, GO2_PLANT_FITTED.wz.L
        # Use whichever channel is active in the trajectory spec.
        kind = self.run.trajectory.spec.get("kind", "")
        if kind in ("step_vx", "straight", "trapezoidal_vx"):
            return tau_vx + L_vx
        return max(tau_vx + L_vx, tau_wz + L_wz)


def _signal_saturation(run: LoadedRun) -> ClassifierSignal:
    rows = _active_rows(run)
    if not rows:
        return ClassifierSignal("saturation", 0.0, "no active rows")
    n = len(rows)
    n_wz_sat = sum(
        1 for r in rows if abs(float(r.get("measured_wz", 0.0))) >= SATURATION_WZ_THRESHOLD
    )
    n_vx_sat = sum(
        1 for r in rows if abs(float(r.get("measured_vx", 0.0))) >= SATURATION_VX_THRESHOLD
    )
    frac = max(n_wz_sat, n_vx_sat) / n
    score = frac if frac >= SATURATION_FRAC_THRESHOLD else 0.0
    detail = f"wz_sat={n_wz_sat}/{n}, vx_sat={n_vx_sat}/{n} ({frac:.0%} of active)"
    return ClassifierSignal("saturation", score, detail)


def _signal_estimation_noise(run: LoadedRun) -> ClassifierSignal:
    pre = _rows_with_pose(_preroll_rows(run))
    act = _rows_with_pose(_active_rows(run))
    if not pre or not act:
        return ClassifierSignal("estimation_noise", 0.0, "no pose data in pre-roll or active")
    # Pre-roll pose RMS deviation around its mean = noise floor of the estimator.
    pre_xs = np.array([float(r["pose_x"]) for r in pre])
    pre_ys = np.array([float(r["pose_y"]) for r in pre])
    pre_xy_rms = float(
        np.sqrt(np.mean((pre_xs - pre_xs.mean()) ** 2 + (pre_ys - pre_ys.mean()) ** 2))
    )
    # Active-window cross-track RMS — compute against the active-rebased ref.
    act_cross_rms = _active_cross_track_rms(run, act)
    if act_cross_rms < 1e-6:
        ratio = float("inf") if pre_xy_rms > 1e-6 else 0.0
    else:
        ratio = pre_xy_rms / act_cross_rms
    score = ratio if ratio >= NOISE_FLOOR_RATIO else 0.0
    detail = f"preroll_xy_rms={pre_xy_rms:.4f} m, active_cross_rms={act_cross_rms:.4f} m (ratio {ratio:.2f})"
    return ClassifierSignal("estimation_noise", score, detail)


def _active_cross_track_rms(run: LoadedRun, rows: list[dict[str, Any]]) -> float:
    rows = _rows_with_pose(rows)
    if not rows:
        return 0.0
    t0_mono = float(rows[0]["tx_mono"])
    sq = 0.0
    ref_fn = run.anchored_ref_fn
    for r in rows:
        t_active = float(r["tx_mono"]) - t0_mono
        ref = ref_fn(t_active)
        ex = float(r["pose_x"]) - ref.x
        ey = float(r["pose_y"]) - ref.y
        cross = -math.sin(ref.yaw) * ex + math.cos(ref.yaw) * ey
        sq += cross * cross
    return math.sqrt(sq / len(rows))


def _signal_jitter(run: LoadedRun) -> ClassifierSignal:
    rows = _active_rows(run)
    if len(rows) < 3:
        return ClassifierSignal("jitter", 0.0, "too few active rows")
    monos = np.array([float(r["tx_mono"]) for r in rows])
    diffs = np.diff(monos)
    std_s = float(np.std(diffs))
    score = std_s if std_s >= JITTER_DT_STD_THRESHOLD else 0.0
    detail = f"intersample dt std={std_s * 1000:.2f} ms (threshold {JITTER_DT_STD_THRESHOLD * 1000:.1f} ms)"
    return ClassifierSignal("jitter", score, detail)


def _signal_odom_bias(run: LoadedRun) -> ClassifierSignal:
    pre = _rows_with_pose(_preroll_rows(run))
    if len(pre) < 5:
        return ClassifierSignal("odom_bias", 0.0, "too few preroll rows with pose")
    # Linear least-squares slope of pose_x and pose_y over pre-roll time;
    # if either has magnitude over the threshold (per second), flag bias.
    monos = np.array([float(r["tx_mono"]) for r in pre])
    monos -= monos[0]
    xs = np.array([float(r["pose_x"]) for r in pre])
    ys = np.array([float(r["pose_y"]) for r in pre])
    # Slope = cov(t, p) / var(t)
    t_var = float(np.var(monos))
    if t_var < 1e-9:
        return ClassifierSignal("odom_bias", 0.0, "pre-roll time variance zero")
    slope_x = float(np.cov(monos, xs, bias=True)[0, 1] / t_var)
    slope_y = float(np.cov(monos, ys, bias=True)[0, 1] / t_var)
    drift_total = math.hypot(slope_x, slope_y) * (monos[-1] - monos[0])
    score = drift_total if drift_total >= ODOM_BIAS_DRIFT_THRESHOLD else 0.0
    detail = (
        f"preroll slope=({slope_x * 1000:.2f}, {slope_y * 1000:.2f}) mm/s; "
        f"total drift over pre-roll={drift_total * 1000:.2f} mm"
    )
    return ClassifierSignal("odom_bias", score, detail)


def _lag_ratio(
    run: LoadedRun, score: ScoreResult, expected_lag_s: float
) -> tuple[float, float, float]:
    """Return (ratio, measured_m, expected_m). ratio = measured / FOPDT floor."""
    v_ref = _nominal_vx(run.trajectory)
    expected_lag_m = expected_lag_s * abs(v_ref)
    measured = score.along_track_lag_rms
    ratio = 0.0 if expected_lag_m < 1e-6 else measured / expected_lag_m
    return ratio, measured, expected_lag_m


def _signal_clean(run: LoadedRun, score: ScoreResult, expected_lag_s: float) -> ClassifierSignal:
    """Fires when along-track lag is at or under the FOPDT floor.

    The floor ``(tau+L)*v`` is the lag an *ideal* controller against this
    plant cannot beat in open loop; a closed loop can partly compensate
    and come in under it. Lag at/under the floor means the plant +
    controller is tracking as well as the physics allows — there is no
    deadtime *bottleneck* to fix. This is the healthy outcome, and it
    must outrank the deadtime_lag fallback so recommend.py stops
    proposing MPC against a plant that's already at its floor.
    """
    ratio, measured, expected_m = _lag_ratio(run, score, expected_lag_s)
    fires = 0.0 < ratio <= CLEAN_RATIO_MAX
    sig = 1.0 if fires else 0.0
    detail = (
        f"along_track_lag_rms={measured:.4f} m vs FOPDT floor≈{expected_m:.4f} m "
        f"(ratio {ratio:.2f}; ≤{CLEAN_RATIO_MAX:.1f} ⇒ tracking at the plant's "
        f"physical floor, controllers are not the bottleneck)"
    )
    return ClassifierSignal("clean", sig, detail)


def _signal_deadtime_lag(
    run: LoadedRun, score: ScoreResult, expected_lag_s: float
) -> ClassifierSignal:
    """Fires only when lag *substantially exceeds* the FOPDT floor.

    That is the genuine deadtime-bound regime: the controller cannot keep
    up with what the plant could physically deliver, so acting on a
    predicted future pose (Smith predictor / MPC) is warranted. Lag near
    or under the floor is NOT this case — see :func:`_signal_clean`.
    """
    ratio, measured, expected_m = _lag_ratio(run, score, expected_lag_s)
    if ratio >= DEADTIME_RATIO_MIN:
        sig_score = 1.0  # clearly lag-dominated beyond the plant floor
    elif ratio > CLEAN_RATIO_MAX:
        sig_score = 0.4  # marginal: above floor but not clearly deadtime-bound
    else:
        sig_score = 0.0  # at/under floor — clean, not a deadtime problem
    detail = (
        f"measured along_track_lag_rms={measured:.4f} m, "
        f"expected≈{expected_m:.4f} m (tau+L={expected_lag_s:.3f}s, ratio {ratio:.2f}; "
        f"≥{DEADTIME_RATIO_MIN:.1f} ⇒ deadtime-bound)"
    )
    return ClassifierSignal("deadtime_lag", sig_score, detail)


def _nominal_vx(traj: Trajectory) -> float:
    spec = traj.spec
    kind = spec.get("kind", "")
    if kind == "straight":
        return float(spec.get("v", 0.0))
    if kind == "step_vx":
        return float(spec.get("v_target", 0.0))
    if kind in ("circle", "step_wz", "sinusoidal_wz"):
        return float(spec.get("vx", spec.get("v", 0.0)))
    if kind == "trapezoidal_vx":
        return float(spec.get("v_max", 0.0))
    return 0.0


def classify(run: LoadedRun, score: ScoreResult) -> TrialDiagnosis:
    diagnosis = TrialDiagnosis(run=run, score=score)
    expected_lag = diagnosis.expected_lag_s

    # Order matters for tie-breaks: an actual fault (saturation / noise /
    # jitter / bias) outranks "clean" even when lag is small, because the
    # fault is the real problem. "clean" outranks "deadtime_lag" because a
    # plant tracking at its physical floor has no deadtime bottleneck to
    # fix — this is what stops recommend.py proposing MPC on healthy data.
    signals = [
        _signal_saturation(run),
        _signal_estimation_noise(run),
        _signal_jitter(run),
        _signal_odom_bias(run),
        _signal_clean(run, score, expected_lag),
        _signal_deadtime_lag(run, score, expected_lag),
    ]
    diagnosis.signals = signals

    winner = max(signals, key=lambda s: s.score)
    if winner.score == 0.0:
        # Nothing fired and lag isn't large → benign, not alarmist.
        diagnosis.classification = "clean"
        diagnosis.classification_detail = (
            "no fault signal fired and lag is not above the FOPDT floor"
        )
    else:
        diagnosis.classification = winner.name
        diagnosis.classification_detail = winner.detail
    return diagnosis


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _decompose_per_tick(run: LoadedRun) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (t_active, along_lag, cross_track, heading_err) arrays."""
    rows = _rows_with_pose(_active_rows(run))
    if not rows:
        return np.array([]), np.array([]), np.array([]), np.array([])
    t0 = float(rows[0]["tx_mono"])
    ts = []
    along = []
    cross = []
    heading = []
    ref_fn = run.anchored_ref_fn
    for r in rows:
        t_active = float(r["tx_mono"]) - t0
        ref = ref_fn(t_active)
        ex = float(r["pose_x"]) - ref.x
        ey = float(r["pose_y"]) - ref.y
        c, s = math.cos(ref.yaw), math.sin(ref.yaw)
        along_signed = c * ex + s * ey
        lag = -along_signed
        ct = -s * ex + c * ey
        he = (float(r["pose_yaw"]) - ref.yaw + math.pi) % (2 * math.pi) - math.pi
        ts.append(t_active)
        along.append(lag)
        cross.append(ct)
        heading.append(he)
    return np.array(ts), np.array(along), np.array(cross), np.array(heading)


def plot_run(run: LoadedRun, diagnosis: TrialDiagnosis, out_dir: Path) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t, along, cross, heading = _decompose_per_tick(run)
    if t.size == 0:
        logger.warning("plot_run: %s has no pose data; skipping plot", run.run_id)
        return None
    rows = _rows_with_pose(_active_rows(run))
    measured_wz = np.array([float(r.get("measured_wz") or 0.0) for r in rows])
    cmd_wz = np.array([float(r.get("wz", 0.0)) for r in rows])
    ref_wz = np.array([float(r.get("ref_wz", 0.0)) for r in rows])

    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(t, along, color="#1f77b4")
    axes[0].axhline(0.0, color="#666", linewidth=0.8)
    axes[0].set_ylabel("along-track\nlag (m)")
    axes[0].set_title(f"{run.run_id} — {run.controller_mode} — class: {diagnosis.classification}")

    axes[1].plot(t, cross, color="#d62728")
    axes[1].axhline(0.0, color="#666", linewidth=0.8)
    axes[1].set_ylabel("cross-track\nerror (m)")

    axes[2].plot(t, heading, color="#2ca02c")
    axes[2].axhline(0.0, color="#666", linewidth=0.8)
    axes[2].set_ylabel("heading\nerror (rad)")

    axes[3].plot(t, ref_wz, color="#aaa", label="ref_wz")
    axes[3].plot(t, cmd_wz, color="#1f77b4", linewidth=0.8, label="cmd_wz")
    axes[3].plot(t, measured_wz, color="#ff7f0e", linewidth=0.8, label="measured_wz")
    axes[3].axhline(1.5, color="#d62728", linewidth=0.5, linestyle="--")
    axes[3].axhline(-1.5, color="#d62728", linewidth=0.5, linestyle="--")
    axes[3].set_ylabel("wz (rad/s)")
    axes[3].set_xlabel("t (s, active window)")
    axes[3].legend(loc="best", fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "diagnosis.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _format_score(score: ScoreResult) -> str:
    return (
        f"- along_track_lag: rms={score.along_track_lag_rms:.4f} m, max={score.along_track_lag_max:.4f} m\n"
        f"- cross_track:     rms={score.cross_track_traj_rms:.4f} m, max={score.cross_track_traj_max:.4f} m\n"
        f"- heading_err:     rms={score.heading_err_traj_rms:.4f} rad, max={score.heading_err_traj_max:.4f} rad\n"
        f"- traj_completed:  {score.traj_completed_on_time_pct:.0%}\n"
        f"- linear_speed_rms: {score.linear_speed_rms:.3f} m/s\n"
        f"- angular_speed_rms: {score.angular_speed_rms:.3f} rad/s\n"
        f"- cmd_rate_integral: {score.cmd_rate_integral:.3f}\n"
        f"- n_ticks: {score.n_ticks}"
    )


def _format_signals(signals: list[ClassifierSignal]) -> str:
    return "\n".join(f"  - **{s.name}** (score {s.score:.3f}): {s.detail}" for s in signals)


def make_report(session_dir: Path, diagnoses: list[TrialDiagnosis]) -> Path:
    report_dir = session_dir / "diagnose"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.md"

    lines: list[str] = []
    lines.append(f"# Trajectory-tracking diagnostic — `{session_dir.name}`\n")
    lines.append("## Interpretation note\n")
    lines.append(
        "`along_track_lag` is the spatial distance between the robot and the reference,\n"
        "measured in the reference yaw frame. **Positive = robot is BEHIND the reference.**\n"
        "Under the FOPDT plant model, the *expected* steady-state lag is `(tau + L) * v_ref`\n"
        "— a perfectly-behaved robot will always show this. The diagnostic is interested in\n"
        "**deviations from this baseline**, not the absolute number. See the\n"
        "`expected≈X m` annotation on the `deadtime_lag` signal in each trial.\n"
    )
    lines.append("\n## Classifier signals (per trial)\n")
    for d in diagnoses:
        lines.append(f"### {d.run.run_id} (mode: `{d.run.controller_mode}`)\n")
        lines.append(f"**Classification: `{d.classification}`** — {d.classification_detail}\n")
        lines.append("Score:\n")
        lines.append(_format_score(d.score) + "\n")
        lines.append("Signals:\n")
        lines.append(_format_signals(d.signals) + "\n")
        png_rel = f"./{d.run.run_dir.name}/diagnosis.png"
        lines.append(f"![diagnosis]({png_rel})\n")

    lines.append("## Summary\n")
    counts: dict[str, int] = {}
    for d in diagnoses:
        counts[d.classification] = counts.get(d.classification, 0) + 1
    lines.append("Classification breakdown across trials:\n")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{name}`: {n}/{len(diagnoses)} trials\n")
    lines.append(
        "\nNext: run `python -m dimos.utils.characterization.processing.recommend`\n"
        "to map the dominant classification to a phase-2 action.\n"
    )

    with report_path.open("w") as fh:
        fh.write("\n".join(lines))
    return report_path


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def diagnose_session(session_dir: Path) -> Path:
    session_dir = Path(session_dir).expanduser().resolve()
    runs = load_session(session_dir)
    if not runs:
        raise ValueError(f"no trajectory runs found in {session_dir}")
    diagnoses: list[TrialDiagnosis] = []
    for run in runs:
        executed = build_executed_trajectory(run)
        score = score_run_with_trajectory(executed, run.anchored_ref_fn, duration_s=run.duration_s)
        diagnosis = classify(run, score)
        plot_run(run, diagnosis, run.run_dir)
        diagnoses.append(diagnosis)
        logger.info(
            "trial %s: class=%s (lag_rms=%.3f m, cross_rms=%.3f m)",
            run.run_id,
            diagnosis.classification,
            score.along_track_lag_rms,
            score.cross_track_traj_rms,
        )
    return make_report(session_dir, diagnoses)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Diagnose a trajectory-tracking session.")
    parser.add_argument("session_dir", type=Path, help="Path to a session_* directory.")
    args = parser.parse_args()
    report = diagnose_session(args.session_dir)
    print(f"report: {report}")


if __name__ == "__main__":
    main()
