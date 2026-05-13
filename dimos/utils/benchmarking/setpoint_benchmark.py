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

"""Single-pose setpoint benchmark with FOPDT plant + 3 noise modes.

Drives a "go to pose" controller against the fitted FOPDT plant
(:class:`Go2PlantSim`) under three noise regimes — *zero*, *Gaussian*,
*replay residuals* — and produces a precision-vs-time-to-arrive Pareto
curve as the user sweeps an aggressiveness knob.

The controller here (:class:`SetpointController`) is intentionally tiny;
real production controllers (RPP, PurePursuit, FF+PI) plug into the same
harness once the framework is validated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from dimos.utils.benchmarking.plant import (
    FopdtChannelParams,
    Go2PlantParams,
    Go2PlantSim,
)

# ---------------------------------------------------------------- JSON loader


def load_fitted_plant(
    summary_paths: Path | Sequence[Path],
    *,
    fallback: Go2PlantParams | None = None,
    pool_strategy: Literal["per_amp_median", "pooled_mean"] = "per_amp_median",
) -> Go2PlantParams:
    """Build :class:`Go2PlantParams` from one or more ``model_summary.json`` files.

    Each per-session summary contains exactly one channel (``vx``, ``vy``,
    or ``wz``). Pass paths for every channel you have characterized;
    missing channels fall back to ``fallback`` (typically
    ``GO2_PLANT_FITTED`` from :mod:`plant_models`).

    ``pool_strategy`` controls how (K, τ, L) are reduced to a single value:

    - ``"per_amp_median"`` (default) — median across the per-amplitude fits.
      Robust to outlier fits that drag the pooled mean toward zero.
    - ``"pooled_mean"`` — use ``channels[ch].pooled.{K,tau,L}.mean`` directly.
      Faithful to whatever the fitter wrote, but susceptible to the bug
      where degenerate τ→0 fits poison the pool.
    """
    if isinstance(summary_paths, (Path, str)):
        summary_paths = [Path(summary_paths)]
    found: dict[str, FopdtChannelParams] = {}
    for p in summary_paths:
        d = json.loads(Path(p).expanduser().read_text())
        for ch_name, ch in d["channels"].items():
            if pool_strategy == "pooled_mean":
                pooled = ch["pooled"]
                K = float(pooled["K"]["mean"])
                tau = float(pooled["tau"]["mean"])
                L = float(pooled["L"]["mean"])
            else:
                entries = ch.get("per_amplitude", [])
                if not entries:
                    pooled = ch["pooled"]
                    K = float(pooled["K"]["mean"])
                    tau = float(pooled["tau"]["mean"])
                    L = float(pooled["L"]["mean"])
                else:
                    K = float(np.median([e["K"]["mean"] for e in entries]))
                    tau = float(np.median([e["tau"]["mean"] for e in entries]))
                    L = float(np.median([e["L"]["mean"] for e in entries]))
            found[ch_name] = FopdtChannelParams(
                K=K,
                tau=max(1e-3, tau),
                L=max(0.0, L),
            )

    def _pick(ch: str) -> FopdtChannelParams:
        if ch in found:
            return found[ch]
        if fallback is not None:
            return getattr(fallback, ch)
        raise KeyError(f"No fit for channel {ch!r} in {summary_paths} and no fallback supplied.")

    return Go2PlantParams(vx=_pick("vx"), vy=_pick("vy"), wz=_pick("wz"))


# ---------------------------------------------------------------- noise types


class NoiseMode(str, Enum):
    NONE = "none"
    GAUSSIAN = "gaussian"
    REPLAY = "replay"


@dataclass
class NoiseConfig:
    mode: NoiseMode = NoiseMode.NONE
    sigma_vx: float = 0.0
    sigma_vy: float = 0.0
    sigma_wz: float = 0.0
    residual_vx: np.ndarray | None = None  # 1-D, dt-spaced
    residual_vy: np.ndarray | None = None
    residual_wz: np.ndarray | None = None
    seed: int = 0


class NoisyPlant:
    """Wraps :class:`Go2PlantSim` and injects per-tick velocity noise.

    The FOPDT lag+delay runs as usual. Noise is added to the *output*
    velocity (post-FOPDT, pre-integration) — that's the regime where
    process noise + sensor noise + leg-rocking show up in the loop a
    real controller has to live with.
    """

    def __init__(self, plant: Go2PlantSim, noise: NoiseConfig | None = None) -> None:
        self.plant = plant
        self.noise = noise or NoiseConfig()
        self._rng = np.random.default_rng(self.noise.seed)
        self._tick = 0

    def reset(self, x: float, y: float, yaw: float, dt: float) -> None:
        self.plant.reset(x, y, yaw, dt)
        self._rng = np.random.default_rng(self.noise.seed)
        self._tick = 0

    def step(self, cmd_vx: float, cmd_vy: float, cmd_wz: float, dt: float) -> None:
        # 1. FOPDT lag+delay (per channel)
        vx = self.plant.ch_vx.step(cmd_vx, dt)
        vy = self.plant.ch_vy.step(cmd_vy, dt)
        wz = self.plant.ch_wz.step(cmd_wz, dt)

        # 2. Noise on the output velocity
        if self.noise.mode == NoiseMode.GAUSSIAN:
            vx += self._rng.normal(0.0, self.noise.sigma_vx)
            vy += self._rng.normal(0.0, self.noise.sigma_vy)
            wz += self._rng.normal(0.0, self.noise.sigma_wz)
        elif self.noise.mode == NoiseMode.REPLAY:
            vx += _sample_at(self.noise.residual_vx, self._tick)
            vy += _sample_at(self.noise.residual_vy, self._tick)
            wz += _sample_at(self.noise.residual_wz, self._tick)

        self.plant.vx, self.plant.vy, self.plant.wz = vx, vy, wz

        # 3. Unicycle kinematic integration (world frame)
        cy, sy = math.cos(self.plant.yaw), math.sin(self.plant.yaw)
        self.plant.x += (vx * cy - vy * sy) * dt
        self.plant.y += (vx * sy + vy * cy) * dt
        self.plant.yaw = (self.plant.yaw + wz * dt + math.pi) % (2 * math.pi) - math.pi
        self._tick += 1

    @property
    def x(self) -> float:
        return self.plant.x

    @property
    def y(self) -> float:
        return self.plant.y

    @property
    def yaw(self) -> float:
        return self.plant.yaw

    @property
    def vx(self) -> float:
        return self.plant.vx

    @property
    def vy(self) -> float:
        return self.plant.vy

    @property
    def wz(self) -> float:
        return self.plant.wz


def _sample_at(arr: np.ndarray | None, idx: int) -> float:
    if arr is None or arr.size == 0:
        return 0.0
    return float(arr[idx % arr.size])


# ---------------------------------------------------------------- noise builders


def gaussian_from_pose_floor(
    run_dirs: Sequence[Path],
    *,
    pre_step_window_s: float = 0.5,
    margin_s: float = 0.05,
) -> NoiseConfig:
    """Estimate per-channel sigma from the pre-step "still" window of real runs.

    Reconstructs body velocity for each run, finds the time when the
    first non-zero command fires, then measures std-dev of velocity over
    the window before that. Averages sigma across runs.

    Returns a :class:`NoiseConfig` with ``mode=GAUSSIAN``.
    """
    from dimos.utils.characterization.scripts.analyze import (
        _reconstruct_or_empty,
        load_run,
    )

    sigs_vx, sigs_vy, sigs_wz = [], [], []
    for rd in run_dirs:
        try:
            run = load_run(Path(rd))
        except FileNotFoundError:
            continue
        if run.meas_ts_rel.size < 5 or run.cmd_ts_rel.size == 0:
            continue
        vx, vy, wz = _reconstruct_or_empty(run)
        if vx.size == 0:
            continue

        cmd_mag = np.abs(run.cmd_vx) + np.abs(run.cmd_vy) + np.abs(run.cmd_wz)
        nz = np.where(cmd_mag > 1e-6)[0]
        if nz.size == 0:
            continue
        step_t = float(run.cmd_ts_rel[nz[0]])

        meas_ts = np.sort(run.meas_ts_rel)
        keep = np.concatenate([[True], np.diff(meas_ts) > 0])
        meas_ts = meas_ts[keep]

        mask = (meas_ts < step_t - margin_s) & (meas_ts > step_t - pre_step_window_s - margin_s)
        if mask.sum() >= 3:
            sigs_vx.append(float(np.std(vx[mask])))
            sigs_vy.append(float(np.std(vy[mask])))
            sigs_wz.append(float(np.std(wz[mask])))

    if not sigs_vx:
        return NoiseConfig(mode=NoiseMode.GAUSSIAN, sigma_vx=0.05, sigma_vy=0.05, sigma_wz=0.1)
    return NoiseConfig(
        mode=NoiseMode.GAUSSIAN,
        sigma_vx=float(np.mean(sigs_vx)),
        sigma_vy=float(np.mean(sigs_vy)),
        sigma_wz=float(np.mean(sigs_wz)),
    )


def replay_from_run(
    run_dir: Path,
    plant_params: Go2PlantParams,
    *,
    dt: float = 0.05,
    demean: bool = True,
) -> NoiseConfig:
    """Build a REPLAY noise config from one real run.

    Pipeline:
      1. Load real cmd + measured pose, reconstruct measured body velocity.
      2. Resample cmd onto a uniform timeline at ``dt`` (zero-order hold).
      3. Resample measured velocity onto the same timeline (linear interp).
      4. Drive ``Go2PlantSim`` (no noise) with the resampled cmd at ``dt``.
      5. Residual = measured - predicted, replayable as a noise timeseries.

    With ``demean=True`` (default) the per-channel residual mean is
    subtracted, so the replay captures *noise structure* (leg-rocking,
    sensor jitter, comms-induced glitches) without any steady model-bias
    offset. Set ``demean=False`` to include model-mismatch bias in the
    "real mess" — that's the worst case a controller would face.
    """
    from dimos.utils.characterization.scripts.analyze import (
        _reconstruct_or_empty,
        load_run,
    )

    run = load_run(Path(run_dir))
    vx_meas, vy_meas, wz_meas = _reconstruct_or_empty(run)
    if vx_meas.size == 0:
        return NoiseConfig(mode=NoiseMode.REPLAY)

    meas_ts = np.sort(run.meas_ts_rel)
    keep = np.concatenate([[True], np.diff(meas_ts) > 0])
    meas_ts = meas_ts[keep]

    t0, t1 = float(meas_ts[0]), float(meas_ts[-1])
    n = max(2, int((t1 - t0) / dt) + 1)
    ts_u = t0 + np.arange(n) * dt

    vx_m = np.interp(ts_u, meas_ts, vx_meas)
    vy_m = np.interp(ts_u, meas_ts, vy_meas)
    wz_m = np.interp(ts_u, meas_ts, wz_meas)

    cmd_ts = run.cmd_ts_rel

    def _zoh(target: np.ndarray, src: np.ndarray, vals: np.ndarray) -> np.ndarray:
        out = np.zeros_like(target)
        for i, t in enumerate(target):
            j = int(np.searchsorted(src, t, side="right") - 1)
            out[i] = vals[j] if j >= 0 else 0.0
        return out

    cmd_vx_u = _zoh(ts_u, cmd_ts, run.cmd_vx)
    cmd_vy_u = _zoh(ts_u, cmd_ts, run.cmd_vy)
    cmd_wz_u = _zoh(ts_u, cmd_ts, run.cmd_wz)

    sim = Go2PlantSim(plant_params)
    sim.reset(0.0, 0.0, 0.0, dt)
    vx_p = np.zeros(n)
    vy_p = np.zeros(n)
    wz_p = np.zeros(n)
    for i in range(n):
        sim.step(cmd_vx_u[i], cmd_vy_u[i], cmd_wz_u[i], dt)
        vx_p[i], vy_p[i], wz_p[i] = sim.vx, sim.vy, sim.wz

    res_vx = vx_m - vx_p
    res_vy = vy_m - vy_p
    res_wz = wz_m - wz_p
    if demean:
        res_vx = res_vx - res_vx.mean()
        res_vy = res_vy - res_vy.mean()
        res_wz = res_wz - res_wz.mean()
    return NoiseConfig(
        mode=NoiseMode.REPLAY,
        residual_vx=res_vx,
        residual_vy=res_vy,
        residual_wz=res_wz,
    )


# ---------------------------------------------------------------- controller


@dataclass
class SetpointControllerConfig:
    """Knobs for the tiny "drive to pose" policy.

    For the Pareto sweep the natural aggressiveness knobs are
    ``max_vx`` (how fast we let it run) and ``kp_heading`` (how
    aggressively we steer toward the goal).
    """

    max_vx: float = 0.6
    max_wz: float = 1.5
    kp_heading: float = 1.5
    kp_distance: float = 1.0
    align_threshold_rad: float = 0.4
    settle_xy: float = 0.05
    settle_yaw: float = 0.05


class SetpointController:
    """Two-phase policy: align→drive in xy, then rotate to goal yaw."""

    def __init__(self, cfg: SetpointControllerConfig | None = None) -> None:
        self.cfg = cfg or SetpointControllerConfig()
        self.gx = self.gy = self.gyaw = 0.0

    def reset(self, gx: float, gy: float, gyaw: float) -> None:
        self.gx, self.gy, self.gyaw = gx, gy, gyaw

    def compute(self, x: float, y: float, yaw: float) -> tuple[float, float, float]:
        cfg = self.cfg
        dx, dy = self.gx - x, self.gy - y
        dist = math.hypot(dx, dy)

        if dist > cfg.settle_xy:
            heading_to_goal = math.atan2(dy, dx)
            heading_err = _wrap(heading_to_goal - yaw)
            cmd_wz = _clip(cfg.kp_heading * heading_err, cfg.max_wz)
            if abs(heading_err) > cfg.align_threshold_rad:
                cmd_vx = 0.0
            else:
                align = max(0.0, math.cos(heading_err))
                cmd_vx = max(0.0, min(cfg.max_vx, cfg.kp_distance * dist * align))
            return (cmd_vx, 0.0, cmd_wz)

        yaw_err = _wrap(self.gyaw - yaw)
        return (0.0, 0.0, _clip(cfg.kp_heading * yaw_err, cfg.max_wz))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _clip(v: float, mag: float) -> float:
    return max(-mag, min(mag, v))


# ---------------------------------------------------------------- generic Controller


class Controller(Protocol):
    """Minimal contract the harness needs from any controller.

    The tiny :class:`SetpointController` and the :class:`PathFollowerAdapter`
    (wrapping prod path-followers like RPP / PurePursuit) both conform to it.
    """

    def reset(
        self,
        start_pose: tuple[float, float, float],
        goal_pose: tuple[float, float, float],
        dt: float,
    ) -> None: ...

    def compute(self, x: float, y: float, yaw: float) -> tuple[float, float, float]: ...


def _setpoint_reset(self, start_pose, goal_pose, dt):
    self.gx, self.gy, self.gyaw = goal_pose


# Patch SetpointController's reset to accept the (start_pose, goal_pose, dt)
# protocol; the original signature was just (gx, gy, gyaw).
SetpointController.reset = _setpoint_reset  # type: ignore[assignment,method-assign]


# --------- helpers to build PoseStamped / Path the prod controllers expect


def _make_pose_stamped(x: float, y: float, yaw: float):
    """Build a PoseStamped at (x, y, yaw) — z=0, no roll/pitch.

    Yaw is encoded as a unit quaternion (z-axis rotation): qz=sin(yaw/2),
    qw=cos(yaw/2).
    """
    from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped

    qz = math.sin(yaw / 2)
    qw = math.cos(yaw / 2)
    return PoseStamped(position=[x, y, 0.0], orientation=[0.0, 0.0, qz, qw])


def make_straight_path(
    start_pose: tuple[float, float, float],
    goal_pose: tuple[float, float, float],
    *,
    n: int = 20,
):
    """Linearly interpolate ``n`` poses between start and goal.

    Yaw is set to the line direction at every intermediate pose, then the
    goal yaw is applied to the last pose. Path-followers need ≥ 2 poses
    and benefit from a denser path (gives lookahead more to bite on).
    """
    from dimos.msgs.nav_msgs.Path import Path as RosPath

    sx, sy, syaw = start_pose
    gx, gy, gyaw = goal_pose
    line_yaw = math.atan2(gy - sy, gx - sx)
    poses = []
    for i in range(n):
        a = i / (n - 1)
        x = sx + a * (gx - sx)
        y = sy + a * (gy - sy)
        # First pose carries start yaw, last carries goal yaw, others line dir.
        if i == 0:
            yaw = syaw
        elif i == n - 1:
            yaw = gyaw
        else:
            yaw = line_yaw
        poses.append(_make_pose_stamped(x, y, yaw))
    return RosPath(poses=poses)


class PathFollowerAdapter:
    """Wrap a prod path-following ControlTask (PP, RPP, …) as a Controller.

    Converts a single-pose goal into a dense straight path, feeds the
    task ``update_odom()`` + a minimal ``CoordinatorState`` each tick,
    and unpacks the ``JointCommandOutput.velocities`` into the
    ``(vx, vy, wz)`` triple our harness expects.
    """

    def __init__(self, task, *, n_path_points: int = 20) -> None:
        self.task = task
        self.n_path_points = n_path_points
        self._t = 0.0
        self._dt = 0.1

    def reset(self, start_pose, goal_pose, dt):
        self._t = 0.0
        self._dt = dt
        # Some tasks have a `reset()` for their internal state (e.g. PID).
        if hasattr(self.task, "reset"):
            try:
                self.task.reset()
            except Exception:
                pass
        path = make_straight_path(start_pose, goal_pose, n=self.n_path_points)
        odom = _make_pose_stamped(*start_pose)
        self.task.start_path(path, odom)

    def compute(self, x, y, yaw):
        from dimos.control.task import CoordinatorState, JointStateSnapshot

        odom = _make_pose_stamped(x, y, yaw)
        self.task.update_odom(odom)
        self._t += self._dt
        state = CoordinatorState(
            joints=JointStateSnapshot(),
            t_now=self._t,
            dt=self._dt,
        )
        out = self.task.compute(state)
        if out is None or out.velocities is None:
            return (0.0, 0.0, 0.0)
        v = out.velocities
        return (float(v[0]), float(v[1]), float(v[2]))


# ---------------------------------------------------------------- harness


@dataclass
class EpisodeResult:
    ts: np.ndarray  # (N,)
    pose: np.ndarray  # (N, 3) x, y, yaw
    cmd: np.ndarray  # (N, 3) cmd_vx, cmd_vy, cmd_wz
    vel_actual: np.ndarray  # (N, 3) vx, vy, wz (post-noise)
    goal: tuple[float, float, float]


def run_setpoint_episode(
    plant: NoisyPlant,
    controller: Controller,
    *,
    start_pose: tuple[float, float, float],
    goal_pose: tuple[float, float, float],
    dt: float = 0.1,
    t_max: float = 15.0,
) -> EpisodeResult:
    plant.reset(start_pose[0], start_pose[1], start_pose[2], dt)
    controller.reset(start_pose, goal_pose, dt)

    n = round(t_max / dt) + 1
    ts = np.zeros(n)
    pose = np.zeros((n, 3))
    cmd = np.zeros((n, 3))
    vel = np.zeros((n, 3))

    for i in range(n):
        ts[i] = i * dt
        pose[i] = (plant.x, plant.y, plant.yaw)
        cvx, cvy, cwz = controller.compute(plant.x, plant.y, plant.yaw)
        cmd[i] = (cvx, cvy, cwz)
        plant.step(cvx, cvy, cwz, dt)
        vel[i] = (plant.vx, plant.vy, plant.wz)

    return EpisodeResult(ts=ts, pose=pose, cmd=cmd, vel_actual=vel, goal=goal_pose)


@dataclass
class SetpointScore:
    time_to_arrive: float  # NaN if never settles for hold_s
    terminal_xy_err: float  # at episode end
    terminal_yaw_err: float
    peak_xy_err_after_settle: float  # NaN if never settled


# ---------------------------------------------------------------- BaseControlTask wrapper


class SetpointControlTask:
    """Wrap :class:`SetpointController` as a path-follower-shaped ControlTask.

    The hw + sim battery runners (`_run_path_follower_{hw,sim}`) speak the
    `start_path / update_odom / compute(state) / get_state` protocol from
    [_PathFollowerLike](runner.py:89). This adapter converts a `Path` into a
    single goal pose (the path's last pose), then runs the existing
    pose-PID setpoint policy each tick.

    State machine: ``idle → following → arrived | aborted``.
    """

    def __init__(
        self,
        name: str = "setpoint_follower",
        setpoint_config: SetpointControllerConfig | None = None,
        joint_names: Sequence[str] | None = None,
        priority: int = 20,
    ) -> None:
        self._name = name
        self._setpoint = SetpointController(setpoint_config or SetpointControllerConfig())
        self._joint_names_list = (
            list(joint_names) if joint_names else ["go2_vx", "go2_vy", "go2_wz"]
        )
        self._joint_names = frozenset(self._joint_names_list)
        self._priority = priority
        self._state: str = "idle"
        self._current_odom = None
        self._goal: tuple[float, float, float] | None = None

    @property
    def name(self) -> str:
        return self._name

    def claim(self):
        from dimos.control.task import ControlMode, ResourceClaim

        return ResourceClaim(
            joints=self._joint_names,
            priority=self._priority,
            mode=ControlMode.VELOCITY,
        )

    def is_active(self) -> bool:
        return self._state == "following"

    def get_state(self) -> str:
        return self._state

    def start_path(self, path, current_odom) -> bool:
        if not path or not path.poses:
            return False
        last = path.poses[-1]
        gx = float(last.position.x)
        gy = float(last.position.y)
        gyaw = float(last.orientation.euler[2])
        self._goal = (gx, gy, gyaw)
        sx = float(current_odom.position.x)
        sy = float(current_odom.position.y)
        syaw = float(current_odom.orientation.euler[2])
        self._setpoint.reset((sx, sy, syaw), self._goal, dt=0.1)
        self._current_odom = current_odom
        self._state = "following"
        return True

    def update_odom(self, odom) -> None:
        self._current_odom = odom

    def compute(self, state):
        from dimos.control.task import ControlMode, JointCommandOutput

        if not self.is_active() or self._current_odom is None or self._goal is None:
            return None
        x = float(self._current_odom.position.x)
        y = float(self._current_odom.position.y)
        yaw = float(self._current_odom.orientation.euler[2])
        gx, gy, gyaw = self._goal
        xy_err = math.hypot(gx - x, gy - y)
        yaw_err = abs(_wrap(gyaw - yaw))
        if xy_err < self._setpoint.cfg.settle_xy and yaw_err < self._setpoint.cfg.settle_yaw:
            self._state = "arrived"
            return JointCommandOutput(
                joint_names=self._joint_names_list,
                velocities=[0.0, 0.0, 0.0],
                mode=ControlMode.VELOCITY,
            )
        cvx, cvy, cwz = self._setpoint.compute(x, y, yaw)
        return JointCommandOutput(
            joint_names=self._joint_names_list,
            velocities=[cvx, cvy, cwz],
            mode=ControlMode.VELOCITY,
        )

    def cancel(self) -> bool:
        if self.is_active():
            self._state = "aborted"
            return True
        return False

    def reset(self) -> bool:
        self._state = "idle"
        self._current_odom = None
        self._goal = None
        return True

    def on_preempted(self, by_task: str, joints) -> None:
        if joints & self._joint_names and self.is_active():
            self._state = "aborted"

    # ----- ControlTask Protocol no-op callbacks (we don't use any of these) -----

    def on_buttons(self, msg) -> bool:
        return False

    def on_cartesian_command(self, pose, t_now: float) -> bool:
        return False

    def set_target_by_name(self, positions, t_now: float) -> bool:
        return False

    def set_velocities_by_name(self, velocities, t_now: float) -> bool:
        return False


# ---------------------------------------------------------------- scoring


def score_setpoint_episode(
    result: EpisodeResult,
    *,
    eps_xy: float = 0.05,
    eps_yaw: float = 0.05,
    hold_s: float = 0.5,
) -> SetpointScore:
    gx, gy, gyaw = result.goal
    xy_err = np.hypot(result.pose[:, 0] - gx, result.pose[:, 1] - gy)
    yaw_err = np.abs((result.pose[:, 2] - gyaw + math.pi) % (2 * math.pi) - math.pi)

    dt = float(result.ts[1] - result.ts[0]) if result.ts.size > 1 else 0.1
    hold_n = max(1, round(hold_s / dt))
    settled = (xy_err < eps_xy) & (yaw_err < eps_yaw)

    time_to_arrive = float("nan")
    settle_idx: int | None = None
    for i in range(len(settled) - hold_n + 1):
        if settled[i : i + hold_n].all():
            time_to_arrive = float(result.ts[i])
            settle_idx = i
            break

    peak_after = float("nan")
    if settle_idx is not None:
        peak_after = float(np.max(xy_err[settle_idx:]))

    return SetpointScore(
        time_to_arrive=time_to_arrive,
        terminal_xy_err=float(xy_err[-1]),
        terminal_yaw_err=float(yaw_err[-1]),
        peak_xy_err_after_settle=peak_after,
    )
