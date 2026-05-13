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

"""CLI: analyze characterization runs.

Two subcommands::

    python -m dimos.utils.characterization.scripts.analyze run     <run_dir> [--out plot.svg]
    python -m dimos.utils.characterization.scripts.analyze compare <run_dir> ... [--out compare.svg] [--channel auto|vx|vy|wz]

Velocity reconstruction contract
--------------------------------
The Go2 odometry stream gives us world-frame ``(x, y, yaw)`` but NOT
body-frame ``(vx, vy, wz)``. Every analysis uses the same reconstruction:

    1. Savitzky-Golay filter on ``(x, y, yaw_unwrapped)`` (window 5, order 2).
    2. Central difference on the filtered series.
    3. Rotate ``(dx/dt, dy/dt)`` by ``-yaw`` to get body-frame ``(vx, vy)``.

Implemented here as ``reconstruct_body_velocities`` and reused across
the processing layer. Do not re-implement.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- velocity reconstruction


def _hampel_filter(values: np.ndarray, window: int = 11, n_sigma: float = 3.0) -> np.ndarray:
    """Rolling-median + MAD outlier rejection (Hampel filter).

    For each sample, compute the median over a centered window and the
    median absolute deviation. If ``|x_i - median| > n_sigma * 1.4826 *
    MAD``, replace ``x_i`` with the local median. The ``1.4826`` factor
    converts MAD to a Gaussian-sigma-equivalent.

    Designed for legged-robot pose data where leg impacts inject occasional
    one- or two-sample spikes that downstream Savitzky-Golay smoothing
    would otherwise smear into a wider artifact.

    Returns a copy; original is not modified. ``window`` should be odd.
    """
    n = len(values)
    if n < window or window < 3:
        return values
    half = window // 2
    out = values.copy()
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        win = values[lo:hi]
        med = np.median(win)
        mad = np.median(np.abs(win - med))
        if mad == 0:
            continue
        if abs(values[i] - med) > n_sigma * 1.4826 * mad:
            out[i] = med
    return out


def reconstruct_body_velocities(
    ts: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    yaw: np.ndarray,
    *,
    window: int = 5,
    order: int = 2,
    hampel_window: int = 11,
    hampel_n_sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return body-frame ``(vx, vy, wz)`` sampled at the odom timestamps.

    Pipeline (legged-robot tuned):
      1. Savitzky-Golay smoothing on (x, y, yaw_unwrapped) — preserves
         macroscopic motion, attenuates high-freq noise.
      2. Central difference (np.gradient, non-uniform-ts safe).
      3. Rotate world-frame (dx/dt, dy/dt) into body-frame (vx, vy).
      4. Hampel filter on the *velocity* — rejects leg-impact spikes.
         Tiny mm-scale pose jitter → huge m/s velocity impulse after
         differentiation; Hampel on the derivative catches them while
         pose-level Hampel can't.

    ``ts`` must be strictly increasing. For fewer than ``window`` samples
    the polynomial smoother is bypassed but the Hampel pass still runs.

    Pass ``hampel_n_sigma=float('inf')`` to disable outlier rejection.
    """
    from scipy.signal import savgol_filter

    yaw_u = np.unwrap(yaw)

    # 1. Polynomial smoothing on pose
    if len(ts) >= window and window % 2 == 1 and order < window:
        xf = savgol_filter(x, window, order)
        yf = savgol_filter(y, window, order)
        yf_yaw = savgol_filter(yaw_u, window, order)
    else:
        xf, yf, yf_yaw = x, y, yaw_u

    # 2. Central difference
    dx = np.gradient(xf, ts)
    dy = np.gradient(yf, ts)
    dyaw = np.gradient(yf_yaw, ts)

    # 3. World → body
    cos_y = np.cos(yf_yaw)
    sin_y = np.sin(yf_yaw)
    vx = cos_y * dx + sin_y * dy
    vy = -sin_y * dx + cos_y * dy

    # 4. Spike rejection on the velocity (where leg impacts actually appear)
    vx = _hampel_filter(vx, hampel_window, hampel_n_sigma)
    vy = _hampel_filter(vy, hampel_window, hampel_n_sigma)
    dyaw = _hampel_filter(dyaw, hampel_window, hampel_n_sigma)

    return vx, vy, dyaw


# ---------------------------------------------------------------------- run loading


class LoadedRun:
    def __init__(
        self,
        run_dir: Path,
        metadata: dict[str, Any],
        cmd_ts_mono: np.ndarray,
        cmd_vx: np.ndarray,
        cmd_vy: np.ndarray,
        cmd_wz: np.ndarray,
        meas_ts_wall: np.ndarray,
        meas_x: np.ndarray,
        meas_y: np.ndarray,
        meas_yaw: np.ndarray,
    ) -> None:
        self.run_dir = run_dir
        self.metadata = metadata
        self.cmd_ts_mono = cmd_ts_mono
        self.cmd_vx = cmd_vx
        self.cmd_vy = cmd_vy
        self.cmd_wz = cmd_wz
        self.meas_ts_wall = meas_ts_wall
        self.meas_x = meas_x
        self.meas_y = meas_y
        self.meas_yaw = meas_yaw

    @property
    def test_type(self) -> str:
        return self.metadata["recipe"]["test_type"]

    @property
    def name(self) -> str:
        return self.metadata["recipe"]["name"]

    @property
    def clock_anchor_mono(self) -> float:
        return float(self.metadata["clock_anchor"]["monotonic"])

    @property
    def clock_anchor_wall(self) -> float:
        return float(self.metadata["clock_anchor"]["wall"])

    @property
    def cmd_ts_rel(self) -> np.ndarray:
        return self.cmd_ts_mono - self.clock_anchor_mono

    @property
    def meas_ts_rel(self) -> np.ndarray:
        # Rebase wall-clock measured ts onto the monotonic run-start.
        # Wall and monotonic are aligned once at start; subtract the wall anchor.
        return self.meas_ts_wall - self.clock_anchor_wall


def load_run(run_dir: Path) -> LoadedRun:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"not a directory: {run_dir}")

    run_json_path = run_dir / "run.json"
    cmd_jsonl_path = run_dir / "cmd_monotonic.jsonl"
    db_path = run_dir / "recording.db"

    if not run_json_path.exists():
        raise FileNotFoundError(
            f"{run_dir} is missing run.json — not a valid characterization run directory. "
            "If this dir came from an older run that crashed mid-setup, delete it."
        )

    with run_json_path.open() as fh:
        metadata = json.load(fh)

    cmd_ts_mono: list[float] = []
    cmd_vx: list[float] = []
    cmd_vy: list[float] = []
    cmd_wz: list[float] = []
    with cmd_jsonl_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cmd_ts_mono.append(d["tx_mono"])
            cmd_vx.append(d["vx"])
            cmd_vy.append(d["vy"])
            cmd_wz.append(d["wz"])

    meas_ts_wall: list[float] = []
    meas_x: list[float] = []
    meas_y: list[float] = []
    meas_yaw: list[float] = []

    # Prefer session-level DB when run.json points at one; fall back to
    # a per-run recording.db for single-run mode.
    session_db_rel = metadata.get("session_db_path")
    ts_window = metadata.get("ts_window_wall") or {}
    effective_db: Path | None = None
    if session_db_rel:
        candidate = (run_dir / session_db_rel).resolve()
        if candidate.exists():
            effective_db = candidate
        else:
            logger.warning(
                "session_db_path %s not found (resolved %s); falling back to %s",
                session_db_rel,
                candidate,
                db_path,
            )
    if effective_db is None and db_path.exists():
        effective_db = db_path

    if effective_db is not None:
        try:
            from dimos.memory2.store.sqlite import SqliteStore
            from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped

            store = SqliteStore(path=str(effective_db))
            store.start()
            try:
                measured = store.stream("measured", PoseStamped)
                ts_lo = ts_window.get("start") if ts_window else None
                ts_hi = ts_window.get("end") if ts_window else None
                for obs in measured.to_list():
                    t = float(obs.ts)
                    # Slice to the run's wall-clock window when we're reading
                    # a session-level DB; otherwise keep everything.
                    if session_db_rel:
                        if ts_lo is not None and t < ts_lo:
                            continue
                        if ts_hi is not None and t > ts_hi:
                            continue
                    meas_ts_wall.append(t)
                    p = obs.data
                    meas_x.append(float(p.x))
                    meas_y.append(float(p.y))
                    meas_yaw.append(float(p.yaw))
            finally:
                store.stop()
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to load measured stream from %s: %s", effective_db, e)

    return LoadedRun(
        run_dir=run_dir,
        metadata=metadata,
        cmd_ts_mono=np.asarray(cmd_ts_mono, dtype=float),
        cmd_vx=np.asarray(cmd_vx, dtype=float),
        cmd_vy=np.asarray(cmd_vy, dtype=float),
        cmd_wz=np.asarray(cmd_wz, dtype=float),
        meas_ts_wall=np.asarray(meas_ts_wall, dtype=float),
        meas_x=np.asarray(meas_x, dtype=float),
        meas_y=np.asarray(meas_y, dtype=float),
        meas_yaw=np.asarray(meas_yaw, dtype=float),
    )


# ---------------------------------------------------------------------- helpers used by modeling


def _reconstruct_or_empty(run: LoadedRun) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Body vx/vy/wz aligned to ``run.meas_ts_wall`` — empty arrays if no odom."""
    if run.meas_ts_wall.size < 3:
        empty = np.zeros(0, dtype=float)
        return empty, empty, empty
    ts = run.meas_ts_wall
    order = np.argsort(ts, kind="stable")
    ts_s = ts[order]
    keep = np.concatenate([[True], np.diff(ts_s) > 0])
    return reconstruct_body_velocities(
        ts_s[keep], run.meas_x[order][keep], run.meas_y[order][keep], run.meas_yaw[order][keep]
    )


def _channel_arrays(run: LoadedRun, channel: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(cmd_array, meas_array)`` for the requested channel."""
    vx_meas, vy_meas, wz_meas = _reconstruct_or_empty(run)
    if channel == "vx":
        return run.cmd_vx, vx_meas
    if channel == "vy":
        return run.cmd_vy, vy_meas
    return run.cmd_wz, wz_meas


def _dominant_channel(run: LoadedRun) -> str:
    """Pick the channel with the largest commanded amplitude — so an E2
    wz-step run is recognised as wz, not vx.
    """
    amps = {
        "vx": float(np.max(np.abs(run.cmd_vx))) if run.cmd_vx.size else 0.0,
        "vy": float(np.max(np.abs(run.cmd_vy))) if run.cmd_vy.size else 0.0,
        "wz": float(np.max(np.abs(run.cmd_wz))) if run.cmd_wz.size else 0.0,
    }
    return max(amps, key=lambda k: amps[k]) if any(amps.values()) else "vx"


# ---------------------------------------------------------------------- step metrics


def step_metrics(
    ts: np.ndarray,
    meas: np.ndarray,
    *,
    step_t: float,
    target: float,
    settle_band_frac: float = 0.02,
    active_end_t: float | None = None,
) -> dict[str, float | None]:
    """Rise time (10-90%), settle time (±band around steady), overshoot peak.

    Returns ``None`` fields when the trace didn't reach a value (e.g. odom
    never caught up). ``target`` is the commanded steady-state.

    ``active_end_t`` (optional): wall/relative time when the recipe's
    active window ends and post-roll zeros begin. When provided, the
    steady-state estimate is computed within ``[step_t, active_end_t]``,
    which excludes post-roll deceleration. Without it, the metric used
    to look at last-10% of *all* post-step samples — which silently
    averaged in the post-roll stop and produced badly biased
    steady-state values for short active windows.
    """
    if ts.size == 0:
        return {"rise_10_90_s": None, "settle_s": None, "overshoot": None, "steady_state": None}

    mask_post = ts >= step_t
    if active_end_t is not None:
        mask_post = mask_post & (ts <= active_end_t)
    if not mask_post.any():
        return {"rise_10_90_s": None, "settle_s": None, "overshoot": None, "steady_state": None}

    # Steady-state estimate: mean of the last 10% of samples in the active window.
    post_ts = ts[mask_post]
    post_meas = meas[mask_post]
    n = len(post_ts)
    ss = float(np.mean(post_meas[max(0, int(n * 0.9)) :])) if n else float("nan")

    # Rise time 10-90% of (ss - pre_value). Use 0 as pre if step_t is at run start.
    pre_mask = ts < step_t
    pre_val = float(np.mean(meas[pre_mask])) if pre_mask.any() else 0.0
    span = ss - pre_val
    rise_10_90: float | None = None
    if abs(span) > 1e-6:
        low = pre_val + 0.1 * span
        high = pre_val + 0.9 * span
        t_low = _first_cross(post_ts, post_meas, low, ascending=span > 0)
        t_high = _first_cross(post_ts, post_meas, high, ascending=span > 0)
        if t_low is not None and t_high is not None:
            rise_10_90 = t_high - t_low

    # Settle: last time meas leaves ±band*target around steady-state.
    band = settle_band_frac * max(abs(target), 1e-6)
    if post_ts.size:
        outside = np.abs(post_meas - ss) > band
        if outside.any():
            last_out_i = int(np.where(outside)[0][-1])
            if last_out_i + 1 < post_ts.size:
                settle_s = float(post_ts[last_out_i + 1] - step_t)
            else:
                settle_s = None  # never settled inside the window
        else:
            settle_s = 0.0
    else:
        settle_s = None

    overshoot: float | None = None
    if abs(span) > 1e-6:
        peak = float(np.max(post_meas) if span > 0 else np.min(post_meas))
        overshoot = (peak - ss) / span if span > 0 else (ss - peak) / (-span)

    return {
        "rise_10_90_s": rise_10_90,
        "settle_s": settle_s,
        "overshoot": overshoot,
        "steady_state": ss,
    }


def _first_cross(
    ts: np.ndarray, y: np.ndarray, threshold: float, *, ascending: bool
) -> float | None:
    cond = (y >= threshold) if ascending else (y <= threshold)
    idx = np.argmax(cond)
    if not cond[idx]:
        return None
    return float(ts[idx])


# ---------------------------------------------------------------------- CLI


def _main_run(args: argparse.Namespace) -> int:
    """``analyze run <run_dir>``: render canonical plot + persist metrics.json."""
    from dimos.utils.characterization.plotting import render_run

    run_dir = Path(args.run_dir)
    run = load_run(run_dir)
    out_path = Path(args.out) if args.out else (run_dir / "plot.svg")

    # Odom tripwire — warn if the measured stream is suspiciously short.
    expected_odom_hz = 10.0  # real Go2 lower bound; mujoco is higher, so this is conservative
    duration_s = (
        float(run.metadata["recipe"]["duration_s"])
        + float(run.metadata["recipe"].get("pre_roll_s", 0.0))
        + float(run.metadata["recipe"].get("post_roll_s", 0.0))
    )
    min_expected = duration_s * expected_odom_hz * 0.5
    if run.meas_ts_wall.size < min_expected:
        print(
            f"WARNING: measured stream has {run.meas_ts_wall.size} samples; "
            f"expected ≥ {min_expected:.0f} for a {duration_s:.1f}s run at ≥{expected_odom_hz:.0f}Hz. "
            "Odom may have been silent — check the coordinator.",
            file=sys.stderr,
        )

    metrics = render_run(run, out_path)

    print(f"run:     {run.name}  (type={run.test_type})")
    print(f"cmd:     {run.cmd_ts_mono.size} samples")
    print(f"meas:    {run.meas_ts_wall.size} samples")
    if metrics:
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    print(f"plot:    {out_path}")

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    return 0


def _main_compare(args: argparse.Namespace) -> int:
    """``analyze compare <run_dir> ...``: overlay cmd/meas across runs."""
    from dimos.utils.characterization.plotting import render_overlay

    runs = [load_run(Path(p)) for p in args.run_dirs]
    types = {r.test_type for r in runs}
    if len(types) != 1:
        print(
            f"ERROR: runs have different test_types: {types}. Cannot compare.",
            file=sys.stderr,
        )
        return 2

    channel = None if args.channel == "auto" else args.channel
    out_path = Path(args.out).expanduser().resolve()
    render_overlay(runs, out_path, channel=channel)
    print(f"compared {len(runs)} runs → {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze characterization runs (per-run plot or multi-run overlay)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Render the canonical per-run plot.")
    p_run.add_argument("run_dir", help="Path to a run directory produced by run_session")
    p_run.add_argument("--out", default=None, help="Output SVG path (default: <run_dir>/plot.svg)")
    p_run.add_argument("-v", "--verbose", action="store_true")

    p_cmp = sub.add_parser("compare", help="Overlay cmd/meas traces of multiple runs.")
    p_cmp.add_argument("run_dirs", nargs="+", help="Two or more run directories.")
    p_cmp.add_argument(
        "--out", default="compare.svg", help="Output SVG path (default: compare.svg)"
    )
    p_cmp.add_argument(
        "--channel",
        choices=("vx", "vy", "wz", "auto"),
        default="auto",
        help="Body-frame channel to compare. 'auto' picks the dominant commanded channel from the first run.",
    )
    p_cmp.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.cmd == "run":
        return _main_run(args)
    if args.cmd == "compare":
        return _main_compare(args)
    parser.print_help()
    return 2


__all__ = [
    "LoadedRun",
    "load_run",
    "main",
    "reconstruct_body_velocities",
    "step_metrics",
]


if __name__ == "__main__":
    sys.exit(main())
