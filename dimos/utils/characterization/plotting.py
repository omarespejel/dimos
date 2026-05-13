# Copyright 2026 Dimensional Inc.
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

# Copyright 2025-2026 Dimensional Inc.
# Licensed under the Apache License, Version 2.0.

"""SVG renderers for characterization runs, all routed through memory2.Plot.

Two entry points:

  - ``render_run(run, out_path)``    → per-run plot for one ``LoadedRun``
  - ``render_overlay(runs, out_path, channel)`` → multi-run cmd/meas overlay

Internally each builds a ``dimos.memory2.vis.plot.Plot`` and writes its
``to_svg()`` result. memory2 owns the rendering primitives (axes,
legend, ticks); this module owns dispatch by ``test_type``, channel
selection, and the shared color palette.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from dimos.utils.characterization.scripts.analyze import LoadedRun


# Color palette for compare/overlay plots (cycled by run index).
_PALETTE = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
)

# Per-channel display info for single-run plots.
# `cmd` = saturated color for the commanded trace; `meas` = lighter for measured.
_CHANNEL_INFO: dict[str, dict[str, str]] = {
    "vx": {"unit": "m/s", "cmd": "#1f77b4", "meas": "#aec7e8"},
    "vy": {"unit": "m/s", "cmd": "#2ca02c", "meas": "#98df8a"},
    "wz": {"unit": "rad/s", "cmd": "#ff7f0e", "meas": "#ffbb78"},
}


# ---------------------------------------------------------------------- public API


def render_run(run: LoadedRun, out_path: Path) -> dict[str, Any]:
    """Render the canonical per-run plot to ``out_path``; return a metrics dict.

    Dispatches by ``run.test_type``:
      - step / chirp → cmd vs meas overlay on the dominant channel + step metrics
      - ramp         → cmd vs meas with cmd markers
      - constant / composite → all three commanded channels overlaid with meas
    """
    tt = run.test_type
    if tt == "step" or tt == "chirp":
        svg, metrics = _render_step(run)
    elif tt == "ramp":
        svg, metrics = _render_ramp(run)
    elif tt in ("constant", "composite"):
        svg, metrics = _render_constant(run)
    else:
        raise ValueError(f"unknown test_type: {tt}")
    Path(out_path).write_text(svg)
    return metrics


def render_overlay(runs: list[LoadedRun], out_path: Path, *, channel: str | None = None) -> str:
    """Overlay cmd/meas traces of multiple runs on one channel.

    ``channel`` defaults to the dominant commanded channel of the first
    run. Each run gets a unique color from the palette; commanded traces
    are dashed, measured are solid, so a glance shows tracking.
    """
    from dimos.memory2.vis.plot.elements import Series, Style
    from dimos.memory2.vis.plot.plot import Plot, TimeAxis

    if not runs:
        raise ValueError("render_overlay requires at least one run")

    ch = channel or _dominant_channel(runs[0])
    unit = _CHANNEL_INFO[ch]["unit"]

    plot = Plot(time_axis=TimeAxis.raw)
    for i, run in enumerate(runs):
        color = _PALETTE[i % len(_PALETTE)]
        cmd_arr, meas_arr = _channel_arrays(run, ch)
        plot.add(
            Series(
                ts=run.cmd_ts_rel.tolist(),
                values=cmd_arr.tolist(),
                label=f"{run.name}: cmd_{ch} [{unit}]",
                color=color,
                style=Style.dashed,
            )
        )
        if run.meas_ts_rel.size:
            plot.add(
                Series(
                    ts=run.meas_ts_rel.tolist(),
                    values=meas_arr.tolist(),
                    label=f"{run.name}: meas_{ch} [{unit}]",
                    color=color,
                )
            )
    svg = plot.to_svg()
    Path(out_path).write_text(svg)
    return svg


# ---------------------------------------------------------------------- per-test renderers


def _render_step(run: LoadedRun) -> tuple[str, dict[str, Any]]:
    from dimos.memory2.vis.plot.elements import HLine, Series, VLine
    from dimos.memory2.vis.plot.plot import Plot, TimeAxis
    from dimos.utils.characterization.scripts.analyze import step_metrics

    channel = _dominant_channel(run)
    cmd_arr, meas_raw, meas_arr = _channel_arrays_dual(run, channel)
    info = _CHANNEL_INFO[channel]
    meas_ts = run.meas_ts_rel

    plot = Plot(time_axis=TimeAxis.raw)
    plot.add(
        Series(
            ts=run.cmd_ts_rel.tolist(),
            values=cmd_arr.tolist(),
            label=f"cmd_{channel} [{info['unit']}]",
            color=info["cmd"],
        )
    )
    if meas_ts.size:
        # Raw (no Hampel) — light gray, so you can see what the filter caught
        plot.add(
            Series(
                ts=meas_ts.tolist(),
                values=meas_raw.tolist(),
                label=f"meas_{channel} raw",
                color="#bbbbbb",
            )
        )
        # Hampel-filtered — main color (this is what the FOPDT fit uses)
        plot.add(
            Series(
                ts=meas_ts.tolist(),
                values=meas_arr.tolist(),
                label=f"meas_{channel} (Hampel) [{info['unit']}]",
                color=info["meas"],
            )
        )

    target = float(np.max(np.abs(cmd_arr))) if cmd_arr.size else 0.0
    nonzero = np.flatnonzero(np.abs(cmd_arr) > 1e-6)
    step_t = float(run.cmd_ts_rel[nonzero[0]]) if nonzero.size else 0.0
    active_end_t = step_t + float(run.metadata["recipe"]["duration_s"])

    n_replaced = int(np.sum(np.abs(meas_raw - meas_arr) > 1e-9)) if meas_ts.size else 0
    metrics: dict[str, Any] = {
        "channel": channel,
        "step_t": step_t,
        "target": target,
        "active_end_t": active_end_t,
        "hampel_replaced": n_replaced,
        "hampel_total": int(meas_arr.size),
    }
    if meas_ts.size >= 3:
        m = step_metrics(meas_ts, meas_arr, step_t=step_t, target=target, active_end_t=active_end_t)
        metrics.update(m)
        if m["steady_state"] is not None:
            plot.add(HLine(y=float(m["steady_state"]), label="steady", color="#888888"))
        plot.add(VLine(x=step_t, label="step", color="#aaa"))

    return plot.to_svg(), metrics


def _render_ramp(run: LoadedRun) -> tuple[str, dict[str, Any]]:
    from dimos.memory2.vis.plot.elements import Markers, Series
    from dimos.memory2.vis.plot.plot import Plot, TimeAxis

    channel = _dominant_channel(run)
    cmd_arr, meas_arr = _channel_arrays(run, channel)
    info = _CHANNEL_INFO[channel]
    meas_ts = run.meas_ts_rel

    plot = Plot(time_axis=TimeAxis.raw)
    plot.add(
        Series(
            ts=run.cmd_ts_rel.tolist(),
            values=cmd_arr.tolist(),
            label=f"cmd_{channel} [{info['unit']}]",
            color=info["cmd"],
        )
    )
    if meas_ts.size:
        plot.add(
            Series(
                ts=meas_ts.tolist(),
                values=meas_arr.tolist(),
                label=f"meas_{channel} [{info['unit']}]",
                color=info["meas"],
            )
        )
        plot.add(
            Markers(
                ts=run.cmd_ts_rel.tolist(),
                values=cmd_arr.tolist(),
                label="cmd (markers)",
                color=info["cmd"],
                radius=0.3,
            )
        )

    return plot.to_svg(), {
        "channel": channel,
        "cmd_max": float(np.max(cmd_arr)) if cmd_arr.size else 0.0,
        "cmd_min": float(np.min(cmd_arr)) if cmd_arr.size else 0.0,
    }


def _render_constant(run: LoadedRun) -> tuple[str, dict[str, Any]]:
    from dimos.memory2.vis.plot.elements import Series
    from dimos.memory2.vis.plot.plot import Plot, TimeAxis

    vx_meas, vy_meas, wz_meas = _reconstruct_or_empty(run)
    meas_ts = run.meas_ts_rel

    plot = Plot(time_axis=TimeAxis.raw)
    for ch, cmd_values in (("vx", run.cmd_vx), ("vy", run.cmd_vy), ("wz", run.cmd_wz)):
        plot.add(
            Series(
                ts=run.cmd_ts_rel.tolist(),
                values=cmd_values.tolist(),
                label=f"cmd_{ch}",
                color=_CHANNEL_INFO[ch]["cmd"],
            )
        )
    if meas_ts.size:
        for ch, meas_values in (("vx", vx_meas), ("vy", vy_meas), ("wz", wz_meas)):
            plot.add(
                Series(
                    ts=meas_ts.tolist(),
                    values=meas_values.tolist(),
                    label=f"meas_{ch}",
                    color=_CHANNEL_INFO[ch]["meas"],
                )
            )
    return plot.to_svg(), {}


# ---------------------------------------------------------------------- channel helpers


def _dominant_channel(run: LoadedRun) -> str:
    """Pick the channel with the largest commanded amplitude.

    Used so an E2 wz-step run plots wz, not vx.
    """
    amps = {
        "vx": float(np.max(np.abs(run.cmd_vx))) if run.cmd_vx.size else 0.0,
        "vy": float(np.max(np.abs(run.cmd_vy))) if run.cmd_vy.size else 0.0,
        "wz": float(np.max(np.abs(run.cmd_wz))) if run.cmd_wz.size else 0.0,
    }
    return max(amps, key=lambda k: amps[k]) if any(amps.values()) else "vx"


def _channel_arrays(run: LoadedRun, channel: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (cmd_array, meas_array) for the requested channel.

    ``meas_array`` is the *Hampel-filtered* velocity (current default
    pipeline). For raw vs filtered diagnostics see ``_channel_arrays_dual``.
    """
    vx_meas, vy_meas, wz_meas = _reconstruct_or_empty(run)
    if channel == "vx":
        return run.cmd_vx, vx_meas
    if channel == "vy":
        return run.cmd_vy, vy_meas
    return run.cmd_wz, wz_meas


def _channel_arrays_dual(run: LoadedRun, channel: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (cmd, meas_raw, meas_filtered).

    ``meas_raw`` is the velocity reconstruction with Hampel disabled; the
    filtered version uses default ``hampel_n_sigma=3.0``. Diff between
    the two = samples the Hampel filter replaced.
    """
    vx_h, vy_h, wz_h = _reconstruct_or_empty(run, hampel_n_sigma=3.0)
    vx_r, vy_r, wz_r = _reconstruct_or_empty(run, hampel_n_sigma=float("inf"))
    if channel == "vx":
        return run.cmd_vx, vx_r, vx_h
    if channel == "vy":
        return run.cmd_vy, vy_r, vy_h
    return run.cmd_wz, wz_r, wz_h


def _reconstruct_or_empty(
    run: LoadedRun, *, hampel_n_sigma: float = 3.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Body vx/vy/wz arrays aligned to ``run.meas_ts_wall``, or empty arrays if no odom."""
    from dimos.utils.characterization.scripts.analyze import reconstruct_body_velocities

    if run.meas_ts_wall.size < 3:
        empty = np.zeros(0, dtype=float)
        return empty, empty, empty
    # Ensure strictly increasing ts (memory2 SQLite may have dup-ts near clock ticks).
    ts = run.meas_ts_wall
    order = np.argsort(ts, kind="stable")
    ts_s = ts[order]
    x_s = run.meas_x[order]
    y_s = run.meas_y[order]
    yaw_s = run.meas_yaw[order]
    keep = np.concatenate([[True], np.diff(ts_s) > 0])
    return reconstruct_body_velocities(
        ts_s[keep], x_s[keep], y_s[keep], yaw_s[keep], hampel_n_sigma=hampel_n_sigma
    )


__all__ = ["render_overlay", "render_run"]
