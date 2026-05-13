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

"""Principled tuning of RPP-FF for a given (path, speed, plant).

Two pieces, derived rather than guessed:

* :func:`geometric_lookahead` — Pure-Pursuit lookahead from path curvature
  and the robot's physical minimum-turn-radius. Reports infeasible when the
  path is tighter than the plant can follow at the requested speed.
* :func:`imc_cross_track_gains` — damped-second-order (K_p, K_d) for the
  cross-track PID, parameterised by a single closed-loop bandwidth knob.

:func:`tune_rpp_for_path` composes both into a single config dict that
:func:`runner.run_rpp_sim` and :func:`runner.run_rpp_hw` accept directly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dimos.msgs.nav_msgs.Path import Path


def path_curvature(path: Path) -> np.ndarray:
    """Curvature (1/m) at each waypoint via Menger's three-point formula.

    Endpoints get 0. Degenerate triples (collinear or coincident) also get 0.
    """
    xs = np.array([p.position.x for p in path.poses])
    ys = np.array([p.position.y for p in path.poses])
    n = len(xs)
    kappa = np.zeros(n)
    for i in range(1, n - 1):
        x1, y1 = xs[i - 1], ys[i - 1]
        x2, y2 = xs[i], ys[i]
        x3, y3 = xs[i + 1], ys[i + 1]
        a = math.hypot(x1 - x2, y1 - y2)
        b = math.hypot(x2 - x3, y2 - y3)
        c = math.hypot(x3 - x1, y3 - y1)
        if a * b * c < 1e-12:
            continue
        s = 0.5 * (a + b + c)
        area_sq = max(0.0, s * (s - a) * (s - b) * (s - c))
        kappa[i] = 4.0 * math.sqrt(area_sq) / (a * b * c)
    return kappa


def geometric_lookahead(
    path: Path,
    v: float,
    *,
    wz_max: float = 1.5,
    L_min: float = 0.2,
    L_max: float = 1.5,
    smoothing_window: int = 5,
    margin: float = 1.2,
) -> tuple[float, dict]:
    """Recommended PurePursuit lookahead for a (path, speed) pair.

    L_optimal = clip(2 · max(R_curve_min, R_robot_min) · margin, L_min, L_max)

    Where:
      R_curve_min = 1 / max(smoothed κ along path)
      R_robot_min = v / wz_max

    Returns ``(L_optimal, diagnostics)``. ``diagnostics['infeasible']`` is True
    when the path's tightest curve is tighter than what the robot can follow
    at the requested speed.
    """
    kappa = path_curvature(path)
    if smoothing_window > 1 and len(kappa) >= smoothing_window:
        half = smoothing_window // 2
        smoothed = np.array(
            [float(np.max(kappa[max(0, i - half) : i + half + 1])) for i in range(len(kappa))]
        )
    else:
        smoothed = kappa
    kappa_max = float(np.max(smoothed))
    R_curve_min = (1.0 / kappa_max) if kappa_max > 1e-6 else float("inf")
    R_robot_min = v / max(wz_max, 1e-6)
    infeasible = R_curve_min < R_robot_min
    R_design = max(R_curve_min, R_robot_min) * margin
    L = float(np.clip(2.0 * R_design, L_min, L_max))
    return L, {
        "R_curve_min": R_curve_min,
        "R_robot_min": R_robot_min,
        "kappa_max": kappa_max,
        "infeasible": infeasible,
        "L_optimal": L,
    }


def imc_cross_track_gains(
    v: float, *, omega_n: float = 0.6, zeta: float = 0.7
) -> tuple[float, float]:
    """Damped-second-order target gains for the cross-track PID.

    The cross-track loop is approximately a double integrator (wz → yaw
    → lateral position) with a wz-FOPDT lag. For a desired closed-loop
    natural frequency ``omega_n`` and damping ``zeta``, the linearised
    gains around forward speed ``v`` are::

        K_p = omega_n^2 / v
        K_d = 2·zeta·omega_n / v

    Integral term is intentionally zero — there's no DC bias on a smooth
    path, and an integrator just adds wind-up risk.

    ``omega_n`` is the *single* tuning knob (rad/s of desired loop bandwidth).
    Higher = more aggressive = more sensitive to noise.
    """
    if v < 1e-3:
        return 0.0, 0.0
    K_p = (omega_n**2) / v
    K_d = (2.0 * zeta * omega_n) / v
    return K_p, K_d


def tune_rpp_for_path(
    path: Path,
    v: float,
    *,
    wz_max: float = 1.5,
    omega_n: float = 0.6,
    zeta: float = 0.7,
    L_min: float = 0.2,
    L_max: float = 1.5,
) -> dict:
    """One-call tuning. Returns kwargs for :func:`run_rpp_sim` / :func:`run_rpp_hw`.

    Composes :func:`geometric_lookahead` (for ``max_lookahead``) and
    :func:`imc_cross_track_gains` (for ``ct_kp``, ``ct_kd``) into one config
    dict. Pass the result through with ``**kwargs``.

    The dict also carries a ``_diagnostics`` entry (for inspection) — strip
    it before forwarding to the runner if you don't want kwargs noise.
    """
    L, diag = geometric_lookahead(path, v, wz_max=wz_max, L_min=L_min, L_max=L_max)
    K_p, K_d = imc_cross_track_gains(v, omega_n=omega_n, zeta=zeta)
    return {
        "max_lookahead": L,
        "ct_kp": K_p,
        "ct_kd": K_d,
        "_diagnostics": diag,
    }


__all__ = [
    "geometric_lookahead",
    "imc_cross_track_gains",
    "path_curvature",
    "tune_rpp_for_path",
]
