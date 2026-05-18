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

"""Layer 1 sim plant for the Go2 base.

Per-channel FOPDT velocity tracking + unicycle kinematics. Tick-based:
each call to :meth:`Go2PlantSim.step` advances one control period.

The vendored fitted parameters (``GO2_PLANT_FITTED``) live at the bottom
of this module — types, simulator, and the measured values in one place.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math


@dataclass
class FopdtChannelParams:
    """First-order-plus-dead-time params for a single velocity channel.

    Symbols match the characterization fitter:
      K   - steady-state gain (output / commanded)
      tau - first-order time constant (s)
      L   - pure dead-time (s)
    """

    K: float
    tau: float
    L: float


class FOPDTChannel:
    """First-order lag + dead-time for one velocity axis.

    Tick-based: feed one commanded value per :meth:`step` call, get the
    delayed/lagged actual velocity back.
    """

    def __init__(self, params: FopdtChannelParams) -> None:
        self.params = params
        self._delay_buf: deque[float] = deque()
        self._delay_samples = 0
        self._y = 0.0

    def reset(self, dt: float) -> None:
        self._delay_samples = max(1, int(self.params.L / dt))
        self._delay_buf = deque([0.0] * self._delay_samples, maxlen=self._delay_samples)
        self._y = 0.0

    def step(self, u: float, dt: float) -> float:
        self._delay_buf.append(u)
        u_delayed = self._delay_buf[0]
        alpha = dt / (self.params.tau + dt)
        self._y += alpha * (self.params.K * u_delayed - self._y)
        return self._y


@dataclass
class Go2PlantParams:
    """FOPDT params for all three velocity channels."""

    vx: FopdtChannelParams
    vy: FopdtChannelParams
    wz: FopdtChannelParams


class Go2PlantSim:
    """Unicycle kinematic sim with FOPDT velocity response per channel.

    Body-frame velocities `(vx, vy, wz)` are commanded; the plant produces
    actual velocities (filtered + delayed) that drive a unicycle integrator
    in the world frame.
    """

    def __init__(self, params: Go2PlantParams) -> None:
        self.params = params
        self.ch_vx = FOPDTChannel(params.vx)
        self.ch_vy = FOPDTChannel(params.vy)
        self.ch_wz = FOPDTChannel(params.wz)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0

    def reset(self, x: float, y: float, yaw: float, dt: float) -> None:
        self.x, self.y, self.yaw = x, y, yaw
        self.vx = self.vy = self.wz = 0.0
        for ch in (self.ch_vx, self.ch_vy, self.ch_wz):
            ch.reset(dt)

    def step(self, cmd_vx: float, cmd_vy: float, cmd_wz: float, dt: float) -> None:
        self.vx = self.ch_vx.step(cmd_vx, dt)
        self.vy = self.ch_vy.step(cmd_vy, dt)
        self.wz = self.ch_wz.step(cmd_wz, dt)

        self.x += (self.vx * math.cos(self.yaw) - self.vy * math.sin(self.yaw)) * dt
        self.y += (self.vx * math.sin(self.yaw) + self.vy * math.cos(self.yaw)) * dt
        self.yaw = (self.yaw + self.wz * dt + math.pi) % (2 * math.pi) - math.pi


# --- Vendored fitted FOPDT plant for the Go2 base ------------------------
#
# Source: concrete surface, normal/default mode, data collected
# 2026-05-07, fitted by the characterization pipeline. RISE tau/L
# corrected 2026-05-16: an earlier pooled fit was degenerate (tau pinned
# at the solver lower bound with all lag collapsed into L); a per-run
# re-fit of the same raw E1/E2 data (median over converged forward
# trials, fresh-fit r2=0.92 vx / 0.82 wz) gives the true structure —
# small dead-time (L ~ 0.05-0.07 s), larger tau (vx ~ 0.40,
# wz ~ 0.55-0.60 s). K is unchanged (independently validated).
#
# This is the ground truth the sim self-test injects and recovers, and
# the documented rationale for the derived feedforward gains. vy on the
# real robot strafes and should be characterized for real (--mode hw);
# the sim ground truth copies vx into vy only because the sim FOPDT has
# no independent lateral model.

GO2_VX_RISE = FopdtChannelParams(K=0.922, tau=0.395, L=0.065)
GO2_WZ_RISE = FopdtChannelParams(K=2.453, tau=0.596, L=0.052)

GO2_PLANT_FITTED = Go2PlantParams(
    vx=GO2_VX_RISE,
    vy=GO2_VX_RISE,  # sim ground-truth placeholder; real vy via --mode hw
    wz=GO2_WZ_RISE,
)
