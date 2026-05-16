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

"""Vendored fitted FOPDT plant models for the Go2 base.

Source artifacts (concrete surface, normal/default mode, data collected
2026-05-07, fitted 2026-05-11):
  ~/char_data/2026-05-07_concrete_normal/session_20260507-193709/modeling/model_summary.json  (vx)
  ~/char_data/2026-05-07_concrete_normal/session_20260507-201228/modeling/model_summary.json  (wz)

Produced by the characterization fitting pipeline at
:mod:`dimos.utils.characterization.modeling.session.fit_session`.

History: previously vendored from old April rage/default sessions with
``wz K=2.175``. The trajectory-tracking diagnostic showed that stale
``K_wz`` caused ~12% wz over-rotation on sustained-curvature paths
(robot swept 14.0 rad where the reference wanted 12.5). Re-vendored
2026-05-15 from the latest concrete normal-mode fit.

Caveats:
  - Only "rise" params feed :class:`Go2PlantParams`. The real plant has
    rise/fall asymmetry (wz: K_fall/K_rise=1.13, verdict "differs"; see
    GO2_*_FALL below). :class:`FOPDTChannel` is single-regime; modeling
    the asymmetry is the open follow-up for the closed-loop residual.
  - vx rise tau fitted as ~0.001 s (degenerate: the fit collapsed the
    lag into deadtime). We substitute the physically-meaningful fall tau
    (0.231 s) for the rise model; vx K and L are the fresh fitted values.
    vx is not the binding channel — open-loop vx trials track within
    ~0.9x of prediction.
  - vy is a placeholder copy of vx params: Go2 has no native lateral
    velocity, so any controller commanding vy on the real robot will
    behave very differently from the sim. Treat vy commands as a sim
    artifact, not a hardware-relevant signal.
"""

from __future__ import annotations

from dimos.utils.benchmarking.plant import FopdtChannelParams, Go2PlantParams

# Fresh fit (concrete, normal mode, 2026-05-07). See docstring re: vx tau
# substitution (fitted rise tau ~0.001 s is degenerate; using fall tau).
GO2_VX_RISE = FopdtChannelParams(K=0.922, tau=0.231, L=0.213)
GO2_VX_FALL = FopdtChannelParams(K=1.044, tau=0.231, L=0.123)
GO2_WZ_RISE = FopdtChannelParams(K=2.453, tau=0.172, L=0.148)
GO2_WZ_FALL = FopdtChannelParams(K=2.765, tau=0.202, L=0.101)

GO2_PLANT_FITTED = Go2PlantParams(
    vx=GO2_VX_RISE,
    vy=GO2_VX_RISE,  # placeholder - see module docstring
    wz=GO2_WZ_RISE,
)
