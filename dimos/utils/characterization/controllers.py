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

"""Controller-fn factories for the trajectory-tracking diagnostic.

These produce callables of shape ``(t, pose, ref) -> (vx, vy, wz)`` that
the characterization session ticks at the recipe's sample rate.

Two flavors:

  - :func:`openloop_ff_controller` ignores pose entirely; sends the
    reference velocity through the static plant-gain inverse. Used for
    short transient trials where open-loop integration drift is not a
    concern (single steps, brief profiles).

  - :func:`lowgain_p_controller` adds a body-frame proportional
    correction on top of the feedforward. The gain is small enough that
    plant dynamics still dominate the response — but enough to keep the
    robot near the reference for the 20-30 s sustained trials where
    pure open-loop would drift off the page.

Both fall back to feedforward-only when ``pose`` is ``None`` (stale or
unseen).
"""

from __future__ import annotations

from collections.abc import Callable
import math

from dimos.control.tasks.feedforward_gain_compensator import (
    FeedforwardGainCompensator,
)
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.utils.characterization.trajectories import TrajRefState
from dimos.utils.trigonometry import angle_diff

ControllerFn = Callable[[float, PoseStamped | None, TrajRefState], tuple[float, float, float]]


def openloop_ff_controller(ff: FeedforwardGainCompensator) -> ControllerFn:
    """Feedforward-only: ``cmd = (ref.vx/K_vx, 0, ref.wz/K_wz)``, ignores pose."""

    def fn(t: float, pose: PoseStamped | None, ref: TrajRefState) -> tuple[float, float, float]:
        del t, pose  # unused
        return ff.compute(ref.vx, 0.0, ref.wz)

    return fn


def lowgain_p_controller(
    ff: FeedforwardGainCompensator,
    *,
    k_pos: float = 0.0,
    k_yaw: float = 0.15,
) -> ControllerFn:
    """FF + low-gain proportional correction in body frame.

    Position error ``(ref.x - pose.x, ref.y - pose.y)`` is rotated into
    the robot's body frame (so the controller's "forward correction"
    is along the robot's current heading), scaled by ``k_pos``, and
    added to the feedforward ``vx``. Yaw correction uses :func:`angle_diff`
    and is added to feedforward ``wz``. Returns FF-only when ``pose`` is
    ``None``.
    """

    def fn(t: float, pose: PoseStamped | None, ref: TrajRefState) -> tuple[float, float, float]:
        del t  # unused
        ff_vx, ff_vy, ff_wz = ff.compute(ref.vx, 0.0, ref.wz)
        if pose is None:
            return ff_vx, ff_vy, ff_wz

        ex = ref.x - pose.position.x
        ey = ref.y - pose.position.y
        yaw = pose.orientation.euler[2]
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        # Body-frame projection (robot's +x forward, +y left)
        body_forward = cos_y * ex + sin_y * ey
        # body_left = -sin_y * ex + cos_y * ey  # unused: vy is not commanded

        yaw_err = angle_diff(ref.yaw, yaw)

        return (
            ff_vx + k_pos * body_forward,
            ff_vy,
            ff_wz + k_yaw * yaw_err,
        )

    return fn


__all__ = [
    "ControllerFn",
    "lowgain_p_controller",
    "openloop_ff_controller",
]
