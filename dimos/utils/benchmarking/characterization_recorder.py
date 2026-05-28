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

"""Telemetry recorder for the Go2 characterization blueprint.

Captures the live streams a characterization session produces — commanded
twist, coord-aggregated joint_state, raw odom, and operator gate events
— into a per-session SQLite DB so post-process tools can re-fit, dissect
spikes, or compare runs without re-running on hardware.

The DB lands next to the JSON+PNG artifact at
``<repo>/data/characterization/<robot_id>/<robot_id>_recording_<date>_<sha>.db``
by default. Read it back with::

    from dimos.memory2.store.sqlite import SqliteStore
    store = SqliteStore(path="<the .db file>")
    store.start()
    for obs in store.stream("joint_state", JointState).iterate_ts():
        ts, msg = obs.ts, obs.data
        # re-fit, plot, etc.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from dimos.core.core import rpc
from dimos.core.stream import In
from dimos.memory2.module import Recorder, RecorderConfig
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Int8 import Int8
from dimos.utils.benchmarking.tuning import git_sha
from dimos.utils.path_utils import get_project_root

DEFAULT_OUT_DIR = get_project_root() / "data" / "characterization"


class CharacterizationRecorderConfig(RecorderConfig):
    """Same as :class:`RecorderConfig` but with per-session db_path
    resolution from ``out_dir`` + ``robot_id``. Set ``db_path`` explicitly
    to bypass the default naming convention."""

    out_dir: str | None = None  # None -> <repo>/data/characterization/
    robot_id: str = "go2"
    # Timestamped filenames make rerun-safe defaults; never silently
    # clobber prior recordings.
    overwrite: bool = False


class CharacterizationRecorder(Recorder):
    """Records the streams a characterization session emits.

    Composed alongside :class:`~dimos.utils.benchmarking.characterization.Characterizer`
    in the ``unitree-go2-characterization`` blueprint; ports are wired
    to the same LCM topics the rest of the stack already uses (LCM is
    multicast — additional subscribers are free).
    """

    config: CharacterizationRecorderConfig

    cmd_vel: In[Twist]  # commanded /cmd_vel during each SI step
    joint_state: In[JointState]  # /coordinator/joint_state (x, y, yaw)
    odom: In[PoseStamped]  # raw /go2/odom from GO2Connection
    gate: In[Int8]  # operator gate events (advance/skip/quit)

    @rpc
    def start(self) -> None:
        # Resolve a per-session db_path before super().start() opens the
        # SqliteStore. Mirrors the JSON+PNG artifact naming so a
        # session's three files (config.json, fits.png, recording.db)
        # land in the same dir with the same date+sha suffix.
        out_dir = (
            Path(self.config.out_dir).expanduser()
            if self.config.out_dir
            else DEFAULT_OUT_DIR / self.config.robot_id
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        self.config.db_path = (
            out_dir / f"{self.config.robot_id}_recording_{date.today().isoformat()}_{git_sha()}.db"
        )
        super().start()


__all__ = ["CharacterizationRecorder", "CharacterizationRecorderConfig"]
