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

"""Interactive blueprint: Evaluator + MlsPlanner + rerun visualization.

Run with:
    python -m dimos.navigation.nav_stack.tests.mls_tester
"""

from __future__ import annotations

from typing import Any

from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.navigation.nav_stack.modules.mls_planner.module import MlsPlanner
from dimos.navigation.nav_stack.tests.evaluator import Evaluator, default_scene
from dimos.visualization.rerun.bridge import RerunBridgeModule


def _render_surfaces(msg: PointCloud2) -> Any:
    return msg.to_rerun(voxel_size=0.075, mode="boxes")


def build_blueprint() -> Blueprint:
    return autoconnect(
        Evaluator.blueprint(scene=default_scene()),
        MlsPlanner.blueprint(),
        RerunBridgeModule.blueprint(
            visual_override={"world/surfaces": _render_surfaces},
        ),
    )


if __name__ == "__main__":
    ModuleCoordinator.build(build_blueprint()).loop()
