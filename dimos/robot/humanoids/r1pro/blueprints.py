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

"""R1 Pro dual-arm blueprints (single URDF, two 7-DOF arms).

Usage:
    dimos run r1pro-dual-mock             # Mock coordinator only
    dimos run r1pro-planner-coordinator   # Planner + coordinator (plan & execute)
"""

from dimos.control.coordinator import ControlCoordinator
from dimos.core.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.catalog.galaxea import r1pro_arm

_left = r1pro_arm(side="left")
_right = r1pro_arm(side="right")

# Mock dual-arm coordinator (no planner, no visualization)
r1pro_dual_mock = ControlCoordinator.blueprint(
    hardware=[_left.to_hardware_component(), _right.to_hardware_component()],
    tasks=[_left.to_task_config(), _right.to_task_config()],
).transports(
    {
        ("joint_state", JointState): LCMTransport("/coordinator/joint_state", JointState),
    }
)

# Planner + coordinator (plan, preview in Meshcat, execute via mock adapters)
r1pro_planner_coordinator = autoconnect(
    ManipulationModule.blueprint(
        robots=[
            _left.to_robot_model_config(),
            _right.to_robot_model_config(),
        ],
        planning_timeout=10.0,
        enable_viz=True,
    ),
    ControlCoordinator.blueprint(
        tick_rate=100.0,
        publish_joint_state=True,
        joint_state_frame_id="coordinator",
        hardware=[_left.to_hardware_component(), _right.to_hardware_component()],
        tasks=[_left.to_task_config(), _right.to_task_config()],
    ),
).transports(
    {
        ("joint_state", JointState): LCMTransport("/coordinator/joint_state", JointState),
    }
)


__all__ = [
    "r1pro_dual_mock",
    "r1pro_planner_coordinator",
]
