#!/usr/bin/env python3
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

"""Basic G1 stack: base sensors plus real robot connection and ROS nav."""

from dimos.core.coordination.blueprints import autoconnect
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner
from dimos.robot.unitree.g1.blueprints.primitive.unitree_g1_onboard import unitree_g1_onboard
from dimos.robot.unitree.g1.blueprints.primitive.unitree_g1_vis import unitree_g1_vis

unitree_g1_nav_simple = autoconnect(
    unitree_g1_onboard,
    RayTracingVoxelMap.blueprint(voxel_size=0.05),
    CostMapper.blueprint(),
    ReplanningAStarPlanner.blueprint(),
    MovementManager.blueprint(),
    unitree_g1_vis,
).global_config(n_workers=10, robot_model="unitree_g1")

__all__ = ["unitree_g1_nav_simple"]
