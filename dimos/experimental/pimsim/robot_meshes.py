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

"""MuJoCo-based robot mesh extractor + FK helper for the Viser viewer.

Loads an MJCF (the same one the sim subprocess loads), pulls out visual
mesh geoms with their parent-body indices and local poses, and runs FK
on demand to give world poses for every body in the model.

Lets the Viser render module display the same robot the simulation is
stepping, without forcing a separate URDF or duplicating geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class GeomInstance:
    """A single visual mesh geom, parented to a body."""

    body_name: str
    vertices: np.ndarray  # (V, 3)
    faces: np.ndarray  # (F, 3)
    local_pos: np.ndarray  # (3,)
    local_wxyz: np.ndarray  # (4,)
    rgba: tuple[float, float, float, float]


@dataclass
class RobotMeshes:
    model: mujoco.MjModel
    data: mujoco.MjData
    geoms: list[GeomInstance]
    # joint-name (in MJCF order) -> qpos address.  Used to splice incoming
    # joint_state values into the right slots of qpos.
    qpos_addr_by_mjcf_name: dict[str, int]
    # body_id -> body name (for viser entity paths).
    body_names: list[str]


def load_robot_meshes(
    mjcf_path: str | Path,
    *,
    visual_groups: tuple[int, ...] = (0, 1, 2),
    assets: dict[str, bytes] | None = None,
) -> RobotMeshes:
    """Parse the MJCF, pull visual mesh geoms into Python arrays.

    ``visual_groups`` defaults to MuJoCo's convention where group 0-2 are
    visual and group 3+ are collision.  Most menagerie / dimos models
    follow this; if a model uses different groups, override it.

    ``assets`` is an optional ``{filename: bytes}`` map for mesh files
    referenced by bare name in the MJCF (e.g. menagerie meshes).
    Pass ``dimos.simulation.mujoco.model.get_assets()`` for G1.
    When omitted, meshes are resolved from disk relative to ``mjcf_path``
    (the MJCF's own ``meshdir`` attribute, if present, applies normally).
    """
    if assets is None:
        # Disk-based: mujoco resolves <mesh file="..."/> relative to the
        # MJCF's meshdir. Works for any robot that ships meshes on disk.
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    else:
        with open(mjcf_path) as f:
            xml = f.read()
        model = mujoco.MjModel.from_xml_string(xml, assets)
    data = mujoco.MjData(model)

    geoms: list[GeomInstance] = []
    for gid in range(model.ngeom):
        if int(model.geom_group[gid]) not in visual_groups:
            continue
        gtype = int(model.geom_type[gid])
        body_id = int(model.geom_bodyid[gid])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        rgba = tuple(float(x) for x in model.geom_rgba[gid])
        local_pos = np.array(model.geom_pos[gid], dtype=np.float32).copy()
        local_wxyz = np.array(model.geom_quat[gid], dtype=np.float32).copy()
        size = model.geom_size[gid]

        # Mesh: pull vertices/faces from the model.
        if gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(model.geom_dataid[gid])
            if mesh_id < 0:
                continue
            v_start = int(model.mesh_vertadr[mesh_id])
            v_count = int(model.mesh_vertnum[mesh_id])
            f_start = int(model.mesh_faceadr[mesh_id])
            f_count = int(model.mesh_facenum[mesh_id])
            vertices = np.array(
                model.mesh_vert[v_start : v_start + v_count],
                dtype=np.float32,
            ).copy()
            faces = np.array(
                model.mesh_face[f_start : f_start + f_count],
                dtype=np.int32,
            ).copy()
        # Box: tessellate as 8 verts + 12 triangles, half-sizes from
        # geom_size[0..2].  Lets us render <geom type="box"> primitives
        # (manip_table, manip_cube, scene-editor exports) without a
        # mesh asset.
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            hx, hy, hz = float(size[0]), float(size[1]), float(size[2])
            vertices = np.array(
                [
                    [-hx, -hy, -hz],
                    [hx, -hy, -hz],
                    [hx, hy, -hz],
                    [-hx, hy, -hz],
                    [-hx, -hy, hz],
                    [hx, -hy, hz],
                    [hx, hy, hz],
                    [-hx, hy, hz],
                ],
                dtype=np.float32,
            )
            # Outward-facing CCW triangles (verified by cross-product).
            faces = np.array(
                [
                    [0, 2, 1],
                    [0, 3, 2],  # -Z (bottom)
                    [4, 5, 6],
                    [4, 6, 7],  # +Z (top)
                    [0, 1, 5],
                    [0, 5, 4],  # -Y
                    [1, 2, 6],
                    [1, 6, 5],  # +X
                    [2, 3, 7],
                    [2, 7, 6],  # +Y
                    [3, 0, 4],
                    [3, 4, 7],  # -X
                ],
                dtype=np.int32,
            )
        else:
            # Sphere, cylinder, plane, etc. — skip for now.  Only manip
            # rigs and scene-editor exports use boxes; everything else
            # the dimos sims care about is a mesh.
            continue

        geoms.append(
            GeomInstance(
                body_name=body_name,
                vertices=vertices,
                faces=faces,
                local_pos=local_pos,
                local_wxyz=local_wxyz,
                rgba=rgba,  # type: ignore[arg-type]
            )
        )

    qpos_addr_by_mjcf_name: dict[str, int] = {}
    for jid in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jname is None:
            continue
        qpos_addr_by_mjcf_name[jname] = int(model.jnt_qposadr[jid])

    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body_{bid}"
        for bid in range(model.nbody)
    ]

    return RobotMeshes(
        model=model,
        data=data,
        geoms=geoms,
        qpos_addr_by_mjcf_name=qpos_addr_by_mjcf_name,
        body_names=body_names,
    )
