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

"""Cook authored interactables out of a static source scene."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from dimos.simulation.scene_assets.mesh_scene import (
    SceneMeshAlignment,
    ScenePrimMesh,
    load_scene_prims,
)
from dimos.simulation.scene_assets.sidecar import InteractableSpec, SceneCookSidecar
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


def cook_interactable_assets(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    sidecar: SceneCookSidecar,
    alignment: SceneMeshAlignment,
    rebake: bool = False,
) -> list[dict[str, Any]]:
    """Write per-interactable assets and return runtime metadata.

    The visual mesh is stored in the entity's local frame, centered on the
    exported AABB.  Runtime physics may still use a simple primitive collider;
    the visual asset is just the authored mesh that pimsim can parent to that
    collider.
    """
    if not sidecar.interactables:
        return []

    prims = load_scene_prims(source_path, alignment=alignment)
    entities_dir = Path(output_dir).expanduser().resolve() / "entities"
    entities: list[dict[str, Any]] = []
    for interactable in sidecar.interactables:
        matched = [prim for prim in prims if interactable.matches(prim)]
        if not matched:
            logger.warning(
                "scene interactable %s matched no prims: %s",
                interactable.id,
                ", ".join(interactable.source_prim_paths),
            )
            continue
        metadata = _cook_one_interactable(
            interactable,
            matched,
            entities_dir,
            rebake=rebake,
        )
        entities.append(metadata)
    return entities


def _cook_one_interactable(
    interactable: InteractableSpec,
    prims: list[ScenePrimMesh],
    entities_dir: Path,
    *,
    rebake: bool,
) -> dict[str, Any]:
    safe_id = _safe_entity_id(interactable.id)
    entity_dir = entities_dir / safe_id
    entity_dir.mkdir(parents=True, exist_ok=True)
    visual_path = entity_dir / "visual.glb"

    vertices = np.concatenate([prim.vertices for prim in prims], axis=0)
    aabb_min = vertices.min(axis=0).astype(float)
    aabb_max = vertices.max(axis=0).astype(float)
    center = ((aabb_min + aabb_max) * 0.5).astype(float)
    extents = np.maximum(aabb_max - aabb_min, 1e-4).astype(float)

    if rebake or not visual_path.exists():
        _write_interactable_glb(visual_path, prims, center)

    physics = dict(interactable.physics)
    shape_hint = str(physics.get("shape", "box"))
    shape_extents = physics.get("extents")
    if shape_extents is None and shape_hint == "box":
        shape_extents = extents.tolist()
    elif shape_extents is None and shape_hint == "sphere":
        shape_extents = [float(max(extents) * 0.5)]
    elif shape_extents is None and shape_hint == "cylinder":
        shape_extents = [float(max(extents[0], extents[1]) * 0.5), float(extents[2])]
    elif shape_extents is None:
        shape_extents = []

    descriptor = {
        "entity_id": interactable.id,
        "kind": interactable.kind,
        "mesh_ref": f"entities/{safe_id}/visual.glb",
        "shape_hint": shape_hint,
        "extents": [float(value) for value in shape_extents],
        "mass": float(interactable.mass),
    }

    return {
        "id": interactable.id,
        "tags": list(interactable.tags),
        "source_prim_paths": list(interactable.source_prim_paths),
        "matched_prim_paths": [prim.prim_path or prim.name for prim in prims],
        "remove_from_static": interactable.remove_from_static,
        "spawn": interactable.spawn,
        "aabb": {
            "min": [float(value) for value in aabb_min],
            "max": [float(value) for value in aabb_max],
        },
        "initial_pose": {
            "x": float(center[0]),
            "y": float(center[1]),
            "z": float(center[2]),
            "qw": 1.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
        },
        "visual_path": str(visual_path),
        "descriptor": descriptor,
        "physics": physics,
        "visual": interactable.visual,
        "sensor": interactable.sensor,
    }


def _write_interactable_glb(
    visual_path: Path,
    prims: list[ScenePrimMesh],
    center: np.ndarray,
) -> None:
    combined_vertices: list[np.ndarray] = []
    combined_faces: list[np.ndarray] = []
    offset = 0
    for prim in prims:
        verts = prim.vertices.astype(np.float64, copy=False) - center[None, :]
        faces = prim.triangles.astype(np.int64, copy=False) + offset
        combined_vertices.append(verts)
        combined_faces.append(faces)
        offset += len(verts)
    mesh = trimesh.Trimesh(
        vertices=np.concatenate(combined_vertices, axis=0),
        faces=np.concatenate(combined_faces, axis=0),
        process=False,
    )
    mesh.export(str(visual_path))


def _safe_entity_id(entity_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in entity_id)
    return safe or "entity"


__all__ = ["cook_interactable_assets"]
