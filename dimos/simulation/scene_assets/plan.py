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

"""Resolved scene cook plan.

The authored sidecar is user intent.  The cook plan is resolved source-scene
membership: exactly which source prims become each runtime entity, which visual
nodes Blender must extract/delete, and which collision policy every downstream
cook must consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import re
from typing import Any

import numpy as np

from dimos.simulation.mujoco.collision_spec import CollisionSpec
from dimos.simulation.scene_assets.mesh_scene import (
    SceneMeshAlignment,
    ScenePrimMesh,
    load_scene_prims,
)
from dimos.simulation.scene_assets.sidecar import InteractableSpec, SceneCookSidecar

_HASH_SUFFIX_RE = re.compile(r"_[0-9a-fA-F]{6,}$")


@dataclass(frozen=True)
class EntityCookPlan:
    """Resolved authored entity."""

    spec: InteractableSpec
    safe_id: str
    matched_prim_paths: tuple[str, ...]
    visual_node_patterns: tuple[str, ...]
    aabb_min: tuple[float, float, float]
    aabb_max: tuple[float, float, float]
    center: tuple[float, float, float]
    descriptor: dict[str, Any]
    visual_path: Path

    def to_metadata(self) -> dict[str, Any]:
        return {
            "id": self.spec.id,
            "tags": list(self.spec.tags),
            "source_prim_paths": list(self.spec.source_prim_paths),
            "matched_prim_paths": list(self.matched_prim_paths),
            "visual_node_patterns": list(self.visual_node_patterns),
            "remove_from_static": self.spec.remove_from_static,
            "spawn": self.spec.spawn,
            "aabb": {
                "min": list(self.aabb_min),
                "max": list(self.aabb_max),
            },
            "initial_pose": {
                "x": self.center[0],
                "y": self.center[1],
                "z": self.center[2],
                "qw": 1.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
            },
            "visual_path": str(self.visual_path),
            "descriptor": self.descriptor,
            "physics": self.spec.physics,
            "visual": self.spec.visual,
            "sensor": self.spec.sensor,
        }

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "id": self.spec.id,
            "safe_id": self.safe_id,
            "matched_prim_paths": list(self.matched_prim_paths),
            "visual_node_patterns": list(self.visual_node_patterns),
            "aabb": {"min": list(self.aabb_min), "max": list(self.aabb_max)},
            "center": list(self.center),
            "descriptor": self.descriptor,
            "visual_path": str(self.visual_path),
            "remove_from_static": self.spec.remove_from_static,
        }


@dataclass(frozen=True)
class SceneCookPlan:
    """Resolved plan shared by every artifact writer."""

    source_path: Path
    alignment: SceneMeshAlignment
    sidecar: SceneCookSidecar
    collision_spec: CollisionSpec
    entities: tuple[EntityCookPlan, ...] = ()
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def has_entities(self) -> bool:
        return bool(self.entities)

    def entities_metadata(self) -> list[dict[str, Any]]:
        return [entity.to_metadata() for entity in self.entities]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "alignment": {
                "scale": self.alignment.scale,
                "rotation_zyx_deg": list(self.alignment.rotation_zyx_deg),
                "translation": list(self.alignment.translation),
                "y_up": self.alignment.y_up,
            },
            "sidecar_path": str(self.sidecar.path) if self.sidecar.path else None,
            "entities": [entity.to_json_dict() for entity in self.entities],
            "stats": self.stats,
        }


def build_scene_cook_plan(
    source_path: str | Path,
    *,
    sidecar: SceneCookSidecar,
    alignment: SceneMeshAlignment,
    output_dir: str | Path,
    collision_spec: CollisionSpec | None = None,
) -> SceneCookPlan:
    source = Path(source_path).expanduser().resolve()
    base_collision = collision_spec or sidecar.collision
    if not sidecar.interactables:
        return SceneCookPlan(
            source_path=source,
            alignment=alignment,
            sidecar=sidecar,
            collision_spec=base_collision,
            stats={"source_prims": 0, "entities": 0},
        )

    prims = load_scene_prims(source, alignment=alignment)
    entities_dir = Path(output_dir).expanduser().resolve() / "entities"
    entities = tuple(
        _build_entity_plan(item, prims, entities_dir) for item in sidecar.interactables
    )
    effective_collision = _collision_spec_with_entity_skips(base_collision, entities)
    return SceneCookPlan(
        source_path=source,
        alignment=alignment,
        sidecar=sidecar,
        collision_spec=effective_collision,
        entities=entities,
        stats={"source_prims": len(prims), "entities": len(entities)},
    )


def _build_entity_plan(
    spec: InteractableSpec,
    prims: list[ScenePrimMesh],
    entities_dir: Path,
) -> EntityCookPlan:
    matched = sorted(
        (prim for prim in prims if spec.matches(prim)),
        key=_prim_sort_key,
    )
    if not matched:
        patterns = ", ".join(spec.source_prim_paths)
        raise ValueError(f"scene interactable {spec.id!r} matched no source prims: {patterns}")

    vertices = np.concatenate([prim.vertices for prim in matched], axis=0)
    aabb_min_np = vertices.min(axis=0).astype(float)
    aabb_max_np = vertices.max(axis=0).astype(float)
    center_np = ((aabb_min_np + aabb_max_np) * 0.5).astype(float)
    extents = np.maximum(aabb_max_np - aabb_min_np, 1e-4).astype(float)
    safe_id = _safe_entity_id(spec.id)
    visual_path = entities_dir / safe_id / "visual.glb"

    shape_hint = str(spec.physics.get("shape", "box"))
    shape_extents = spec.physics.get("extents")
    if shape_extents is None and shape_hint == "box":
        shape_extents = extents.tolist()
    elif shape_extents is None and shape_hint == "sphere":
        shape_extents = [float(max(extents) * 0.5)]
    elif shape_extents is None and shape_hint == "cylinder":
        shape_extents = [float(max(extents[0], extents[1]) * 0.5), float(extents[2])]
    elif shape_extents is None:
        shape_extents = []

    descriptor = {
        "entity_id": spec.id,
        "kind": spec.kind,
        "mesh_ref": f"entities/{safe_id}/visual.glb",
        "shape_hint": shape_hint,
        "extents": [float(value) for value in shape_extents],
        "mass": float(spec.mass),
    }

    return EntityCookPlan(
        spec=spec,
        safe_id=safe_id,
        matched_prim_paths=tuple(prim.prim_path or prim.name for prim in matched),
        visual_node_patterns=_visual_node_patterns(matched),
        aabb_min=tuple(float(value) for value in aabb_min_np),
        aabb_max=tuple(float(value) for value in aabb_max_np),
        center=tuple(float(value) for value in center_np),
        descriptor=descriptor,
        visual_path=visual_path,
    )


def _visual_node_patterns(prims: list[ScenePrimMesh]) -> tuple[str, ...]:
    names: list[str] = []
    for prim in prims:
        prim_path = prim.visual_node_name or prim.prim_path or prim.name
        basename = prim_path.lstrip("/").rsplit("/", 1)[-1]
        visual_name = _HASH_SUFFIX_RE.sub("", basename)
        if visual_name not in names:
            names.append(visual_name)
    return tuple(names)


def _collision_spec_with_entity_skips(
    collision_spec: CollisionSpec,
    entities: tuple[EntityCookPlan, ...],
) -> CollisionSpec:
    prim_overrides: dict[str, dict[str, Any]] = dict(collision_spec.prim_overrides)
    for entity in entities:
        if not entity.spec.remove_from_static:
            continue
        for prim_path in sorted(entity.matched_prim_paths):
            prim_overrides.setdefault(prim_path, {"type": "skip"})
    return replace(collision_spec, prim_overrides=prim_overrides)


def _prim_sort_key(prim: ScenePrimMesh) -> tuple[str, str]:
    return (prim.prim_path or prim.name, prim.name)


def _safe_entity_id(entity_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in entity_id)
    return safe or "entity"


__all__ = ["EntityCookPlan", "SceneCookPlan", "build_scene_cook_plan"]
