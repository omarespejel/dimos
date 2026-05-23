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

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dimos.simulation.scene_assets import plan as plan_module
from dimos.simulation.scene_assets.mesh_scene import SceneMeshAlignment, ScenePrimMesh
from dimos.simulation.scene_assets.sidecar import SceneCookSidecar
from dimos.simulation.scene_assets.spec import ARTIFACT_FRAMES, load_scene_package


def _metadata(tmp_path: Path) -> dict[str, object]:
    return {
        "source_path": str(tmp_path / "source.glb"),
        "package_dir": str(tmp_path),
        "alignment": {
            "scale": 1.0,
            "rotation_zyx_deg": [0.0, 0.0, 0.0],
            "translation": [0.0, 0.0, 0.0],
            "y_up": True,
        },
        "artifact_frames": ARTIFACT_FRAMES,
        "artifacts": {
            "browser_visual": str(tmp_path / "visual.glb"),
            "browser_collision": str(tmp_path / "collision.glb"),
            "mujoco_model": str(tmp_path / "compiled.mjb"),
            "mujoco_wrapper": str(tmp_path / "wrapper.xml"),
        },
        "stats": {},
    }


def test_load_scene_package_rejects_missing_artifact_frames(tmp_path: Path) -> None:
    raw = _metadata(tmp_path)
    raw.pop("artifact_frames")
    metadata_path = tmp_path / "scene.meta.json"
    metadata_path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="missing artifact frame metadata"):
        load_scene_package(metadata_path)


def test_load_scene_package_rejects_mismatched_artifact_frames(tmp_path: Path) -> None:
    raw = _metadata(tmp_path)
    raw["artifact_frames"] = {
        "browser_visual": "dimos_world",
        "browser_collision": "source",
        "mujoco": "dimos_world",
    }
    metadata_path = tmp_path / "scene.meta.json"
    metadata_path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="artifact frame mismatch"):
        load_scene_package(metadata_path)


def test_load_scene_package_accepts_expected_artifact_frames(tmp_path: Path) -> None:
    metadata_path = tmp_path / "scene.meta.json"
    metadata_path.write_text(json.dumps(_metadata(tmp_path)))

    package = load_scene_package(metadata_path)

    assert package.visual_path == tmp_path / "visual.glb"
    assert package.browser_collision_path == tmp_path / "collision.glb"
    assert package.mujoco_model_path == tmp_path / "compiled.mjb"


def test_load_scene_package_preserves_packaged_entities(tmp_path: Path) -> None:
    raw = _metadata(tmp_path)
    raw["entities"] = [
        {
            "id": "chair_016",
            "descriptor": {"entity_id": "chair_016", "shape_hint": "box"},
        }
    ]
    metadata_path = tmp_path / "scene.meta.json"
    metadata_path.write_text(json.dumps(raw))

    package = load_scene_package(metadata_path)

    assert package.entities[0]["id"] == "chair_016"


def test_scene_cook_plan_maps_collision_prims_to_blender_visual_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_load_scene_prims(
        path: str | Path,
        alignment: SceneMeshAlignment | None = None,
    ) -> list[ScenePrimMesh]:
        del path, alignment
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        return [
            ScenePrimMesh(
                name="Chair_seat",
                prim_path="Chair_a1b2c3",
                vertices=np.array(
                    [[-1.0, -1.0, 0.2], [-0.5, -1.0, 0.2], [-1.0, -0.5, 0.8]],
                    dtype=np.float32,
                ),
                triangles=triangles,
            ),
            ScenePrimMesh(
                name="Chair.016_seat",
                prim_path="Chair.016_a1b2c3",
                vertices=np.array(
                    [[1.0, 2.0, 0.2], [2.0, 2.0, 0.2], [1.0, 3.0, 0.8]],
                    dtype=np.float32,
                ),
                triangles=triangles,
            ),
            ScenePrimMesh(
                name="Chair.016_back",
                prim_path="Chair.016_d4e5f6",
                vertices=np.array(
                    [[1.0, 2.0, 0.8], [2.0, 3.0, 1.4], [1.5, 2.5, 1.2]],
                    dtype=np.float32,
                ),
                triangles=triangles,
            ),
        ]

    monkeypatch.setattr(plan_module, "load_scene_prims", fake_load_scene_prims)
    sidecar = SceneCookSidecar.from_dict(
        {
            "interactables": [
                {
                    "id": "chair_000",
                    "source_prim_paths": ["Chair_*"],
                    "physics": {"shape": "box"},
                },
                {
                    "id": "chair_016",
                    "source_prim_paths": ["Chair.016_*"],
                    "physics": {"shape": "box"},
                },
            ]
        }
    )

    plan = plan_module.build_scene_cook_plan(
        tmp_path / "office.glb",
        sidecar=sidecar,
        alignment=SceneMeshAlignment(scale=2.0, y_up=False),
        output_dir=tmp_path,
    )

    base_entity = plan.entities[0]
    assert base_entity.matched_prim_paths == ("Chair_a1b2c3",)
    assert base_entity.visual_node_patterns == ("Chair",)
    assert base_entity.descriptor["mesh_ref"] == "entities/chair_000/visual.glb"

    entity = plan.entities[1]
    assert entity.matched_prim_paths == ("Chair.016_a1b2c3", "Chair.016_d4e5f6")
    assert entity.visual_node_patterns == ("Chair.016",)
    assert entity.descriptor["mesh_ref"] == "entities/chair_016/visual.glb"
    assert plan.collision_spec.resolve("Chair_a1b2c3")["type"] == "skip"
    assert plan.collision_spec.resolve("Chair.016_a1b2c3")["type"] == "skip"
    assert plan.collision_spec.resolve("Chair.001_a1b2c3")["type"] == "auto"
