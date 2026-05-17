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

"""Central catalog of artist-built scene assets used by the simulators."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from dimos.utils.data import get_data


@dataclass(frozen=True)
class SceneAsset:
    """A scene mesh plus how to align it in the world.

    ``floor_z`` is the world-frame z of the floor surface after alignment —
    spawn the robot at this z so its feet land on the ground.
    """

    mesh_path: Path
    scale: float
    translation: tuple[float, float, float]
    rotation_zyx_deg: tuple[float, float, float]
    y_up: bool

    @property
    def floor_z(self) -> float:
        return self.translation[2]

    def viewer_kwargs(self) -> dict[str, Any]:
        """Splat into BabylonSceneViewerModule.blueprint(**scene.viewer_kwargs())."""
        return {
            "scene_path": str(self.mesh_path),
            "scene_scale": self.scale,
            "scene_translation": self.translation,
            "scene_rotation_zyx_deg": self.rotation_zyx_deg,
            "scene_y_up": self.y_up,
        }


def get_dimos_office() -> SceneAsset:
    """The artist-built dimos office mesh shipped via LFS.

    Reads alignment from the bundled ``dimos_office_mesh.json``
    (``suggested_alignment`` block, originally derived from the Reference
    splat). Pull happens lazily via ``get_data``.
    """
    data_dir = Path(get_data("dimos_office_mesh"))
    mesh_path = data_dir / "dimos_office_mesh.glb"
    meta_path = data_dir / "dimos_office_mesh.json"
    meta = json.loads(meta_path.read_text())
    alignment = meta["suggested_alignment"]
    tx, ty, tz = alignment["translation"]
    rz, ry, rx = alignment["rotation_zyx_deg"]
    return SceneAsset(
        mesh_path=mesh_path,
        scale=float(alignment["scale"]),
        translation=(float(tx), float(ty), float(tz)),
        rotation_zyx_deg=(float(rz), float(ry), float(rx)),
        y_up=bool(alignment["y_up"]),
    )
