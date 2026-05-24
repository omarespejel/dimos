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

from __future__ import annotations

from dataclasses import dataclass, field, fields
import sys
import threading
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Callable

    from dimos.models.embedding.base import Embedding
    from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped

T = TypeVar("T")
R = TypeVar("R")


class _Unloaded:
    """Sentinel indicating data has not been loaded yet."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<unloaded>"


_UNLOADED = _Unloaded()


def _normalize_pose(p: Any) -> tuple[float, float, float, float, float, float, float] | None:
    """Coerce common pose shapes to the storage 7-tuple `(x, y, z, qx, qy, qz, qw)`.

    Accepts: `None`, an already-normalized tuple, or any object with
    `translation`+`rotation` (Transform) or `position`+`orientation`
    (Pose / PoseStamped) attributes. Duck-typed to avoid importing
    `dimos.msgs.geometry_msgs` at module load.
    """
    if p is None or isinstance(p, tuple):
        return p
    trans = getattr(p, "translation", None) or getattr(p, "position", None)
    rot = getattr(p, "rotation", None) or getattr(p, "orientation", None)
    if trans is None or rot is None:
        raise TypeError(
            f"Cannot coerce {type(p).__name__} to a pose tuple — expected "
            "tuple, None, Transform, Pose, or PoseStamped."
        )
    return (
        float(trans.x),
        float(trans.y),
        float(trans.z),
        float(rot.x),
        float(rot.y),
        float(rot.z),
        float(rot.w),
    )


@dataclass
class Observation(Generic[T]):
    """A single timestamped observation with optional spatial pose and metadata.

    `pose` is stored as a 7-tuple `(x, y, z, qx, qy, qz, qw)`. Passing a
    `Transform`, `Pose`, or `PoseStamped` to the constructor or to `derive`
    is normalized to the 7-tuple form automatically.
    """

    id: int
    ts: float
    data_type: type = object
    pose: Any | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    _data: T | _Unloaded = field(default=_UNLOADED, repr=False)
    _loader: Callable[[], T] | None = field(default=None, repr=False)
    _data_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.pose = _normalize_pose(self.pose)

    @property
    def pose_stamped(self) -> PoseStamped:
        from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped

        if self.pose is None:
            raise LookupError("No pose set on this observation")
        x, y, z, qx, qy, qz, qw = self.pose
        ps: PoseStamped = PoseStamped(ts=self.ts, position=(x, y, z), orientation=(qx, qy, qz, qw))
        return ps

    @property
    def data(self) -> T:
        val = self._data
        if isinstance(val, _Unloaded):
            with self._data_lock:
                # Re-check after acquiring lock (double-checked locking)
                val = self._data
                if isinstance(val, _Unloaded):
                    if self._loader is None:
                        raise LookupError("No data and no loader set on this observation")
                    loaded = self._loader()
                    self._data = loaded
                    self._loader = None  # release closure
                    return loaded
            return val
        return val

    def derive(self, *, data: R, **overrides: Any) -> Observation[R]:
        """New observation with replaced ``data``; other fields carry over.

        Passing ``embedding`` on a plain :class:`Observation` promotes it to
        :class:`EmbeddedObservation`.
        """
        cls: type[Observation[Any]] = (
            EmbeddedObservation
            if "embedding" in overrides and not isinstance(self, EmbeddedObservation)
            else type(self)
        )
        kwargs: dict[str, Any] = {f.name: getattr(self, f.name) for f in fields(self)}
        kwargs.update(overrides)
        kwargs.update(data_type=type(data), _data=data, _loader=None, _data_lock=threading.Lock())
        return cast("Observation[R]", cls(**kwargs))

    def tag(self, **tags: Any) -> Self:
        """Return a new observation with tags merged in."""
        kwargs: dict[str, Any] = {f.name: getattr(self, f.name) for f in fields(self)}
        kwargs.update(
            tags={**self.tags, **tags},
            _data=_UNLOADED,
            _loader=lambda: self.data,
            _data_lock=threading.Lock(),
        )
        return type(self)(**kwargs)


@dataclass
class EmbeddedObservation(Observation[T]):
    """Observation enriched with a vector embedding and optional similarity score."""

    embedding: Embedding | None = None
    similarity: float | None = None
