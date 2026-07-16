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

from abc import abstractmethod
import sys
import threading
from typing import TYPE_CHECKING, TypeVar, cast

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from types import TracebackType

from reactivex.abc import DisposableBase
from reactivex.disposable import CompositeDisposable

D = TypeVar("D", bound=DisposableBase)


class Resource(DisposableBase):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    def dispose(self) -> None:
        """
        Makes a Resource disposable
        So you can do a

        from reactivex.disposable import CompositeDisposable

        disposables = CompositeDisposable()

        transport1 = LCMTransport(...)
        transport2 = LCMTransport(...)

        disposables.add(transport1)
        disposables.add(transport2)

        ...

        disposables.dispose()

        """
        self.stop()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exctype: type[BaseException] | None,
        excinst: BaseException | None,
        exctb: TracebackType | None,
    ) -> None:
        self.stop()


class CompositeResource(Resource):
    """Resource that owns child disposables, disposed on stop()."""

    _disposables: CompositeDisposable | None = None
    _disposables_init_lock = threading.Lock()

    def _get_disposables(self) -> CompositeDisposable:
        # Configurable.__init__ is intentionally not cooperative, so several
        # CompositeResource subclasses cannot rely on our __init__ running.
        # Lazily create one per-instance container under a class lock instead.
        # stop() also goes through this path: an empty stop must still be
        # terminal so a later registration is disposed immediately.
        disposables = cast("CompositeDisposable | None", self.__dict__.get("_disposables"))
        if disposables is not None:
            return disposables
        with self._disposables_init_lock:
            disposables = cast("CompositeDisposable | None", self.__dict__.get("_disposables"))
            if disposables is None:
                disposables = CompositeDisposable()
                self._disposables = disposables
            return disposables

    def register_disposable(self, disposable: D) -> D:
        """Register a child disposable to be disposed when this resource stops."""
        disposables = self._get_disposables()
        with disposables.lock:
            if not disposables.is_disposed:
                disposables.disposable.append(disposable)
                return disposable
        self._dispose_robustly(disposable)
        return disposable

    def start(self) -> None: ...

    def stop(self) -> None:
        self._dispose_robustly(self._get_disposables())

    @classmethod
    def _dispose_robustly(cls, disposable: DisposableBase) -> None:
        """Dispose a tree without letting one failing child skip its siblings."""
        if not isinstance(disposable, CompositeDisposable):
            disposable.dispose()
            return

        with disposable.lock:
            if disposable.is_disposed:
                return
            disposable.is_disposed = True
            pending = disposable.disposable
            disposable.disposable = []

        first_error: BaseException | None = None
        for child in pending:
            try:
                cls._dispose_robustly(child)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
