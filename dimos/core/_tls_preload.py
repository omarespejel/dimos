"""
This exists because of a linux ARM problem, which shows up on the G1

On arm we need to preload libraries that use static TLS storage so they get slots before
the static TLS block fills up. There's a number of these libraries (PyTorch's libc10, sklearn's vendored libgomp, libgomp
generally) declare static TLS. On Linux aarch64 if these libraries get loaded
lazily — after other libraries already consumed the block — dlopen fails:

    cannot allocate memory in static TLS block
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path

_done = False


def _try_load(path: str) -> None:
    try:
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass


def _sklearn_bundled_libgomp() -> list[str]:
    spec = importlib.util.find_spec("sklearn")
    if spec is None or spec.origin is None:
        return []
    libs_dir = Path(spec.origin).parent.parent / "scikit_learn.libs"
    if not libs_dir.is_dir():
        return []
    return [str(path) for path in libs_dir.glob("libgomp-*.so*")]


def _torch_libs() -> list[str]:
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        return []
    lib_dir = Path(spec.origin).parent / "lib"
    return [str(lib_dir / name) for name in ("libc10.so", "libtorch.so", "libtorch_cpu.so")]


def _system_tls_libs() -> list[str]:
    return [
        "/lib/aarch64-linux-gnu/libgomp.so.1",
        "/lib/aarch64-linux-gnu/libGLdispatch.so.0",
    ]


def preload_tls_libs() -> None:
    """Idempotent. Safe to call from both parent and worker processes."""
    global _done
    if _done:
        return
    _done = True
    for path in _sklearn_bundled_libgomp() + _torch_libs() + _system_tls_libs():
        if os.path.exists(path):
            _try_load(path)
