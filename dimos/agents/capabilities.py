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

"""Capability registry for skill-level mutual exclusion.

A skill declares the capabilities it occupies via `@skill(uses=[...])`. The MCP
server consults a process-wide `CapabilityRegistry` before dispatching each
`tools/call` and refuses calls that would conflict with another active skill.
The LLM sees a plain text "Cannot start X: capability Y is held by Z" result
and decides what to do (typically: call the appropriate stop tool, then retry).

Capabilities are plain strings. Today the only declared capability is
`CAP_MOVEMENT`. New capabilities should be added as constants here so they are
discoverable from one place.
"""

from __future__ import annotations

import threading

CAP_MOVEMENT = "movement"


class CapabilityRegistry:
    """In-memory map of capability -> holder skill name.

    All methods are thread-safe. The registry is intentionally simple: every
    capability is exclusive (no shared/exclusive distinction yet) and the only
    conflict policy is "refuse". On conflict, the caller is told who holds
    what and is expected to ask for a release explicitly.
    """

    def __init__(self) -> None:
        self._holders: dict[str, str] = {}
        self._lock = threading.Lock()

    def acquire(self, caps: list[str], holder: str) -> tuple[str, str] | None:
        """Atomic all-or-nothing acquire of `caps` for `holder`.

        Returns `None` on success. On conflict, returns `(cap, current_holder)`
        for the first conflicting capability; no caps are acquired in that case.
        """
        with self._lock:
            for cap in caps:
                current = self._holders.get(cap)
                if current is not None and current != holder:
                    return (cap, current)
            for cap in caps:
                self._holders[cap] = holder
        return None

    def release_by_holder(self, holder: str) -> list[str]:
        """Release every capability currently held by `holder`."""
        with self._lock:
            released = [cap for cap, h in self._holders.items() if h == holder]
            for cap in released:
                del self._holders[cap]
        return released

    def release_caps(self, caps: list[str]) -> None:
        with self._lock:
            for cap in caps:
                self._holders.pop(cap, None)

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return dict(self._holders)
