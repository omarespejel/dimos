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


from dimos.agents.capabilities import CapabilityRegistry


def test_acquire_then_release():
    reg = CapabilityRegistry()
    assert reg.acquire(["movement"], holder="start_patrol") is None
    assert reg.snapshot() == {"movement": "start_patrol"}
    released = reg.release_by_holder("start_patrol")
    assert released == ["movement"]
    assert reg.snapshot() == {}


def test_acquire_conflict_reports_existing_holder():
    reg = CapabilityRegistry()
    reg.acquire(["movement"], holder="start_patrol")
    conflict = reg.acquire(["movement"], holder="follow_person")
    assert conflict == ("movement", "start_patrol")
    # State is unchanged after a refused acquire.
    assert reg.snapshot() == {"movement": "start_patrol"}


def test_acquire_multi_cap_is_atomic_on_conflict():
    """If any cap conflicts, no caps are acquired (all-or-nothing)."""
    reg = CapabilityRegistry()
    reg.acquire(["audio"], holder="speak")
    conflict = reg.acquire(["movement", "audio"], holder="multi")
    assert conflict == ("audio", "speak")
    # `movement` must NOT have leaked in.
    assert "movement" not in reg.snapshot()


def test_same_holder_reacquire_is_noop():
    """Re-entering acquire with the same holder must succeed without conflict.

    This matters because the McpServer doesn't know whether a `tools/call`
    is a fresh request or the LLM re-issuing the same call while a previous
    background invocation is still active.
    """
    reg = CapabilityRegistry()
    reg.acquire(["movement"], holder="start_patrol")
    assert reg.acquire(["movement"], holder="start_patrol") is None
    assert reg.snapshot() == {"movement": "start_patrol"}


def test_release_by_holder_only_releases_matching():
    reg = CapabilityRegistry()
    reg.acquire(["movement"], holder="A")
    reg.acquire(["audio"], holder="B")
    released = reg.release_by_holder("A")
    assert released == ["movement"]
    assert reg.snapshot() == {"audio": "B"}


def test_release_caps_drops_named_caps():
    reg = CapabilityRegistry()
    reg.acquire(["movement", "audio"], holder="X")
    reg.release_caps(["audio"])
    assert reg.snapshot() == {"movement": "X"}


def test_release_by_unknown_holder_is_noop():
    reg = CapabilityRegistry()
    reg.acquire(["movement"], holder="A")
    assert reg.release_by_holder("nobody") == []
    assert reg.snapshot() == {"movement": "A"}