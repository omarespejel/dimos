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

from threading import Event, RLock
from unittest.mock import MagicMock, call, patch

from dimos.navigation.replanning_a_star.global_planner import GlobalPlanner


def _planner_without_dependencies() -> GlobalPlanner:
    planner = object.__new__(GlobalPlanner)
    planner._lock = RLock()
    planner._stop_planner = Event()
    planner._current_odom = None
    planner._current_goal = None
    return planner


def test_replan_without_active_goal_is_noop() -> None:
    planner = _planner_without_dependencies()
    planner._current_odom = MagicMock()
    planner._replan_limiter = MagicMock()

    planner._replan_path()

    planner._replan_limiter.get_attempt.assert_not_called()


def test_plan_path_cancelled_before_snapshot_is_noop() -> None:
    planner = _planner_without_dependencies()
    planner._current_odom = MagicMock()

    with (
        patch.object(planner, "cancel_goal") as cancel_goal,
        patch.object(planner, "_find_safe_goal") as find_safe_goal,
    ):
        planner._plan_path()

    cancel_goal.assert_called_once_with(but_will_try_again=True)
    find_safe_goal.assert_not_called()


def test_plan_path_during_shutdown_does_not_cancel_again() -> None:
    planner = _planner_without_dependencies()
    planner._stop_planner.set()

    with patch.object(planner, "cancel_goal") as cancel_goal:
        planner._plan_path()

    cancel_goal.assert_not_called()


def test_plan_path_discards_result_if_goal_is_cancelled_while_planning() -> None:
    planner = _planner_without_dependencies()
    planner._current_odom = MagicMock()
    planner._current_goal = MagicMock()
    planner.path = MagicMock()
    planner._local_planner = MagicMock()

    def cancel_while_planning(*_args: object) -> MagicMock:
        planner._current_goal = None
        return MagicMock()

    with (
        patch.object(planner, "cancel_goal"),
        patch.object(planner, "_find_safe_goal", return_value=MagicMock()),
        patch.object(planner, "_find_wide_path", side_effect=cancel_while_planning),
        patch(
            "dimos.navigation.replanning_a_star.global_planner.smooth_resample_path",
            return_value=MagicMock(),
        ),
    ):
        planner._plan_path()

    planner.path.on_next.assert_not_called()
    planner._local_planner.start_planning.assert_not_called()


def test_stopped_navigating_is_ignored_during_shutdown() -> None:
    planner = _planner_without_dependencies()
    planner._replan_event = Event()
    planner._replan_reason = None
    planner._stop_planner.set()

    planner._on_stopped_navigating("error")

    assert planner._replan_reason is None
    assert not planner._replan_event.is_set()


def test_stop_blocks_replanning_before_stopping_local_planner() -> None:
    planner = _planner_without_dependencies()
    lifecycle = MagicMock()
    planner._stop_planner = Event()
    planner._replan_event = Event()
    planner._thread = None
    planner._disposables = MagicMock()
    planner._local_planner = MagicMock()

    lifecycle.attach_mock(planner._disposables.dispose, "dispose")
    lifecycle.attach_mock(planner._local_planner.stop, "local_stop")
    with patch.object(planner, "cancel_goal") as cancel_goal:
        lifecycle.attach_mock(cancel_goal, "cancel_goal")
        planner.stop()

    assert planner._stop_planner.is_set()
    assert planner._replan_event.is_set()
    assert lifecycle.mock_calls == [call.dispose(), call.cancel_goal(), call.local_stop()]
