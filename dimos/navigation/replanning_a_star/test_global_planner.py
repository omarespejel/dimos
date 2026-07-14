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

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any, cast
from unittest.mock import MagicMock, call

import pytest
from pytest_mock import MockerFixture

from dimos.core.global_config import GlobalConfig
from dimos.navigation.replanning_a_star import global_planner as planner_module
from dimos.navigation.replanning_a_star.global_planner import GlobalPlanner


@pytest.fixture()
def planner(mocker: MockerFixture) -> GlobalPlanner:
    mocker.patch.object(planner_module, "NavigationMap", autospec=True)
    mocker.patch.object(planner_module, "LocalPlanner", autospec=True)
    mocker.patch.object(planner_module, "PositionTracker", autospec=True)
    mocker.patch.object(planner_module, "ReplanLimiter", autospec=True)
    return GlobalPlanner(GlobalConfig())


def test_replan_without_active_goal_is_noop(planner: GlobalPlanner) -> None:
    planner._current_odom = MagicMock()

    planner._replan_path()

    cast("MagicMock", planner._replan_limiter.get_attempt).assert_not_called()


def test_plan_path_cancelled_before_snapshot_is_noop(
    planner: GlobalPlanner, mocker: MockerFixture
) -> None:
    planner._current_odom = MagicMock()
    cancel_goal = mocker.patch.object(planner, "cancel_goal")
    find_safe_goal = mocker.patch.object(planner, "_find_safe_goal")

    planner._plan_path()

    cancel_goal.assert_called_once_with(but_will_try_again=True)
    find_safe_goal.assert_not_called()


def test_plan_path_during_shutdown_does_not_cancel_again(
    planner: GlobalPlanner, mocker: MockerFixture
) -> None:
    planner._stop_planner.set()
    cancel_goal = mocker.patch.object(planner, "cancel_goal")

    planner._plan_path()

    cancel_goal.assert_not_called()


def test_goal_request_during_shutdown_is_ignored(
    planner: GlobalPlanner, mocker: MockerFixture
) -> None:
    planner._stop_planner.set()
    goal = MagicMock()
    plan_path = mocker.patch.object(planner, "_plan_path")

    planner.handle_goal_request(goal)

    assert planner._current_goal is None
    cast("MagicMock", planner._replan_limiter.reset).assert_not_called()
    plan_path.assert_not_called()


def test_plan_path_discards_result_if_goal_is_cancelled_while_planning(
    planner: GlobalPlanner, mocker: MockerFixture
) -> None:
    planner._current_odom = MagicMock()
    planner._current_goal = MagicMock()
    planner.path = MagicMock()
    mocker.patch.object(planner, "cancel_goal")
    mocker.patch.object(planner, "_find_safe_goal", return_value=MagicMock())

    def cancel_while_planning(*_args: Any) -> MagicMock:
        planner._current_goal = None
        return MagicMock()

    mocker.patch.object(planner, "_find_wide_path", side_effect=cancel_while_planning)
    mocker.patch.object(planner_module, "smooth_resample_path", return_value=MagicMock())

    planner._plan_path()

    planner.path.on_next.assert_not_called()
    cast("MagicMock", planner._local_planner.start_planning).assert_not_called()


def test_plan_path_discards_result_cancelled_by_path_subscriber(
    planner: GlobalPlanner, mocker: MockerFixture
) -> None:
    planner._current_odom = MagicMock()
    planner._current_goal = MagicMock()
    resampled_path = MagicMock()
    planner.path = MagicMock()
    mocker.patch.object(planner, "_find_safe_goal", return_value=MagicMock())
    mocker.patch.object(planner, "_find_wide_path", return_value=MagicMock())
    mocker.patch.object(planner_module, "smooth_resample_path", return_value=resampled_path)

    def cancel_on_path(path: Any) -> None:
        if path is resampled_path:
            planner.cancel_goal()

    planner.path.on_next.side_effect = cancel_on_path

    planner._plan_path()

    cast("MagicMock", planner._local_planner.start_planning).assert_not_called()


def test_cancel_cannot_finish_before_inflight_path_activation(
    planner: GlobalPlanner, mocker: MockerFixture
) -> None:
    planner._current_odom = MagicMock()
    planner._current_goal = MagicMock()
    resampled_path = MagicMock()
    activation_started = Event()
    release_activation = Event()
    cancel_started = Event()
    cancel_finished = Event()
    planner.path = MagicMock()
    mocker.patch.object(planner, "_find_safe_goal", return_value=MagicMock())
    mocker.patch.object(planner, "_find_wide_path", return_value=MagicMock())
    mocker.patch.object(planner_module, "smooth_resample_path", return_value=resampled_path)

    def pause_activation(path: Any) -> None:
        if path is resampled_path:
            activation_started.set()
            release_activation.wait(timeout=1.0)

    def cancel() -> None:
        cancel_started.set()
        planner.cancel_goal()
        cancel_finished.set()

    planner.path.on_next.side_effect = pause_activation

    with ThreadPoolExecutor(max_workers=2) as executor:
        plan_future = executor.submit(planner._plan_path)
        activation_was_reached = activation_started.wait(timeout=1.0)
        start_planning = cast("MagicMock", planner._local_planner.start_planning)
        stop_planning = cast("MagicMock", planner._local_planner.stop_planning)
        start_planning.reset_mock()
        stop_planning.reset_mock()
        lifecycle = MagicMock()
        lifecycle.attach_mock(start_planning, "start")
        lifecycle.attach_mock(stop_planning, "stop")
        cancel_future = executor.submit(cancel)
        cancel_was_started = cancel_started.wait(timeout=1.0)
        cancel_finished_before_release = cancel_finished.wait(timeout=0.2)
        release_activation.set()
        plan_future.result(timeout=1.0)
        cancel_future.result(timeout=1.0)

    assert activation_was_reached
    assert cancel_was_started
    assert not cancel_finished_before_release
    assert lifecycle.mock_calls == [call.start(resampled_path), call.stop()]


def test_stopped_navigating_is_ignored_during_shutdown(planner: GlobalPlanner) -> None:
    planner._replan_reason = None
    planner._stop_planner.set()

    planner._on_stopped_navigating("error")

    assert planner._replan_reason is None
    assert not planner._replan_event.is_set()


def test_stop_blocks_replanning_before_stopping_local_planner(
    planner: GlobalPlanner, mocker: MockerFixture
) -> None:
    lifecycle = MagicMock()
    planner._thread = None
    planner._disposables = MagicMock()
    cancel_goal = mocker.patch.object(planner, "cancel_goal")

    def assert_monitor_stopped() -> None:
        assert planner._stop_planner.is_set()
        assert planner._replan_event.is_set()

    planner._disposables.dispose.side_effect = assert_monitor_stopped
    lifecycle.attach_mock(planner._disposables.dispose, "dispose")
    lifecycle.attach_mock(cancel_goal, "cancel_goal")
    lifecycle.attach_mock(cast("MagicMock", planner._local_planner.stop), "local_stop")

    planner.stop()

    assert lifecycle.mock_calls == [call.dispose(), call.cancel_goal(), call.local_stop()]
