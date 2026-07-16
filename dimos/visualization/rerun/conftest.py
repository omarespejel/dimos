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

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager

import pytest

from dimos.core.global_config import global_config
from dimos.visualization.rerun.websocket_server import RerunWebSocketServer

WebSocketServerFactory = Callable[[int], AbstractContextManager[RerunWebSocketServer]]


@pytest.fixture()
def websocket_server_factory() -> WebSocketServerFactory:
    """Create a running WebSocket server while preserving global port config."""

    @contextmanager
    def running_server(port: int) -> Generator[RerunWebSocketServer, None, None]:
        original_port = global_config.rerun_websocket_server_port
        global_config.update(rerun_websocket_server_port=port)
        module: RerunWebSocketServer | None = None
        try:
            module = RerunWebSocketServer()
            module.start()
            yield module
        finally:
            try:
                if module is not None:
                    module.stop()
            finally:
                global_config.update(rerun_websocket_server_port=original_port)

    return running_server


@pytest.fixture()
def server(
    unused_tcp_port: int,
    websocket_server_factory: WebSocketServerFactory,
) -> Generator[RerunWebSocketServer, None, None]:
    """Run the WebSocket server on a per-test port safe for parallel workers."""
    with websocket_server_factory(unused_tcp_port) as module:
        yield module
