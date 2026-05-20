// Copyright 2025-2026 Dimensional Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const wsMsgCamera = 0x01;
const wsMsgPointcloud = 0x02;
const wsMsgRobotPose = 0x03;
const pointcloudHeaderBytes = 8;
const robotPoseHeaderBytes = 16;

let socket = null;
let reconnectTimer = null;

function postStatus(status, ready) {
  self.postMessage({ type: "status", status, ready });
}

function websocketUrl() {
  const protocol = self.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${self.location.host}/ws`;
}

function connect() {
  if (reconnectTimer !== null) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  socket = new WebSocket(websocketUrl());
  socket.binaryType = "arraybuffer";
  socket.onopen = () => postStatus("live", true);
  socket.onerror = () => postStatus("socket error", false);
  socket.onclose = () => {
    postStatus("reconnecting", false);
    reconnectTimer = setTimeout(connect, 1000);
  };
  socket.onmessage = (event) => {
    if (typeof event.data === "string") {
      try {
        const payload = JSON.parse(event.data);
        self.postMessage({ type: "state", payload });
      } catch (error) {
        self.postMessage({ type: "error", message: String(error) });
      }
      return;
    }
    handleBinaryMessage(event.data);
  };
}

function handleBinaryMessage(buffer) {
  const view = new DataView(buffer);
  const msgType = view.getUint8(0);

  if (msgType === wsMsgRobotPose) {
    if (buffer.byteLength < robotPoseHeaderBytes) return;
    const count = view.getUint32(4, false);
    const poseLength = count * 7;
    const poseByteLength = poseLength * 4;
    if (buffer.byteLength < robotPoseHeaderBytes + poseByteLength) return;
    self.postMessage(
      {
        type: "robot_pose",
        count,
        time: view.getFloat64(8, false),
        buffer,
      },
      [buffer],
    );
    return;
  }

  if (msgType === wsMsgPointcloud) {
    if (buffer.byteLength < pointcloudHeaderBytes) return;
    const count = view.getUint32(4, false);
    const positionLength = count * 3;
    const colorOffset = pointcloudHeaderBytes + positionLength * 4;
    if (buffer.byteLength < colorOffset + positionLength) return;
    self.postMessage({ type: "pointcloud", count, buffer }, [buffer]);
    return;
  }

  if (msgType !== wsMsgCamera) return;
  const nameLength = view.getUint16(1, false);
  const jpegOffset = 3 + nameLength;
  if (buffer.byteLength < jpegOffset) return;
  const nameBytes = new Uint8Array(buffer, 3, nameLength);
  const cameraName = new TextDecoder().decode(nameBytes);
  self.postMessage({ type: "camera", cameraName, jpegOffset, buffer }, [buffer]);
}

self.onmessage = (event) => {
  const message = event.data || {};
  if (message.type === "connect") {
    connect();
    return;
  }
  if (message.type === "send_json") {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    socket.send(JSON.stringify(message.payload));
  }
};
