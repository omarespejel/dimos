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

import asyncio
from contextlib import asynccontextmanager
import mimetypes
from pathlib import Path
import threading
import time
from typing import Any, Protocol

import mujoco
import numpy as np
from reactivex.disposable import Disposable
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect
import uvicorn

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Path import Path as PathMsg
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2, _get_colormap_lut
from dimos.spec.utils import Spec
from dimos.utils.logging_config import setup_logger
from dimos.visualization.viser.robot_meshes import (
    RobotMeshes,
    apply_state,
    dimos_joint_to_mjcf,
    load_robot_meshes,
)

logger = setup_logger()

_DEFAULT_BROADCAST_HZ = 20.0
_DEFAULT_PORT = 8091
_DEFAULT_POINTCLOUD_HZ = 2.0
_DEFAULT_POINTCLOUD_MAX_POINTS = 70000
_DEFAULT_CAMERA_HZ = 15.0
_DEFAULT_CAMERA_JPEG_QUALITY = 75
# Binary websocket message tag for a camera frame.
_WS_MSG_CAMERA = 0x01


class MujocoRespawnSpec(Spec, Protocol):
    def respawn(self) -> bool: ...


class HumanoidControlSpec(Spec, Protocol):
    """Optional spec implemented by humanoid robot adapters.

    Auto-wired into the viewer so the per-joint arm sliders in the HUD can
    drive each joint within its declared limits. Stays robot-agnostic — any
    humanoid that knows its own joint range can implement it.
    """

    def set_arm_joint(self, name: str, position: float) -> bool: ...
    def release_arms(self) -> bool: ...
    def arm_joint_limits(self) -> list[tuple[str, float, float]]: ...


class CoordinatorControlSpec(Spec, Protocol):
    """Arm / dry-run knobs on a ControlCoordinator.

    Matches the same RPCs the command center hits — see
    ``WebsocketVisModule._create_server`` ``arm`` / ``disarm`` / ``set_dry_run``
    handlers. Any module exposing these two methods (e.g. ``ControlCoordinator``)
    is auto-wired and unlocks the HUD's Policy toggles.
    """

    def set_activated(self, engaged: bool) -> None: ...
    def set_dry_run(self, enabled: bool) -> None: ...


def _compose_scene_mesh_wxyz(
    *, y_up: bool, rotation_zyx_deg: tuple[float, float, float]
) -> tuple[float, float, float, float]:
    matrix = np.eye(3, dtype=np.float64)
    if y_up:
        matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)

    rz, ry, rx = (np.deg2rad(angle) for angle in rotation_zyx_deg)
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)
    rotate_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    rotate_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rotate_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    matrix = rotate_z @ rotate_y @ rotate_x @ matrix

    out = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(out, matrix.flatten())
    return (float(out[0]), float(out[1]), float(out[2]), float(out[3]))


def _path_contains(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _media_type(path: Path) -> str | None:
    match path.suffix.lower():
        case ".glb":
            return "model/gltf-binary"
        case ".gltf":
            return "model/gltf+json"
        case _:
            return mimetypes.guess_type(path.name)[0]


class BabylonSceneViewerModule(Module):
    joint_state: In[JointState]
    odom: In[PoseStamped]
    path: In[PathMsg]
    pointcloud_overlay: In[PointCloud2]
    camera_image: In[Image]
    clicked_point: Out[PointStamped]
    point_goal: Out[PointStamped]
    cmd_vel: Out[Twist]
    _mujoco_sim: MujocoRespawnSpec | None = None
    _robot_ctrl: HumanoidControlSpec | None = None
    _coordinator_ctrl: CoordinatorControlSpec | None = None

    def __init__(
        self,
        mjcf_path: str | Path,
        *,
        port: int = _DEFAULT_PORT,
        assets: dict[str, bytes] | None = None,
        scene_path: str | Path | None = None,
        scene_scale: float = 1.0,
        scene_translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scene_rotation_zyx_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scene_y_up: bool = True,
        broadcast_hz: float = _DEFAULT_BROADCAST_HZ,
        pointcloud_hz: float = _DEFAULT_POINTCLOUD_HZ,
        pointcloud_max_points: int = _DEFAULT_POINTCLOUD_MAX_POINTS,
        camera_hz: float = _DEFAULT_CAMERA_HZ,
        camera_jpeg_quality: int = _DEFAULT_CAMERA_JPEG_QUALITY,
        camera_name: str = "camera",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._mjcf_path = Path(mjcf_path)
        self._assets = assets
        self._port = port
        self._scene_path = Path(scene_path) if scene_path else None
        self._scene_scale = scene_scale
        self._scene_translation = scene_translation
        self._scene_rotation_zyx_deg = scene_rotation_zyx_deg
        self._scene_y_up = scene_y_up
        self._broadcast_dt = 1.0 / float(broadcast_hz)
        self._pointcloud_min_dt = 1.0 / float(pointcloud_hz)
        self._pointcloud_max_points = pointcloud_max_points
        self._camera_min_dt = 1.0 / float(camera_hz)
        self._camera_jpeg_quality = int(camera_jpeg_quality)
        self._camera_name = camera_name

        self._state_lock = threading.Lock()
        self._latest_joints: dict[str, float] = {}
        self._latest_base_pos: np.ndarray | None = None
        self._latest_base_wxyz: np.ndarray | None = None
        self._latest_path: list[list[float]] = []
        self._pointcloud_lock = threading.Lock()
        self._latest_pointcloud_payload: dict[str, Any] | None = None
        self._last_pointcloud_sent = 0.0

        # Camera state. _turbo_jpeg is lazy-initialised so the viewer still
        # imports cleanly on machines without PyTurboJPEG (it's an optional
        # dep). Only the encode path requires it.
        self._camera_lock = threading.Lock()
        self._last_camera_sent = 0.0
        self._turbo_jpeg: Any = None

        self._robot: RobotMeshes | None = None
        self._uvicorn_server: uvicorn.Server | None = None
        self._server_thread: threading.Thread | None = None
        self._broadcast_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._server_loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[WebSocket] = set()

    @rpc
    def start(self) -> None:
        super().start()

        self._robot = load_robot_meshes(self._mjcf_path, assets=self._assets)
        app = self._create_app()
        config = uvicorn.Config(app, host="0.0.0.0", port=self._port, log_level="warning")
        self._uvicorn_server = uvicorn.Server(config)
        self._server_thread = threading.Thread(
            target=self._uvicorn_server.run,
            name="babylon-viewer-server",
            daemon=True,
        )
        self._server_thread.start()

        self.register_disposable(Disposable(self.joint_state.subscribe(self._on_joint_state)))
        self.register_disposable(Disposable(self.odom.subscribe(self._on_odom)))
        self.register_disposable(Disposable(self.path.subscribe(self._on_path)))
        self.register_disposable(
            Disposable(self.pointcloud_overlay.subscribe(self._on_pointcloud_overlay))
        )
        self.register_disposable(
            Disposable(self.camera_image.subscribe(self._on_camera_image))
        )

        self._broadcast_thread = threading.Thread(
            target=self._broadcast_loop,
            name="babylon-viewer-broadcast",
            daemon=True,
        )
        self._broadcast_thread.start()
        logger.info("Babylon scene viewer: http://localhost:%s/", self._port)

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        if self._broadcast_thread and self._broadcast_thread.is_alive():
            self._broadcast_thread.join(timeout=2.0)
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=2.0)
        super().stop()

    def _create_app(self) -> Starlette:
        @asynccontextmanager
        async def _lifespan(_app: Starlette) -> Any:
            self._server_loop = asyncio.get_running_loop()
            yield

        return Starlette(
            routes=[
                Route("/", self._index),
                Route("/config.json", self._config),
                Route("/robot.json", self._robot_json),
                Route("/arms.json", self._arms_json),
                Route("/assets/{asset_name:path}", self._asset),
                WebSocketRoute("/ws", self._websocket),
            ],
            lifespan=_lifespan,
        )

    async def _arms_json(self, request: Request) -> JSONResponse:
        """Joint-limit catalogue so the page can build sliders with real range."""
        if self._robot_ctrl is None:
            return JSONResponse({"joints": []})
        try:
            limits = self._robot_ctrl.arm_joint_limits()
        except Exception as exc:
            logger.warning("BabylonViewer: arm_joint_limits() failed: %s", exc)
            return JSONResponse({"joints": []})
        joints = [
            {"name": name, "min": float(lo), "max": float(hi)}
            for (name, lo, hi) in limits
        ]
        return JSONResponse({"joints": joints})

    async def _index(self, request: Request) -> HTMLResponse:
        return HTMLResponse(_HTML)

    async def _config(self, request: Request) -> JSONResponse:
        scene_file = None
        scene_bytes = 0
        scene_wxyz = _compose_scene_mesh_wxyz(
            y_up=self._scene_y_up,
            rotation_zyx_deg=self._scene_rotation_zyx_deg,
        )
        if self._scene_path is not None and self._scene_path.exists():
            scene_file = f"scene{self._scene_path.suffix.lower()}"
            scene_bytes = self._scene_path.stat().st_size
        return JSONResponse(
            {
                "sceneFile": scene_file,
                "sceneBytes": scene_bytes,
                "sceneScale": self._scene_scale,
                "scenePosition": list(self._scene_translation),
                "sceneWxyz": list(scene_wxyz),
            }
        )

    async def _robot_json(self, request: Request) -> JSONResponse:
        robot = self._robot
        if robot is None:
            return JSONResponse({"bodyNames": [], "geoms": []})

        geoms: list[dict[str, Any]] = []
        for index, geom in enumerate(robot.geoms):
            geoms.append(
                {
                    "id": index,
                    "body": geom.body_name,
                    "vertices": geom.vertices.astype(np.float32).reshape(-1).tolist(),
                    "indices": geom.faces.astype(np.int32).reshape(-1).tolist(),
                    "position": geom.local_pos.astype(np.float32).tolist(),
                    "wxyz": geom.local_wxyz.astype(np.float32).tolist(),
                    "rgba": [float(value) for value in geom.rgba],
                }
            )
        return JSONResponse({"bodyNames": robot.body_names, "geoms": geoms})

    async def _asset(self, request: Request) -> Response:
        if self._scene_path is None or not self._scene_path.exists():
            return Response("scene asset not configured", status_code=404)

        asset_name = request.path_params["asset_name"]
        scene_asset_name = f"scene{self._scene_path.suffix.lower()}"
        if asset_name == scene_asset_name:
            return FileResponse(self._scene_path, media_type=_media_type(self._scene_path))

        if self._scene_path.suffix.lower() == ".gltf":
            candidate = self._scene_path.parent / asset_name
            if _path_contains(self._scene_path.parent, candidate) and candidate.exists():
                return FileResponse(candidate, media_type=_media_type(candidate))

        return Response("asset not found", status_code=404)

    async def _websocket(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)
        try:
            await websocket.send_json(self._make_state_payload())
            with self._pointcloud_lock:
                pointcloud_payload = self._latest_pointcloud_payload
            if pointcloud_payload is not None:
                await websocket.send_json(pointcloud_payload)
            while True:
                message = await websocket.receive_json()
                self._handle_client_message(message)
        except WebSocketDisconnect:
            pass
        finally:
            self._clients.discard(websocket)

    def _handle_client_message(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        if message_type == "respawn":
            if self._mujoco_sim is not None:
                self._mujoco_sim.respawn()
            self.cmd_vel.publish(Twist.zero())
            return
        if message_type == "cmd_vel":
            twist = self._parse_twist(message)
            if twist is not None:
                self.cmd_vel.publish(twist)
            return
        if message_type == "arm_joint":
            name = message.get("name")
            position = message.get("position")
            if (
                self._robot_ctrl is None
                or not isinstance(name, str)
                or not isinstance(position, (int, float))
            ):
                return
            self._robot_ctrl.set_arm_joint(name, float(position))
            return
        if message_type == "release_arms":
            if self._robot_ctrl is not None:
                self._robot_ctrl.release_arms()
            return
        if message_type == "set_activated":
            engaged = bool(message.get("engaged", False))
            if self._coordinator_ctrl is None:
                logger.warning("BabylonViewer: set_activated requested but no coordinator wired")
                return
            logger.info("BabylonViewer: set_activated=%s", engaged)
            self._coordinator_ctrl.set_activated(engaged=engaged)
            return
        if message_type == "set_dry_run":
            enabled = bool(message.get("enabled", False))
            if self._coordinator_ctrl is None:
                logger.warning("BabylonViewer: set_dry_run requested but no coordinator wired")
                return
            logger.info("BabylonViewer: set_dry_run=%s", enabled)
            self._coordinator_ctrl.set_dry_run(enabled=enabled)
            return
        if message_type not in {"clicked_point", "point_goal"}:
            return
        point = message.get("point")
        if not isinstance(point, list) or len(point) != 3:
            return
        try:
            x, y, z = (float(value) for value in point)
        except (TypeError, ValueError):
            return
        stamped = PointStamped(x=x, y=y, z=z, frame_id="map")
        if message_type == "clicked_point":
            self.clicked_point.publish(stamped)
        else:
            self.point_goal.publish(stamped)

    @staticmethod
    def _parse_twist(message: dict[str, Any]) -> Twist | None:
        linear = message.get("linear", [0.0, 0.0, 0.0])
        angular = message.get("angular", [0.0, 0.0, 0.0])
        if not isinstance(linear, list) or not isinstance(angular, list):
            return None
        if len(linear) != 3 or len(angular) != 3:
            return None
        try:
            return Twist(
                linear=Vector3(*(float(value) for value in linear)),
                angular=Vector3(*(float(value) for value in angular)),
            )
        except (TypeError, ValueError):
            return None

    def _broadcast_loop(self) -> None:
        while not self._stop_event.is_set():
            loop = self._server_loop
            if loop is not None and self._clients:
                payload = self._make_state_payload()
                asyncio.run_coroutine_threadsafe(self._broadcast(payload), loop)
            time.sleep(self._broadcast_dt)

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for websocket in tuple(self._clients):
            try:
                await websocket.send_json(payload)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            self._clients.discard(websocket)

    def _broadcast_from_thread(self, payload: dict[str, Any]) -> None:
        loop = self._server_loop
        if loop is None or not self._clients:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), loop)

    def _make_state_payload(self) -> dict[str, Any]:
        robot = self._robot
        if robot is None:
            return {"type": "state", "time": time.time(), "bodies": [], "path": []}

        with self._state_lock:
            joints = dict(self._latest_joints)
            base_pos = None if self._latest_base_pos is None else self._latest_base_pos.copy()
            base_wxyz = None if self._latest_base_wxyz is None else self._latest_base_wxyz.copy()
            path_points = [point[:] for point in self._latest_path]

        apply_state(
            robot,
            base_pos=base_pos,
            base_wxyz=base_wxyz,
            joint_positions=joints,
        )

        bodies = []
        for body_id, body_name in enumerate(robot.body_names):
            bodies.append(
                {
                    "name": body_name,
                    "position": robot.data.xpos[body_id].astype(np.float32).tolist(),
                    "wxyz": robot.data.xquat[body_id].astype(np.float32).tolist(),
                }
            )
        # Normalise joint names to the short form the slider HUD uses:
        #   * strip a leading "<hwid>/" prefix (e.g. "g1/left_shoulder_pitch"
        #     → "left_shoulder_pitch") for coordinator-style names.
        #   * strip a trailing "_joint" suffix (e.g. "left_shoulder_pitch_joint"
        #     → "left_shoulder_pitch") for URDF-style names.
        def _canon(k: str) -> str:
            if "/" in k:
                k = k.split("/", 1)[1]
            if k.endswith("_joint"):
                k = k[: -len("_joint")]
            return k

        joint_positions = {_canon(k): float(v) for k, v in joints.items()}
        return {
            "type": "state",
            "time": time.time(),
            "bodies": bodies,
            "path": path_points,
            "joints": joint_positions,
        }

    def _on_joint_state(self, msg: JointState) -> None:
        with self._state_lock:
            self._latest_joints = {
                dimos_joint_to_mjcf(name): float(position)
                for name, position in zip(msg.name, msg.position, strict=False)
            }

    def _on_odom(self, msg: PoseStamped) -> None:
        with self._state_lock:
            self._latest_base_pos = np.array([msg.x, msg.y, msg.z], dtype=np.float64)
            self._latest_base_wxyz = np.array(
                [
                    msg.orientation.w,
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                ],
                dtype=np.float64,
            )

    def _on_path(self, msg: PathMsg) -> None:
        with self._state_lock:
            self._latest_path = [[pose.x, pose.y, pose.z] for pose in msg.poses]

    def _on_pointcloud_overlay(self, msg: PointCloud2) -> None:
        now = time.monotonic()
        if now - self._last_pointcloud_sent < self._pointcloud_min_dt:
            return

        payload = self._make_pointcloud_payload(msg)
        if payload is None:
            return

        with self._pointcloud_lock:
            self._latest_pointcloud_payload = payload
            self._last_pointcloud_sent = now
        self._broadcast_from_thread(payload)

    def _on_camera_image(self, msg: Image) -> None:
        # Rate-limit to avoid saturating the websocket with multi-MB frames
        # when the publisher pushes at 30+ Hz.
        now = time.monotonic()
        with self._camera_lock:
            if now - self._last_camera_sent < self._camera_min_dt:
                return
            self._last_camera_sent = now

        try:
            jpeg = self._encode_jpeg(msg)
        except Exception as exc:
            logger.warning("BabylonViewer: camera JPEG encode failed: %s", exc)
            return
        if jpeg is None:
            return

        # Binary frame layout:
        #   byte 0:      _WS_MSG_CAMERA (0x01)
        #   bytes 1-2:   name length (big-endian uint16)
        #   bytes 3..:   utf-8 camera name, then JPEG payload
        name = self._camera_name.encode("utf-8")[:65535]
        header = bytes([_WS_MSG_CAMERA]) + len(name).to_bytes(2, "big") + name
        self._broadcast_bytes_from_thread(header + jpeg)

    def _encode_jpeg(self, msg: Image) -> bytes | None:
        if self._turbo_jpeg is None:
            from turbojpeg import TurboJPEG

            self._turbo_jpeg = TurboJPEG()

        from turbojpeg import TJPF_BGR, TJPF_GRAY, TJPF_RGB

        data = msg.data
        if data is None:
            return None
        match msg.format:
            case ImageFormat.RGB:
                pixel_format = TJPF_RGB
            case ImageFormat.BGR:
                pixel_format = TJPF_BGR
            case ImageFormat.GRAY:
                pixel_format = TJPF_GRAY
            case _:
                # RGBA/BGRA: drop alpha to keep encode cheap.
                if data.ndim == 3 and data.shape[2] == 4:
                    data = data[:, :, :3]
                    pixel_format = TJPF_BGR if msg.format == ImageFormat.BGRA else TJPF_RGB
                else:
                    return None

        encoded: bytes = self._turbo_jpeg.encode(
            np.ascontiguousarray(data),
            quality=self._camera_jpeg_quality,
            pixel_format=pixel_format,
        )
        return encoded

    def _broadcast_bytes_from_thread(self, payload: bytes) -> None:
        loop = self._server_loop
        if loop is None or not self._clients:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast_bytes(payload), loop)

    async def _broadcast_bytes(self, payload: bytes) -> None:
        dead: list[WebSocket] = []
        for websocket in tuple(self._clients):
            try:
                await websocket.send_bytes(payload)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            self._clients.discard(websocket)

    def _make_pointcloud_payload(self, msg: PointCloud2) -> dict[str, Any] | None:
        points = msg.points_f32()
        if points.size == 0:
            return None

        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            return None

        if len(points) > self._pointcloud_max_points:
            indices = np.linspace(0, len(points) - 1, self._pointcloud_max_points).astype(np.int64)
            points = points[indices]

        z_values = points[:, 2]
        if len(z_values) >= 20:
            z_min, z_max = np.quantile(z_values, [0.02, 0.98])
        else:
            z_min, z_max = float(z_values.min()), float(z_values.max())
        normalized = np.clip((z_values - z_min) / (z_max - z_min + 1e-6), 0.0, 1.0)
        colors = _get_colormap_lut("turbo")[(normalized * 255).astype(np.uint8)]

        ambient = np.array([34, 42, 50], dtype=np.float32)
        colors = np.clip(colors.astype(np.float32) * 0.82 + ambient * 0.18, 0, 255).astype(np.uint8)

        return {
            "type": "pointcloud",
            "count": len(points),
            "positions": np.round(points, 3).reshape(-1).tolist(),
            "colors": colors.reshape(-1).tolist(),
        }


_HTML = r"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>DimOS Scene Viewer</title>
    <script src="https://cdn.babylonjs.com/babylon.js"></script>
    <script src="https://cdn.babylonjs.com/loaders/babylonjs.loaders.min.js"></script>
    <style>
      html,
      body,
      #renderCanvas {
        width: 100%;
        height: 100%;
        margin: 0;
        overflow: hidden;
        background: #101216;
        color: #e7ebf2;
        font-family:
          Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
          sans-serif;
      }

      #hud {
        position: fixed;
        left: 16px;
        top: 16px;
        display: flex;
        align-items: stretch;
        flex-wrap: wrap;
        gap: 10px;
        max-width: calc(100vw - 32px);
        padding: 8px 10px;
        border: 1px solid rgb(255 255 255 / 10%);
        border-radius: 10px;
        background: rgb(17 20 26 / 86%);
        backdrop-filter: blur(14px);
        box-shadow: 0 6px 24px rgb(0 0 0 / 32%);
      }

      .hud-group {
        display: flex;
        align-items: center;
        gap: 6px;
        padding-right: 12px;
        border-right: 1px solid rgb(255 255 255 / 8%);
      }

      .hud-group:has(+ #status),
      .hud-group:last-of-type {
        border-right: none;
        padding-right: 0;
      }

      .hud-label {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgb(255 255 255 / 38%);
        margin-right: 2px;
        font-weight: 600;
      }

      button,
      #status {
        height: 28px;
        border: 1px solid rgb(255 255 255 / 14%);
        border-radius: 6px;
        background: rgb(255 255 255 / 6%);
        color: inherit;
        font: inherit;
        font-size: 12px;
        white-space: nowrap;
      }

      button {
        padding: 0 10px;
        cursor: pointer;
        transition: background 0.12s, border-color 0.12s, opacity 0.12s;
      }

      button:hover {
        background: rgb(255 255 255 / 12%);
        border-color: rgb(255 255 255 / 22%);
      }

      button:active {
        transform: translateY(1px);
      }

      button[data-active="true"] {
        background: rgb(96 165 250 / 18%);
        border-color: rgb(96 165 250 / 42%);
        color: rgb(180 210 255);
      }

      button[data-active="false"] {
        opacity: 0.6;
      }

      .hud-segmented {
        display: flex;
        border: 1px solid rgb(255 255 255 / 14%);
        border-radius: 6px;
        overflow: hidden;
      }

      .hud-segmented button {
        height: 28px;
        border: none;
        border-radius: 0;
        border-right: 1px solid rgb(255 255 255 / 8%);
        background: transparent;
      }

      .hud-segmented button:last-child {
        border-right: none;
      }

      .hud-segmented button:hover {
        background: rgb(255 255 255 / 8%);
      }

      .hud-segmented button[data-active="true"] {
        background: rgb(96 165 250 / 22%);
        color: rgb(200 220 255);
      }

      .hud-segmented button[data-active="false"] {
        opacity: 1;       /* segmented controls show all 3, just highlight active */
        color: rgb(255 255 255 / 72%);
      }

      .hud-hidden-by-default {
        display: none;
      }

      #status {
        display: flex;
        align-items: center;
        min-width: 140px;
        padding: 0 12px;
        margin-left: 4px;
        color: rgb(255 255 255 / 75%);
        background: rgb(255 255 255 / 3%);
        border-color: rgb(255 255 255 / 8%);
      }

      #cameraPanel {
        position: fixed;
        right: 16px;
        bottom: 16px;
        width: 360px;
        max-width: calc(100vw - 32px);
        border: 1px solid rgb(255 255 255 / 12%);
        border-radius: 8px;
        background: rgb(17 20 26 / 82%);
        backdrop-filter: blur(10px);
        overflow: hidden;
        display: flex;
        flex-direction: column;
      }

      #cameraPanel[data-active="false"] {
        display: none;
      }

      #cameraHeader {
        padding: 6px 10px;
        font-size: 12px;
        color: rgb(255 255 255 / 70%);
        border-bottom: 1px solid rgb(255 255 255 / 8%);
      }

      #cameraImg {
        width: 100%;
        display: block;
        aspect-ratio: 16 / 9;
        object-fit: cover;
        background: #000;
      }

      #cameraPanel[data-has-frame="false"] #cameraImg {
        display: none;
      }

      #cameraEmpty {
        padding: 30px;
        text-align: center;
        color: rgb(255 255 255 / 50%);
        font-size: 12px;
        aspect-ratio: 16 / 9;
        display: flex;
        align-items: center;
        justify-content: center;
      }

      #cameraPanel[data-has-frame="true"] #cameraEmpty {
        display: none;
      }

      #armsPanel {
        position: fixed;
        right: 16px;
        top: 16px;
        width: 420px;
        max-width: calc(100vw - 32px);
        max-height: calc(100vh - 320px);
        overflow-y: auto;
        padding: 14px 16px 16px 16px;
        border: 1px solid rgb(255 255 255 / 10%);
        border-radius: 10px;
        background: rgb(17 20 26 / 90%);
        backdrop-filter: blur(14px);
        box-shadow: 0 8px 28px rgb(0 0 0 / 40%);
        z-index: 5;
      }

      #armsPanel[data-active="false"] {
        display: none;
      }

      #armsHeader {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: rgb(255 255 255 / 50%);
        margin-bottom: 12px;
        font-weight: 600;
      }

      #armsColumns {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
      }

      .arm-col-title {
        font-size: 12px;
        color: rgb(255 255 255 / 80%);
        margin-bottom: 6px;
        font-weight: 600;
        text-align: center;
        padding: 4px 0;
        background: rgb(255 255 255 / 4%);
        border-radius: 4px;
      }

      .arm-sliders {
        display: flex;
        flex-direction: column;
        gap: 8px;
      }

      .arm-slider-row {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 2px 6px;
        align-items: center;
      }

      .arm-slider-row .joint-name {
        font-size: 11px;
        color: rgb(255 255 255 / 75%);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .arm-slider-row .joint-val {
        font-size: 10px;
        color: rgb(96 165 250 / 90%);
        font-variant-numeric: tabular-nums;
        min-width: 48px;
        text-align: right;
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
      }

      .arm-slider-row input[type="range"] {
        grid-column: 1 / span 2;
        appearance: none;
        -webkit-appearance: none;
        width: 100%;
        height: 4px;
        background: rgb(255 255 255 / 8%);
        border-radius: 2px;
        outline: none;
      }

      .arm-slider-row input[type="range"]::-webkit-slider-thumb {
        appearance: none;
        -webkit-appearance: none;
        width: 12px;
        height: 12px;
        background: rgb(96 165 250);
        border-radius: 50%;
        cursor: pointer;
        border: none;
      }

      .arm-slider-row input[type="range"]::-moz-range-thumb {
        width: 12px;
        height: 12px;
        background: rgb(96 165 250);
        border-radius: 50%;
        cursor: pointer;
        border: none;
      }

      .arm-slider-row .joint-range {
        grid-column: 1 / span 2;
        display: flex;
        justify-content: space-between;
        font-size: 9px;
        color: rgb(255 255 255 / 35%);
        font-variant-numeric: tabular-nums;
      }
    </style>
  </head>
  <body>
    <canvas id="renderCanvas"></canvas>
    <div id="cameraPanel" data-active="true" data-has-frame="false">
      <div id="cameraHeader">
        <span id="cameraLabel">camera</span>
      </div>
      <img id="cameraImg" alt="" />
      <div id="cameraEmpty">waiting for frames…</div>
    </div>
    <div id="armsPanel" data-active="false">
      <div id="armsHeader">Arm joints</div>
      <div id="armsColumns">
        <div class="arm-col">
          <div class="arm-col-title">Left</div>
          <div id="leftArmSliders" class="arm-sliders"></div>
        </div>
        <div class="arm-col">
          <div class="arm-col-title">Right</div>
          <div id="rightArmSliders" class="arm-sliders"></div>
        </div>
      </div>
    </div>
    <div id="hud">
      <div class="hud-group">
        <span class="hud-label">View</span>
        <button id="toggleScene" data-active="true">Scene</button>
        <button id="toggleRobot" data-active="true">Robot</button>
        <button id="toggleCamera" data-active="true">Camera</button>
        <button id="toggleLidar" data-active="true">Lidar</button>
        <button id="toggleDepth" data-active="true">Depth</button>
        <button id="toggleWire" data-active="false">Wire</button>
        <button id="forceVisible" data-active="false">Force</button>
      </div>
      <div class="hud-group">
        <span class="hud-label">Policy</span>
        <button id="policyArm" data-active="false" title="Arm/disarm the coordinator's control tasks">Arm</button>
        <button id="policyDryRun" data-active="true" title="Dry-run: task computes but coordinator does not write to hardware">Dry-run</button>
      </div>
      <div class="hud-group">
        <span class="hud-label">Arms</span>
        <button id="armsToggle" data-active="false">Sliders</button>
        <button id="armsRelease">Release</button>
      </div>
      <div class="hud-group">
        <span class="hud-label">Interact</span>
        <button id="toggleDrive" data-active="false">Drive</button>
        <button id="navClick" data-active="false">Nav</button>
        <button id="pointClick" data-active="false">Point</button>
        <button id="focusRobot">Focus</button>
        <button id="loadScene">Load Scene</button>
        <button id="respawnRobot" class="hud-hidden-by-default">Respawn</button>
      </div>
      <span id="status">starting</span>
    </div>
    <script>
      const canvas = document.getElementById("renderCanvas");
      const statusEl = document.getElementById("status");
      const engine = new BABYLON.Engine(canvas, true, {
        preserveDrawingBuffer: true,
        stencil: true,
        antialias: true,
      });
      const scene = new BABYLON.Scene(engine);
      scene.useRightHandedSystem = true;
      scene.clearColor = new BABYLON.Color4(0.055, 0.063, 0.078, 1);

      const camera = new BABYLON.ArcRotateCamera(
        "camera",
        -Math.PI / 2,
        Math.PI / 3,
        8,
        new BABYLON.Vector3(0, 0, 1),
        scene,
      );
      camera.upVector = new BABYLON.Vector3(0, 0, 1);
      camera.minZ = 0.01;
      camera.maxZ = 100000;
      camera.wheelPrecision = 40;
      camera.attachControl(canvas, true);

      new BABYLON.HemisphericLight("skyLight", new BABYLON.Vector3(0.2, 0.4, 1), scene);
      const sun = new BABYLON.DirectionalLight("sun", new BABYLON.Vector3(-0.4, -0.6, -1), scene);
      sun.position = new BABYLON.Vector3(20, 30, 40);
      scene.environmentIntensity = 0.85;
      try {
        scene.environmentTexture = BABYLON.CubeTexture.CreateFromPrefilteredData(
          "https://assets.babylonjs.com/environments/environmentSpecular.env",
          scene,
        );
      } catch (error) {
        console.warn("environment map unavailable", error);
      }

      const bodyNodes = new Map();
      const sceneMeshes = [];
      const robotMeshes = [];
      const maxAutoSceneBytes = 2 * 1024 * 1024 * 1024;
      const params = new URLSearchParams(window.location.search);
      const useRobotMesh = params.get("robot") !== "proxy";
      const sceneMode = params.get("scene") || "auto";
      let sceneConfig = null;
      let sceneLoadStarted = false;
      let pathMesh = null;
      let lidarMesh = null;
      let lidarMaterial = null;
      let lidarVisible = true;
      let clickMode = null;
      let navGoalMarker = null;
      let pointGoalMarker = null;
      let latestRootPosition = null;
      let sceneDepthEnabled = true;
      let sceneWireEnabled = false;
      let forceVisibleEnabled = false;
      let driveEnabled = false;
      let lastDriveSendTime = 0;
      let lastDriveSignature = "";
      let proxyMaterial = null;
      const pressedKeys = new Set();
      const driveSendPeriod = 0.08;
      const driveLinearSpeed = 0.35;
      const driveStrafeSpeed = 0.25;
      const driveAngularSpeed = 0.8;

      const vec3 = (values) => new BABYLON.Vector3(values[0], values[1], values[2]);
      const quatWxyz = (values) =>
        new BABYLON.Quaternion(values[1], values[2], values[3], values[0]);

      function setStatus(message) {
        statusEl.textContent = message;
      }

      function setButtonActive(id, active) {
        document.getElementById(id).dataset.active = String(active);
      }

      function setSceneVisibility(visible) {
        for (const mesh of sceneMeshes) mesh.setEnabled(visible);
        setButtonActive("toggleScene", visible);
      }

      function setRobotVisibility(visible) {
        for (const mesh of robotMeshes) mesh.setEnabled(visible);
        setButtonActive("toggleRobot", visible);
      }

      function setLidarVisibility(visible) {
        lidarVisible = visible;
        if (lidarMesh) lidarMesh.setEnabled(visible);
        setButtonActive("toggleLidar", visible);
      }

      function setClickMode(mode) {
        clickMode = clickMode === mode ? null : mode;
        setButtonActive("navClick", clickMode === "nav");
        setButtonActive("pointClick", clickMode === "point");
        if (clickMode === "nav") setStatus("click nav target");
        if (clickMode === "point") setStatus("click point target");
        if (clickMode === null) setStatus("live");
      }

      function markerMaterial(name, color) {
        const material = new BABYLON.StandardMaterial(name, scene);
        material.diffuseColor = color;
        material.emissiveColor = color.scale(0.75);
        material.specularColor = BABYLON.Color3.Black();
        material.disableLighting = true;
        return material;
      }

      const navMarkerMaterial = markerMaterial(
        "navGoalMaterial",
        new BABYLON.Color3(0.06, 0.82, 1.0),
      );
      const pointMarkerMaterial = markerMaterial(
        "pointGoalMaterial",
        new BABYLON.Color3(1.0, 0.18, 0.78),
      );

      function placeMarker(existingMarker, name, position, material, diameter) {
        if (existingMarker) existingMarker.dispose();
        const marker = BABYLON.MeshBuilder.CreateSphere(
          name,
          { diameter, segments: 16 },
          scene,
        );
        marker.position = position;
        marker.material = material;
        marker.isPickable = false;
        return marker;
      }

      function updateKeyboardCamera() {
        if (driveEnabled) return;
        const deltaSeconds = Math.min(engine.getDeltaTime() / 1000, 0.05);
        const speed = (pressedKeys.has("shift") ? 8.0 : 2.7) * deltaSeconds;
        const up = new BABYLON.Vector3(0, 0, 1);
        const forward = camera.getForwardRay().direction;
        forward.z = 0;
        if (forward.lengthSquared() < 1e-8) return;
        forward.normalize();
        const right = BABYLON.Vector3.Cross(forward, up).normalize();
        const move = BABYLON.Vector3.Zero();

        if (pressedKeys.has("w")) move.addInPlace(forward);
        if (pressedKeys.has("s")) move.subtractInPlace(forward);
        if (pressedKeys.has("d")) move.addInPlace(right);
        if (pressedKeys.has("a")) move.subtractInPlace(right);
        if (pressedKeys.has("e")) move.addInPlace(up);
        if (pressedKeys.has("q")) move.subtractInPlace(up);

        if (move.lengthSquared() === 0) return;
        move.normalize().scaleInPlace(speed);
        camera.target.addInPlace(move);
      }

      function sendSocketPayload(payload) {
        const socket = socketRef.current;
        if (!socket || socket.readyState !== WebSocket.OPEN) return false;
        socket.send(JSON.stringify(payload));
        return true;
      }

      function currentDriveTwist() {
        const speedScale = pressedKeys.has("shift") ? 1.8 : 1.0;
        let linearX = 0.0;
        let linearY = 0.0;
        let angularZ = 0.0;

        if (pressedKeys.has("w")) linearX += driveLinearSpeed * speedScale;
        if (pressedKeys.has("s")) linearX -= driveLinearSpeed * speedScale;
        if (pressedKeys.has("q")) linearY += driveStrafeSpeed * speedScale;
        if (pressedKeys.has("e")) linearY -= driveStrafeSpeed * speedScale;
        if (pressedKeys.has("a")) angularZ += driveAngularSpeed * speedScale;
        if (pressedKeys.has("d")) angularZ -= driveAngularSpeed * speedScale;

        return {
          linear: [linearX, linearY, 0.0],
          angular: [0.0, 0.0, angularZ],
        };
      }

      function sendDriveCommand(force = false) {
        if (!driveEnabled && !force) return;
        const now = performance.now() / 1000;
        if (!force && now - lastDriveSendTime < driveSendPeriod) return;

        const twist = force
          ? { linear: [0.0, 0.0, 0.0], angular: [0.0, 0.0, 0.0] }
          : currentDriveTwist();
        const signature = JSON.stringify(twist);
        const isZero =
          twist.linear.every((value) => Math.abs(value) < 1e-6) &&
          twist.angular.every((value) => Math.abs(value) < 1e-6);
        if (!force && isZero && signature === lastDriveSignature) return;

        if (sendSocketPayload({ type: "cmd_vel", ...twist })) {
          lastDriveSendTime = now;
          lastDriveSignature = signature;
        }
      }

      function setDriveEnabled(enabled) {
        driveEnabled = enabled;
        setButtonActive("toggleDrive", enabled);
        setStatus(enabled ? "drive: WASD turn/move, QE strafe" : "live");
        if (!enabled) sendDriveCommand(true);
      }

      function setSceneDepthWrite(enabled) {
        sceneDepthEnabled = enabled;
        const visited = new Set();
        for (const mesh of sceneMeshes) {
          const material = mesh.material;
          if (!material || visited.has(material.uniqueId)) continue;
          visited.add(material.uniqueId);
          material.disableDepthWrite = !enabled;
          material.needDepthPrePass = enabled;
        }
        setButtonActive("toggleDepth", enabled);
      }

      function setSceneWireframe(enabled) {
        sceneWireEnabled = enabled;
        const visited = new Set();
        for (const mesh of sceneMeshes) {
          const material = mesh.material;
          if (!material || visited.has(material.uniqueId)) continue;
          visited.add(material.uniqueId);
          material.wireframe = enabled;
        }
        setButtonActive("toggleWire", enabled);
      }

      function setForceVisible(enabled) {
        forceVisibleEnabled = enabled;
        const visited = new Set();
        for (const mesh of sceneMeshes) {
          const material = mesh.material;
          if (!material || visited.has(material.uniqueId)) continue;
          visited.add(material.uniqueId);
          material.backFaceCulling = false;
          material.alpha = 1;
          material.disableDepthWrite = false;
          material.forceDepthWrite = true;
          material.transparencyMode = enabled
            ? BABYLON.Material.MATERIAL_OPAQUE
            : material.transparencyMode;
          if (enabled && material.albedoColor) {
            material.metallic = 0;
            material.roughness = 0.9;
            material.environmentIntensity = 1;
          }
          if (enabled && material.diffuseColor) {
            material.diffuseColor = material.diffuseColor || new BABYLON.Color3(0.8, 0.8, 0.8);
          }
        }
        setButtonActive("forceVisible", enabled);
      }

      function fitCameraToMeshes(meshes) {
        const min = new BABYLON.Vector3(Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY);
        const max = new BABYLON.Vector3(Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY);
        let count = 0;
        for (const mesh of meshes) {
          if (!mesh.getTotalVertices || mesh.getTotalVertices() === 0) continue;
          mesh.computeWorldMatrix(true);
          mesh.refreshBoundingInfo(true);
          const box = mesh.getBoundingInfo().boundingBox;
          BABYLON.Vector3.MinimizeToRef(min, box.minimumWorld, min);
          BABYLON.Vector3.MaximizeToRef(max, box.maximumWorld, max);
          count += 1;
        }
        if (count === 0) return;
        const center = min.add(max).scale(0.5);
        const extent = max.subtract(min);
        camera.setTarget(center);
        camera.radius = Math.max(2, extent.length() * 0.55);
        setStatus(`scene ${count} meshes`);
      }

      function focusRobot() {
        if (!latestRootPosition) return;
        camera.setTarget(latestRootPosition);
        camera.radius = Math.max(4, camera.radius);
      }

      async function loadConfig() {
        const response = await fetch("/config.json", { cache: "no-store" });
        return await response.json();
      }

      async function loadSceneAsset(config) {
        if (sceneLoadStarted) return;
        sceneLoadStarted = true;
        if (!config.sceneFile) return;
        if (config.sceneBytes > maxAutoSceneBytes) {
          setStatus("scene exceeds browser load guard");
          return;
        }
        setStatus("loading scene");
        const root = new BABYLON.TransformNode("sceneRoot", scene);
        root.position = vec3(config.scenePosition);
        root.scaling = new BABYLON.Vector3(config.sceneScale, config.sceneScale, config.sceneScale);
        root.rotationQuaternion = quatWxyz(config.sceneWxyz);

        engine.stopRenderLoop(renderFrame);
        let result = null;
        try {
          result = await BABYLON.SceneLoader.ImportMeshAsync(null, "/assets/", config.sceneFile, scene);
        } finally {
          engine.runRenderLoop(renderFrame);
        }

        for (const light of result.lights || []) {
          light.dispose();
        }
        for (const camera of result.cameras || []) {
          camera.dispose();
        }
        for (const mesh of result.meshes) {
          if (mesh.parent === null) mesh.parent = root;
          mesh.isPickable = true;
          mesh.metadata = { dimosSceneMesh: true };
          if (mesh.getTotalVertices && mesh.getTotalVertices() > 0) sceneMeshes.push(mesh);
          if (mesh.material) {
            mesh.material.backFaceCulling = false;
            mesh.material.forceDepthWrite = true;
          }
        }
        setSceneDepthWrite(sceneDepthEnabled);
        setSceneWireframe(sceneWireEnabled);
        setForceVisible(forceVisibleEnabled);
        fitCameraToMeshes(sceneMeshes);
      }

      async function loadRobot() {
        setStatus("loading robot");
        const response = await fetch("/robot.json", { cache: "no-store" });
        const payload = await response.json();
        for (const bodyName of payload.bodyNames) {
          const node = new BABYLON.TransformNode(`body:${bodyName}`, scene);
          node.rotationQuaternion = BABYLON.Quaternion.Identity();
          bodyNodes.set(bodyName, node);
        }
        for (const geom of payload.geoms) {
          const mesh = new BABYLON.Mesh(`robot:${geom.id}`, scene);
          const normals = [];
          BABYLON.VertexData.ComputeNormals(geom.vertices, geom.indices, normals);
          const vertexData = new BABYLON.VertexData();
          vertexData.positions = geom.vertices;
          vertexData.indices = geom.indices;
          vertexData.normals = normals;
          vertexData.applyToMesh(mesh);

          const material = new BABYLON.StandardMaterial(`robotMat:${geom.id}`, scene);
          material.diffuseColor = new BABYLON.Color3(geom.rgba[0], geom.rgba[1], geom.rgba[2]);
          material.specularColor = new BABYLON.Color3(0.18, 0.18, 0.18);
          material.alpha = geom.rgba[3] > 0 ? geom.rgba[3] : 1;
          material.backFaceCulling = false;
          mesh.material = material;

          mesh.parent = bodyNodes.get(geom.body);
          mesh.position = vec3(geom.position);
          mesh.rotationQuaternion = quatWxyz(geom.wxyz);
          mesh.isPickable = false;
          robotMeshes.push(mesh);
        }
      }

      function proxyDiameter(bodyName) {
        if (bodyName === "world") return 0;
        if (bodyName.includes("pelvis") || bodyName.includes("torso")) return 0.22;
        if (bodyName.includes("hip") || bodyName.includes("shoulder")) return 0.14;
        if (bodyName.includes("knee") || bodyName.includes("elbow")) return 0.11;
        if (bodyName.includes("ankle") || bodyName.includes("wrist")) return 0.09;
        return 0.075;
      }

      function ensureBodyNode(bodyName) {
        let node = bodyNodes.get(bodyName);
        if (node) return node;
        node = new BABYLON.TransformNode(`body:${bodyName}`, scene);
        node.rotationQuaternion = BABYLON.Quaternion.Identity();
        bodyNodes.set(bodyName, node);
        if (!useRobotMesh) {
          const diameter = proxyDiameter(bodyName);
          if (diameter > 0) {
            if (!proxyMaterial) {
              proxyMaterial = new BABYLON.StandardMaterial("robotProxyMat", scene);
              proxyMaterial.diffuseColor = new BABYLON.Color3(0.95, 0.62, 0.24);
              proxyMaterial.specularColor = new BABYLON.Color3(0.22, 0.22, 0.22);
            }
            const marker = BABYLON.MeshBuilder.CreateSphere(
              `robotProxy:${bodyName}`,
              { diameter, segments: 8 },
              scene,
            );
            marker.material = proxyMaterial;
            marker.parent = node;
            marker.isPickable = false;
            robotMeshes.push(marker);
          }
        }
        return node;
      }

      function updateState(payload) {
        for (const body of payload.bodies) {
          const node = ensureBodyNode(body.name);
          node.position = vec3(body.position);
          node.rotationQuaternion = quatWxyz(body.wxyz);
          if (body.name === "pelvis" || body.name === "torso_link" || body.name === "body_1") {
            latestRootPosition = node.position.clone();
          }
        }

        if (!latestRootPosition && payload.bodies.length > 1) {
          latestRootPosition = vec3(payload.bodies[1].position);
        }

        if (pathMesh) {
          pathMesh.dispose();
          pathMesh = null;
        }
        if (payload.path && payload.path.length > 1) {
          pathMesh = BABYLON.MeshBuilder.CreateLines(
            "navPath",
            { points: payload.path.map((point) => vec3([point[0], point[1], point[2] + 0.08])) },
            scene,
          );
          pathMesh.color = new BABYLON.Color3(0.15, 0.95, 0.68);
          pathMesh.isPickable = false;
        }
      }

      function updatePointCloud(payload) {
        const count = payload.count || 0;
        if (count === 0 || !payload.positions || !payload.colors) return;

        const positions = Float32Array.from(payload.positions);
        const packedColors = payload.colors;
        const colors = new Float32Array(count * 4);
        for (let i = 0; i < count; i += 1) {
          colors[i * 4 + 0] = packedColors[i * 3 + 0] / 255;
          colors[i * 4 + 1] = packedColors[i * 3 + 1] / 255;
          colors[i * 4 + 2] = packedColors[i * 3 + 2] / 255;
          colors[i * 4 + 3] = 0.86;
        }

        const nextMesh = new BABYLON.Mesh("lidarCloud", scene);
        const vertexData = new BABYLON.VertexData();
        vertexData.positions = positions;
        vertexData.colors = colors;
        vertexData.applyToMesh(nextMesh, true);
        nextMesh.hasVertexAlpha = true;
        nextMesh.alwaysSelectAsActiveMesh = true;
        nextMesh.isPickable = false;

        if (!lidarMaterial) {
          lidarMaterial = new BABYLON.StandardMaterial("lidarMaterial", scene);
          lidarMaterial.pointsCloud = true;
          lidarMaterial.pointSize = 2.4;
          lidarMaterial.disableLighting = true;
          lidarMaterial.emissiveColor = BABYLON.Color3.White();
          lidarMaterial.alpha = 0.9;
        }
        nextMesh.material = lidarMaterial;
        nextMesh.setEnabled(lidarVisible);

        if (lidarMesh) lidarMesh.dispose();
        lidarMesh = nextMesh;
      }

      function connectWebSocket(socketRef) {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const socket = new WebSocket(`${protocol}//${window.location.host}/ws`);
        socket.binaryType = "arraybuffer";
        socketRef.current = socket;
        socket.onopen = () => setStatus("live");
        socket.onclose = () => {
          setStatus("reconnecting");
          setTimeout(() => connectWebSocket(socketRef), 1000);
        };
        socket.onerror = () => setStatus("socket error");
        socket.onmessage = (event) => {
          if (typeof event.data === "string") {
            const payload = JSON.parse(event.data);
            if (payload.type === "state") {
              updateState(payload);
              _updateSlidersFromState(payload.joints);
            }
            if (payload.type === "pointcloud") updatePointCloud(payload);
          } else {
            handleBinaryMessage(event.data);
          }
        };
        return socket;
      }

      // Binary websocket frame layout:
      //   byte 0:      message type (0x01 = camera)
      //   bytes 1-2:   name length (big-endian uint16)
      //   bytes 3..n:  utf-8 camera name
      //   bytes n..:   payload (JPEG bytes for camera)
      let _lastCameraURL = null;
      function handleBinaryMessage(buffer) {
        const view = new DataView(buffer);
        const msgType = view.getUint8(0);
        if (msgType !== 0x01) return; // unknown — ignore
        const nameLen = view.getUint16(1, false);
        const nameBytes = new Uint8Array(buffer, 3, nameLen);
        const cameraName = new TextDecoder().decode(nameBytes);
        const jpegBytes = new Uint8Array(buffer, 3 + nameLen);
        const blob = new Blob([jpegBytes], { type: "image/jpeg" });
        const url = URL.createObjectURL(blob);
        const img = document.getElementById("cameraImg");
        const label = document.getElementById("cameraLabel");
        if (img) {
          img.src = url;
          if (_lastCameraURL) URL.revokeObjectURL(_lastCameraURL);
          _lastCameraURL = url;
        }
        if (label) label.textContent = cameraName;
        const panel = document.getElementById("cameraPanel");
        if (panel) panel.dataset.hasFrame = "true";
      }

      function installClickPublisher(socketRef) {
        scene.onPointerObservable.add((pointerInfo) => {
          if (pointerInfo.type !== BABYLON.PointerEventTypes.POINTERPICK) return;
          const event = pointerInfo.event;
          if (event.target !== canvas) return;

          const publishNav = clickMode === "nav" || event.shiftKey;
          const publishPoint = clickMode === "point" || event.altKey;
          if (!publishNav && !publishPoint) return;

          const socket = socketRef.current;
          if (!socket || socket.readyState !== WebSocket.OPEN) return;

          if (publishNav) {
            const ray = scene.createPickingRay(
              scene.pointerX,
              scene.pointerY,
              BABYLON.Matrix.Identity(),
              camera,
            );
            if (Math.abs(ray.direction.z) < 1e-6) return;
            const distance = -ray.origin.z / ray.direction.z;
            if (distance <= 0) return;
            const point = ray.origin.add(ray.direction.scale(distance));
            navGoalMarker = placeMarker(
              navGoalMarker,
              "navGoalMarker",
              new BABYLON.Vector3(point.x, point.y, 0.08),
              navMarkerMaterial,
              0.22,
            );
            socket.send(
              JSON.stringify({
                type: "clicked_point",
                point: [point.x, point.y, 0.0],
              }),
            );
            setClickMode(null);
            setStatus("nav target sent");
            return;
          }

          const pick = pointerInfo.pickInfo;
          let point = null;
          if (pick && pick.hit && pick.pickedPoint) {
            point = pick.pickedPoint;
          } else {
            const ray = scene.createPickingRay(
              scene.pointerX,
              scene.pointerY,
              BABYLON.Matrix.Identity(),
              camera,
            );
            if (Math.abs(ray.direction.z) < 1e-6) return;
            const distance = (1.0 - ray.origin.z) / ray.direction.z;
            if (distance <= 0) return;
            point = ray.origin.add(ray.direction.scale(distance));
          }
          pointGoalMarker = placeMarker(
            pointGoalMarker,
            "pointGoalMarker",
            point,
            pointMarkerMaterial,
            0.16,
          );
          socket.send(
            JSON.stringify({
              type: "point_goal",
              point: [point.x, point.y, point.z],
            }),
          );
          setClickMode(null);
          setStatus("point target sent");
        });
      }

      document.getElementById("toggleScene").onclick = () => {
        const visible = document.getElementById("toggleScene").dataset.active !== "true";
        setSceneVisibility(visible);
      };
      document.getElementById("toggleRobot").onclick = () => {
        const visible = document.getElementById("toggleRobot").dataset.active !== "true";
        setRobotVisibility(visible);
      };
      document.getElementById("toggleDrive").onclick = () => setDriveEnabled(!driveEnabled);
      document.getElementById("respawnRobot").onclick = () => {
        sendDriveCommand(true);
        sendSocketPayload({ type: "respawn" });
        setStatus("respawn requested");
      };
      document.getElementById("toggleLidar").onclick = () => setLidarVisibility(!lidarVisible);
      document.getElementById("toggleCamera").onclick = () => {
        const btn = document.getElementById("toggleCamera");
        const panel = document.getElementById("cameraPanel");
        const active = btn.dataset.active !== "true";
        btn.dataset.active = active ? "true" : "false";
        if (panel) panel.dataset.active = active ? "true" : "false";
      };
      document.getElementById("navClick").onclick = () => setClickMode("nav");
      document.getElementById("pointClick").onclick = () => setClickMode("point");
      document.getElementById("toggleDepth").onclick = () => setSceneDepthWrite(!sceneDepthEnabled);
      document.getElementById("toggleWire").onclick = () => setSceneWireframe(!sceneWireEnabled);
      document.getElementById("forceVisible").onclick = () => setForceVisible(!forceVisibleEnabled);
      document.getElementById("focusRobot").onclick = focusRobot;
      document.getElementById("loadScene").onclick = () => {
        if (!sceneConfig) return;
        loadSceneAsset(sceneConfig).catch((error) => {
          console.error(error);
          setStatus("scene load failed");
        });
      };

      // --- Policy arm / dry-run toggles ---
      // Initial dataset.active reflects the coordinator's defaults for the
      // typical real-hardware blueprint (unarmed, dry-run on). If the
      // blueprint configured different defaults the button is still a plain
      // toggle — click it once to sync.
      document.getElementById("policyArm").onclick = () => {
        const btn = document.getElementById("policyArm");
        const engaged = btn.dataset.active !== "true";
        if (!sendSocketPayload({ type: "set_activated", engaged })) return;
        btn.dataset.active = engaged ? "true" : "false";
        setStatus(engaged ? "policy armed" : "policy disarmed");
      };
      document.getElementById("policyDryRun").onclick = () => {
        const btn = document.getElementById("policyDryRun");
        const enabled = btn.dataset.active !== "true";
        if (!sendSocketPayload({ type: "set_dry_run", enabled })) return;
        btn.dataset.active = enabled ? "true" : "false";
        setStatus(enabled ? "dry-run on" : "dry-run off (live)");
      };

      // --- Arm slider panel ---
      // Toggle visibility
      document.getElementById("armsToggle").onclick = () => {
        const btn = document.getElementById("armsToggle");
        const panel = document.getElementById("armsPanel");
        const active = btn.dataset.active !== "true";
        btn.dataset.active = active ? "true" : "false";
        panel.dataset.active = active ? "true" : "false";
      };

      // Release: stop publishing arm commands (hand control back to MC)
      document.getElementById("armsRelease").onclick = () => {
        if (sendSocketPayload({ type: "release_arms" })) {
          setStatus("arms released");
        }
      };

      // Build the slider list from /arms.json. Each slider sends an
      // {type: arm_joint, name, position} message on input (throttled).
      function _humanLabel(name) {
        // strip "left_"/"right_" prefix for column-internal display
        return name
          .replace(/^left_/, "")
          .replace(/^right_/, "")
          .replace(/_/g, " ");
      }
      // Track which slider the user is currently dragging so we don't
      // overwrite its value from incoming joint-state updates.
      const _armSliders = {};       // joint_name -> {slider, val, dragging}
      let _armSendThrottle = {};
      function _throttledSendJoint(name, position) {
        const now = performance.now();
        const last = _armSendThrottle[name] || 0;
        if (now - last < 30) return; // ~33 Hz max per slider
        _armSendThrottle[name] = now;
        sendSocketPayload({ type: "arm_joint", name, position });
      }
      function _buildSlider(joint) {
        const row = document.createElement("div");
        row.className = "arm-slider-row";

        const labelTop = document.createElement("div");
        labelTop.className = "joint-name";
        labelTop.textContent = _humanLabel(joint.name);
        row.appendChild(labelTop);

        const val = document.createElement("div");
        val.className = "joint-val";
        val.textContent = "  …  ";
        row.appendChild(val);

        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = String(joint.min);
        slider.max = String(joint.max);
        slider.step = "0.005";
        // Default to midpoint until we get a real reading — visually obvious
        // it's not real yet (the value cell shows "..." until first update).
        slider.value = String((joint.min + joint.max) / 2);
        slider.disabled = true;  // enabled once a joint_state arrives

        const entry = { slider, val, dragging: false, ready: false };
        _armSliders[joint.name] = entry;

        slider.addEventListener("pointerdown", () => { entry.dragging = true; });
        slider.addEventListener("pointerup",   () => { entry.dragging = false; });
        slider.addEventListener("pointercancel", () => { entry.dragging = false; });

        slider.oninput = () => {
          const pos = parseFloat(slider.value);
          val.textContent = pos.toFixed(3);
          _throttledSendJoint(joint.name, pos);
        };
        slider.onchange = () => {
          // Final exact value on release (in case the throttle dropped it)
          const pos = parseFloat(slider.value);
          sendSocketPayload({ type: "arm_joint", name: joint.name, position: pos });
        };
        row.appendChild(slider);

        const range = document.createElement("div");
        range.className = "joint-range";
        const lo = document.createElement("span");
        lo.textContent = joint.min.toFixed(2);
        const hi = document.createElement("span");
        hi.textContent = joint.max.toFixed(2);
        range.appendChild(lo);
        range.appendChild(hi);
        row.appendChild(range);

        return row;
      }

      function _updateSlidersFromState(joints) {
        if (!joints) return;
        for (const [name, value] of Object.entries(joints)) {
          const entry = _armSliders[name];
          if (!entry) continue;
          if (!entry.ready) {
            // First sample → enable the slider, set it to actual position.
            entry.ready = true;
            entry.slider.disabled = false;
          }
          if (entry.dragging) continue;  // don't fight the user
          // Only set value when the slider is idle, so the user sees ground truth.
          entry.slider.value = String(value);
          entry.val.textContent = Number(value).toFixed(3);
        }
      }

      (async () => {
        try {
          const resp = await fetch("/arms.json");
          const data = await resp.json();
          const leftCol = document.getElementById("leftArmSliders");
          const rightCol = document.getElementById("rightArmSliders");
          for (const j of data.joints || []) {
            const row = _buildSlider(j);
            if (j.name.startsWith("left_")) leftCol.appendChild(row);
            else if (j.name.startsWith("right_")) rightCol.appendChild(row);
          }
        } catch (e) {
          console.error("Failed to load /arms.json:", e);
        }
      })();

      const socketRef = { current: null };
      (async () => {
        try {
          const config = await loadConfig();
          sceneConfig = config;
          if (useRobotMesh) await loadRobot();
          connectWebSocket(socketRef);
          installClickPublisher(socketRef);
          setStatus("live");
          if (sceneMode !== "0" && sceneMode !== "manual") {
            window.setTimeout(() => loadSceneAsset(config).catch((error) => {
              console.error(error);
              setStatus("scene load failed");
            }), 0);
          }
        } catch (error) {
          console.error(error);
          setStatus("load failed");
        }
      })();

      window.addEventListener("keydown", (event) => {
        const key = event.key.toLowerCase();
        if (key === "shift") pressedKeys.add("shift");
        if (key === " ") {
          if (driveEnabled) sendDriveCommand(true);
          event.preventDefault();
          return;
        }
        if (!["w", "a", "s", "d", "q", "e"].includes(key)) return;
        pressedKeys.add(key);
        event.preventDefault();
      });

      window.addEventListener("keyup", (event) => {
        const key = event.key.toLowerCase();
        if (key === "shift") pressedKeys.delete("shift");
        pressedKeys.delete(key);
      });

      window.addEventListener("blur", () => {
        pressedKeys.clear();
        sendDriveCommand(true);
      });

      function renderFrame() {
        updateKeyboardCamera();
        sendDriveCommand(false);
        scene.render();
      }

      engine.runRenderLoop(renderFrame);
      window.addEventListener("resize", () => engine.resize());
    </script>
  </body>
</html>
"""
