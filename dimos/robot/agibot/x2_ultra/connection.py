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

from abc import ABC, abstractmethod
from threading import Lock, Thread
import time
from typing import Any

import numpy as np
import open3d as o3d
from pydantic import Field

from dimos.agents.annotation import skill
from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.spec.perception import IMU, Camera, Lidar, Pointcloud
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# ROS2 topic constants
# Using the *compressed* (JPEG, ~150 KB/frame, RELIABLE QoS) RGB topic instead
# of the raw 1280x720 RGB topic (~2.7 MB/frame, BEST_EFFORT): the raw stream's
# UDP fragments were being dropped by DDS on the laptop, so callbacks never
# fired. The robot publishes both; the compressed one delivers reliably.
_TOPIC_RGB_IMAGE = "/aima/hal/sensor/rgbd_head_front/rgb_image/compressed"
_TOPIC_DEPTH_IMAGE = "/aima/hal/sensor/rgbd_head_front/depth_image"
_TOPIC_DEPTH_CLOUD = "/aima/hal/sensor/rgbd_head_front/depth_pointcloud"
_TOPIC_RGB_CAM_INFO = "/aima/hal/sensor/rgbd_head_front/rgb_camera_info"
_TOPIC_LIDAR = "/aima/hal/sensor/lidar_chest_front/lidar_pointcloud"
_TOPIC_IMU = "/aima/hal/imu/chest/state"
_TOPIC_VELOCITY = "/aima/mc/locomotion/velocity"
_TOPIC_JOINT_ARM = "/aima/hal/joint/arm/state"
_TOPIC_JOINT_LEG = "/aima/hal/joint/leg/state"
_TOPIC_JOINT_WAIST = "/aima/hal/joint/waist/state"
_TOPIC_JOINT_HEAD = "/aima/hal/joint/head/state"
_SVC_INPUT_SOURCE = "/aimdk_5Fmsgs/srv/SetMcInputSource"

# Joint name ordering per the AgiBot X2 SDK docs (Interface > Control > Joint Control).
# Names match the X2 URDF/MJCF (sans the "_joint" suffix, which the viewer adds).
# JointStateArray.joints[] is positionally indexed in this exact order.
_ARM_JOINT_NAMES: tuple[str, ...] = (
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_yaw", "left_wrist_pitch", "left_wrist_roll",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_yaw", "right_wrist_pitch", "right_wrist_roll",
)
_LEG_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
)
_WAIST_JOINT_NAMES: tuple[str, ...] = ("waist_yaw", "waist_pitch", "waist_roll")
_HEAD_JOINT_NAMES: tuple[str, ...] = ("head_yaw", "head_pitch")

# Velocity limits from the SDK
_MAX_FORWARD = 1.0
_MIN_FORWARD = 0.2
_MAX_LATERAL = 1.0
_MIN_LATERAL = 0.2
_MAX_ANGULAR = 1.0
_MIN_ANGULAR = 0.1

# Input source config for velocity control
_INPUT_SOURCE_NAME = "dimos"
_INPUT_SOURCE_PRIORITY = 40
_INPUT_SOURCE_TIMEOUT = 1000  # ms


class X2ConnectionBase(Module, ABC):
    """Abstract base for X2 Ultra connections (real hardware and future simulation).

    Other modules that depend on X2 RPC methods should reference this base class
    so blueprint wiring works regardless of which concrete connection is deployed.
    """

    config: ModuleConfig

    @rpc
    @abstractmethod
    def start(self) -> None:
        super().start()

    @rpc
    @abstractmethod
    def stop(self) -> None:
        super().stop()

    @rpc
    @abstractmethod
    def move(self, twist: Twist, duration: float = 0.0) -> bool: ...

    @rpc
    @abstractmethod
    def stop_motion(self) -> bool: ...

    @rpc
    @abstractmethod
    def observe(self) -> Image | None: ...


class ConnectionConfig(ModuleConfig):
    ros_domain_id: int = Field(default=0, description="ROS_DOMAIN_ID matching the robot")


def _clamp_velocity(value: float, min_mag: float, max_mag: float) -> float:
    """Apply SDK dead zone: zero if near zero, clamp to [min, max] magnitude otherwise."""
    if abs(value) < 0.005:
        return 0.0
    return (
        float(np.clip(value, -max_mag, max_mag))
        if abs(value) >= min_mag
        else (min_mag * np.sign(value))
    )


def _ros_image_to_dimos(msg: Any) -> Image:
    """Convert a ROS2 sensor_msgs/Image to a dimos Image."""
    ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    enc = msg.encoding.lower()

    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)

    if enc in ("rgb8",):
        arr = data.reshape((msg.height, msg.width, 3))
        fmt = ImageFormat.RGB
    elif enc in ("bgr8",):
        arr = data.reshape((msg.height, msg.width, 3))
        fmt = ImageFormat.BGR
    elif enc in ("rgba8",):
        arr = data.reshape((msg.height, msg.width, 4))
        fmt = ImageFormat.RGBA
    elif enc in ("bgra8",):
        arr = data.reshape((msg.height, msg.width, 4))
        fmt = ImageFormat.BGRA
    elif enc in ("mono8",):
        arr = data.reshape((msg.height, msg.width))
        fmt = ImageFormat.GRAY
    elif enc in ("16uc1", "mono16"):
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape((msg.height, msg.width))
        fmt = ImageFormat.DEPTH16
    elif enc in ("32fc1",):
        arr = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape((msg.height, msg.width))
        fmt = ImageFormat.DEPTH
    else:
        # Fall back to raw reshape; assume 3-channel
        logger.warning("X2Connection: unknown image encoding %s, treating as BGR", enc)
        arr = data.reshape((msg.height, msg.width, -1))
        fmt = ImageFormat.BGR

    return Image(data=arr, format=fmt, frame_id=msg.header.frame_id, ts=ts)


def _ros_pointcloud2_to_dimos(msg: Any) -> PointCloud2:
    """Convert a ROS2 sensor_msgs/PointCloud2 to a dimos PointCloud2.

    Handles any point_step by stride-indexing float32 values at the x/y/z
    field offsets (offsets must be 4-byte aligned, which is standard).
    """
    ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    n_points = msg.width * msg.height
    if n_points == 0:
        return PointCloud2(frame_id=msg.header.frame_id, ts=ts)

    offsets = {f.name: f.offset for f in msg.fields}
    x_off = offsets.get("x", 0)
    y_off = offsets.get("y", 4)
    z_off = offsets.get("z", 8)
    step = msg.point_step
    step_f32 = step // 4

    # Interpret entire buffer as float32 then stride-index each field
    raw_f32 = np.frombuffer(bytes(msg.data), dtype=np.float32)
    xs = raw_f32[x_off // 4 :: step_f32][:n_points]
    ys = raw_f32[y_off // 4 :: step_f32][:n_points]
    zs = raw_f32[z_off // 4 :: step_f32][:n_points]

    pts = np.column_stack([xs, ys, zs]).astype(np.float64)
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    return PointCloud2(pointcloud=pcd, frame_id=msg.header.frame_id, ts=ts)


def _ros_camera_info_to_dimos(msg: Any) -> CameraInfo:
    """Convert a ROS2 sensor_msgs/CameraInfo to a dimos CameraInfo."""
    ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    return CameraInfo(
        height=msg.height,
        width=msg.width,
        distortion_model=msg.distortion_model,
        D=list(msg.d),
        K=list(msg.k),
        R=list(msg.r),
        P=list(msg.p),
        binning_x=msg.binning_x,
        binning_y=msg.binning_y,
        frame_id=msg.header.frame_id,
        ts=ts,
    )


def _ros_imu_to_dimos(msg: Any) -> Imu:
    """Convert a ROS2 sensor_msgs/Imu to a dimos Imu."""
    ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    return Imu(
        angular_velocity=Vector3(
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ),
        linear_acceleration=Vector3(
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ),
        orientation=Quaternion(
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        ),
        orientation_covariance=list(msg.orientation_covariance),
        angular_velocity_covariance=list(msg.angular_velocity_covariance),
        linear_acceleration_covariance=list(msg.linear_acceleration_covariance),
        frame_id=msg.header.frame_id,
        ts=ts,
    )


class X2Connection(X2ConnectionBase, Camera, Pointcloud, IMU, Lidar):
    """DIMOS module for the AgiBot X2 Ultra humanoid robot.

    Connects via ROS2 (rclpy) — set ROS_DOMAIN_ID to match the robot before launching.

    Streams:
      color_image  — RGB image from head RGBD camera
      camera_info  — camera intrinsics (auto-read from robot)
      pointcloud   — depth point cloud from head RGBD camera
      lidar        — chest LiDAR point cloud
      imu          — chest IMU
      joint_state  — combined arm/leg/waist/head joint positions (names match URDF/MJCF
                     without the "_joint" suffix)

    Inputs:
      cmd_vel — geometry_msgs/Twist: linear.x=forward, linear.y=lateral, angular.z=yaw
    """

    config: ConnectionConfig
    cmd_vel: In[Twist]
    color_image: Out[Image]
    camera_info: Out[CameraInfo]
    depth_image: Out[Image]
    depth_camera_info: Out[CameraInfo]
    pointcloud: Out[PointCloud2]
    lidar: Out[PointCloud2]
    imu: Out[Imu]
    joint_state: Out[JointState]

    _ros_node: Any = None
    _ros_thread: Thread | None = None
    _vel_publisher: Any = None
    _latest_video_frame: Image | None = None
    _input_source_registered: bool = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_video_frame = None
        # Joint positions aggregated across the 4 per-bodypart topics.
        # Locked because each subtopic callback writes its slice independently.
        self._joint_lock = Lock()
        self._joint_positions: dict[str, float] = {}
        self._input_source_registered = False

    @rpc
    def start(self) -> None:
        super().start()
        self._start_ros()

    @rpc
    def stop(self) -> None:
        self._stop_ros()
        super().stop()

    def _start_ros(self) -> None:
        try:
            import os

            import rclpy

            os.environ.setdefault("ROS_DOMAIN_ID", str(self.config.ros_domain_id))
            # dimos workers are forked from a forkserver, which can carry a
            # half-initialized DDS context into the child. Reusing it works
            # for the first message or two on high-bandwidth topics, then
            # silently drops. Tear it down and re-init in this worker so
            # FastDDS owns a clean state.
            try:
                rclpy.init()
            except RuntimeError as init_exc:
                if "must only be called once" not in str(init_exc):
                    raise
                logger.info("X2Connection: shutting down inherited rclpy context and re-initing")
                try:
                    rclpy.shutdown()
                except Exception:
                    pass
                rclpy.init()
        except Exception as e:
            logger.error("X2Connection: failed to init rclpy: %s", e)
            raise

        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        node = rclpy.create_node("dimos_x2_connection")
        self._ros_node = node
        # ReentrantCallbackGroup pairs with MultiThreadedExecutor (see _ros_spin):
        # without it, callbacks on this node still serialise even with a thread
        # pool. With it, the slow camera-decode callback runs in parallel with
        # high-rate joint-state callbacks.
        sensor_group = ReentrantCallbackGroup()

        # Compressed publisher uses RELIABLE/VOLATILE/KEEP_LAST. depth=1 was
        # the only setting under which standalone rclpy probes received frames
        # reliably (depth=5 + the dimos worker's other subscriptions caused
        # messages to silently drop after the first frame).
        from rclpy.qos import QoSDurabilityPolicy as _QoSDur
        compressed_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=_QoSDur.VOLATILE,
        )
        node.create_subscription(
            self._import_msg("sensor_msgs.msg", "CompressedImage"),
            _TOPIC_RGB_IMAGE,
            self._on_rgb_image,
            compressed_qos,
            callback_group=sensor_group,
        )
        node.create_subscription(
            self._import_msg("sensor_msgs.msg", "Image"),
            _TOPIC_DEPTH_IMAGE,
            self._on_depth_image,
            sensor_qos,
            callback_group=sensor_group,
        )
        node.create_subscription(
            self._import_msg("sensor_msgs.msg", "PointCloud2"),
            _TOPIC_DEPTH_CLOUD,
            self._on_depth_cloud,
            sensor_qos,
        )
        node.create_subscription(
            self._import_msg("sensor_msgs.msg", "PointCloud2"),
            _TOPIC_LIDAR,
            self._on_lidar,
            sensor_qos,
        )
        node.create_subscription(
            self._import_msg("sensor_msgs.msg", "Imu"),
            _TOPIC_IMU,
            self._on_imu,
            sensor_qos,
            callback_group=sensor_group,
        )

        # Joint state subscriptions (one per body part, all aimdk_msgs/JointStateArray).
        joint_state_msg = self._import_msg("aimdk_msgs.msg", "JointStateArray")
        node.create_subscription(
            joint_state_msg, _TOPIC_JOINT_ARM,
            lambda m: self._on_joint_state_array(m, _ARM_JOINT_NAMES),
            sensor_qos,
            callback_group=sensor_group,
        )
        node.create_subscription(
            joint_state_msg, _TOPIC_JOINT_LEG,
            lambda m: self._on_joint_state_array(m, _LEG_JOINT_NAMES),
            sensor_qos,
            callback_group=sensor_group,
        )
        node.create_subscription(
            joint_state_msg, _TOPIC_JOINT_WAIST,
            lambda m: self._on_joint_state_array(m, _WAIST_JOINT_NAMES),
            sensor_qos,
            callback_group=sensor_group,
        )
        node.create_subscription(
            joint_state_msg, _TOPIC_JOINT_HEAD,
            lambda m: self._on_joint_state_array(m, _HEAD_JOINT_NAMES),
            sensor_qos,
            callback_group=sensor_group,
        )

        from rclpy.qos import QoSDurabilityPolicy

        cam_info_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        node.create_subscription(
            self._import_msg("sensor_msgs.msg", "CameraInfo"),
            _TOPIC_RGB_CAM_INFO,
            self._on_camera_info,
            cam_info_qos,
            callback_group=sensor_group,
        )

        VelMsg = self._import_msg("aimdk_msgs.msg", "McLocomotionVelocity")
        self._vel_publisher = node.create_publisher(VelMsg, _TOPIC_VELOCITY, 10)

        self.register_disposable(self.cmd_vel.subscribe(self.move))

        self._ros_thread = Thread(target=self._ros_spin, daemon=True)
        self._ros_thread.start()

        # Register input source in a background thread so start() returns quickly
        Thread(target=self._register_input_source, daemon=True).start()

    def _ros_spin(self) -> None:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor

        # MultiThreadedExecutor: joint-state topics on this node tick at
        # 50-100 Hz × 4 body parts. A SingleThreadedExecutor starves the
        # camera callback (verified empirically: ~4 RGB frames then silence).
        try:
            executor = MultiThreadedExecutor(num_threads=4)
            executor.add_node(self._ros_node)
            executor.spin()
        except Exception as e:
            logger.warning("X2Connection: ROS spin exited: %s", e)

    def _stop_ros(self) -> None:
        if self._ros_node is not None:
            try:
                import rclpy

                self._ros_node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception as e:
                logger.warning("X2Connection: error during ROS shutdown: %s", e)
            self._ros_node = None

        if self._ros_thread and self._ros_thread.is_alive():
            self._ros_thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        self._ros_thread = None

    def _register_input_source(self) -> None:
        """Register the DIMOS input source with the motion controller.

        Uses polling instead of spin_until_future_complete because _ros_spin
        is already driving the node's executor in its own thread; calling
        spin_until_future_complete concurrently causes both to race and the
        response never gets delivered.
        """
        from aimdk_msgs.srv import SetMcInputSource

        client = self._ros_node.create_client(SetMcInputSource, _SVC_INPUT_SOURCE)

        svc_timeout = 15.0
        start = time.time()
        while not client.service_is_ready():
            if time.time() - start > svc_timeout:
                logger.error(
                    "X2Connection: input source service not available after %.0fs", svc_timeout
                )
                return
            logger.info("X2Connection: waiting for input source service...")
            time.sleep(1.0)

        req = SetMcInputSource.Request()
        req.action.value = 1001
        req.input_source.name = _INPUT_SOURCE_NAME
        req.input_source.priority = _INPUT_SOURCE_PRIORITY
        req.input_source.timeout = _INPUT_SOURCE_TIMEOUT

        future = None
        for i in range(8):
            req.request.header.stamp = self._ros_node.get_clock().now().to_msg()
            future = client.call_async(req)
            # Let the _ros_spin thread process the response — just poll here.
            deadline = time.time() + 3.0
            while not future.done() and time.time() < deadline:
                time.sleep(0.05)
            if future.done():
                break
            logger.info("X2Connection: retrying input source registration [%d]", i)

        if future is not None and future.done():
            try:
                resp = future.result()
                logger.info(
                    "X2Connection: input source registered, state=%s task_id=%s",
                    resp.response.state.value,
                    resp.response.task_id,
                )
                self._input_source_registered = True
            except Exception as e:
                logger.error("X2Connection: input source registration failed: %s", e)
        else:
            logger.error("X2Connection: input source registration timed out")

    # --- ROS2 sensor callbacks ---

    def _on_rgb_image(self, msg: Any) -> None:
        # sensor_msgs/CompressedImage; msg.data is the JPEG byte string.
        # The robot's raw RGB topic (~2.7 MB BEST_EFFORT) is fragmented by DDS
        # and silently dropped on our laptop side; the /compressed topic
        # (RELIABLE, ~150-400 KB) delivers, so we decode it here.
        import cv2

        jpeg = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        arr = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)  # returns BGR
        if arr is None:
            return

        # The X2's head RGBD sensor is mounted with its optical axis rotated
        # 180° (verified by visual inspection of a raw frame). Rotate here so
        # every downstream consumer gets a human-readable view.
        # NOTE: camera_info principal point (cx, cy) is left as-published; if
        # you need projection accuracy, derive (W-1-cx, H-1-cy).
        arr = np.ascontiguousarray(arr[::-1, ::-1])

        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        image = Image(data=arr, format=ImageFormat.BGR, frame_id=msg.header.frame_id, ts=ts)
        self.color_image.publish(image)
        self._latest_video_frame = image

    def _on_depth_image(self, msg: Any) -> None:
        image = _ros_image_to_dimos(msg)
        image.data = np.ascontiguousarray(image.data[::-1, ::-1])
        self.depth_image.publish(image)

    def _on_depth_cloud(self, msg: Any) -> None:
        self.pointcloud.publish(_ros_pointcloud2_to_dimos(msg))

    def _on_lidar(self, msg: Any) -> None:
        self.lidar.publish(_ros_pointcloud2_to_dimos(msg))

    def _on_imu(self, msg: Any) -> None:
        self.imu.publish(_ros_imu_to_dimos(msg))

    def _on_camera_info(self, msg: Any) -> None:
        self.camera_info.publish(_ros_camera_info_to_dimos(msg))

    # Joint state visualization throttle: the robot publishes at ~750 Hz per
    # body part (motor-control telemetry rate). Anything above ~50 Hz is wasted
    # work for visualization and starves slower callbacks (camera) of executor
    # time. Update the internal map every message (cheap dict write); only
    # publish to dimos at JOINT_VIZ_HZ.
    _JOINT_VIZ_HZ = 50.0

    def _on_joint_state_array(self, msg: Any, joint_names: tuple[str, ...]) -> None:
        """Merge one body-part slice into the unified joint_positions and publish.

        AgiBot publishes per-bodypart JointStateArray messages with positionally
        indexed joints; we map them to names via the SDK's documented ordering
        and emit the merged set on every update.
        """
        joints = list(msg.joints)
        if len(joints) != len(joint_names):
            logger.warning(
                "X2Connection: joint topic %s len=%d != expected %d (skipping)",
                joint_names[0] if joint_names else "?",
                len(joints),
                len(joint_names),
            )
            return

        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self._joint_lock:
            for name, joint in zip(joint_names, joints, strict=True):
                self._joint_positions[name] = float(joint.position)
            # Rate-limit dimos publishes: cheap dict updates always happen,
            # only emit a merged snapshot when the throttle interval is up.
            now = time.monotonic()
            last = getattr(self, "_last_joint_pub", 0.0)
            if now - last < 1.0 / self._JOINT_VIZ_HZ:
                return
            self._last_joint_pub = now
            names = list(self._joint_positions.keys())
            positions = list(self._joint_positions.values())

        self.joint_state.publish(
            JointState(ts=ts, frame_id="pelvis", name=names, position=positions)
        )

    # --- Motion control ---

    @rpc
    def move(self, twist: Twist, duration: float = 0.0) -> bool:
        """Send a velocity command to the robot.

        Maps Twist fields:
          linear.x  → forward velocity  (±0.2–1.0 m/s or 0)
          linear.y  → lateral velocity  (±0.2–1.0 m/s or 0)
          angular.z → angular velocity  (±0.1–1.0 rad/s or 0)
        """
        if self._vel_publisher is None:
            return False

        from aimdk_msgs.msg import McLocomotionVelocity, MessageHeader

        msg = McLocomotionVelocity()
        msg.header = MessageHeader()
        msg.header.stamp = self._ros_node.get_clock().now().to_msg()
        msg.source = _INPUT_SOURCE_NAME
        msg.forward_velocity = _clamp_velocity(twist.linear.x, _MIN_FORWARD, _MAX_FORWARD)
        msg.lateral_velocity = _clamp_velocity(twist.linear.y, _MIN_LATERAL, _MAX_LATERAL)
        msg.angular_velocity = _clamp_velocity(twist.angular.z, _MIN_ANGULAR, _MAX_ANGULAR)

        self._vel_publisher.publish(msg)

        if duration > 0.0:
            time.sleep(duration)
            self.stop_motion()

        return True

    @rpc
    def stop_motion(self) -> bool:
        """Send a zero-velocity command to stop the robot."""
        from dimos.msgs.geometry_msgs.Twist import Twist as DimTwist
        from dimos.msgs.geometry_msgs.Vector3 import Vector3 as DimVec3

        return self.move(DimTwist(linear=DimVec3(0.0, 0.0, 0.0), angular=DimVec3(0.0, 0.0, 0.0)))

    @skill
    def observe(self) -> Image | None:
        """Returns the latest RGB frame from the robot's head camera.

        Use this skill for visual world queries.
        Returns None if no frame has been received yet.
        """
        return self._latest_video_frame

    # --- Helpers ---

    @staticmethod
    def _import_msg(module: str, cls: str) -> Any:
        import importlib

        return getattr(importlib.import_module(module), cls)
