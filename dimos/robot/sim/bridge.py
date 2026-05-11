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

"""DimSim bridge — NativeModule wrapper around the browser-based simulator.

Architecture:
    DimSim subprocess (deno + headless Chromium + Rapier3D) publishes on
    LCM channels. NativeModule's lifecycle manages the subprocess; this
    bridge exposes In/Out ports and remaps DimSim's internal channels to
    the blueprint-resolved port topics.

    Ports whose native type matches DimSim's output (lidar PointCloud2,
    cmd_vel Twist) pass through directly via DimSim's --topic-remap flag.
    Ports that need type conversion (odom PoseStamped → Odometry,
    color_image JPEG → Image, synthesized CameraInfo) are handled by a
    separate ``DimSimAdapter`` module — see ``dimos/robot/sim/adapter.py``.

Usage::

    from dimos.core.coordination.blueprints import autoconnect
    from dimos.robot.sim.bridge import DimSimBridge
    from dimos.robot.sim.adapter import DimSimAdapter

    autoconnect(
        DimSimBridge.blueprint(scene="apt", vehicle_height=0.3),
        DimSimAdapter.blueprint(camera_fov=46),
        consumer(),
    ).build().loop()
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from pydantic import Field

from dimos.core.core import rpc
from dimos.core.native_module import NativeModule, NativeModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_GITHUB_REPO = "jeff-hykin/DimSim"
_RELEASES_API = f"https://api.github.com/repos/{_GITHUB_REPO}/releases/latest"

# Camera defaults (DimSim: 640x288, configurable FOV)
_CAM_W = 640
_CAM_H = 288
_DEFAULT_FOV_DEG = int(os.environ.get("DIMSIM_CAMERA_FOV", "46"))


def _detect_gpu() -> bool:
    """Check if a GPU is available for headless rendering."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def _dimsim_bin() -> Path:
    return Path.home() / ".dimsim" / "bin" / "dimsim"


def _deno_home_bin() -> Path:
    return Path.home() / ".deno" / "bin" / "deno"


def _ensure_deno() -> str:
    """Locate or install deno."""
    deno = shutil.which("deno")
    if deno:
        return deno
    home_deno = _deno_home_bin()
    if home_deno.exists():
        deno_dir = str(home_deno.parent)
        current_path = os.environ.get("PATH", "")
        if deno_dir not in current_path.split(os.pathsep):
            os.environ["PATH"] = deno_dir + os.pathsep + current_path
        return str(home_deno)

    logger.info("Installing deno to ~/.deno/bin/...")
    result = subprocess.run(
        ["curl", "-fsSL", "https://deno.land/install.sh"],
        capture_output=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to download installer: {result.stderr.decode()}")
    install = subprocess.run(
        ["sh", "-s", "--", "-y"],
        input=result.stdout,
        env={**os.environ, "DENO_INSTALL": str(Path.home() / ".deno")},
        capture_output=True, timeout=120,
    )
    if install.returncode != 0:
        raise RuntimeError(f"Installer failed: {install.stderr.decode()}")

    deno_dir = str(home_deno.parent)
    current_path = os.environ.get("PATH", "")
    if deno_dir not in current_path.split(os.pathsep):
        os.environ["PATH"] = deno_dir + os.pathsep + current_path
    return str(home_deno)


def _find_local_cli() -> Path | None:
    """Find local DimSim/dimos-cli/cli.ts for development.

    Checks ~/repos/DimSim and <dimos-repo-root>/../DimSim. If neither exists,
    auto-clones jeff-hykin/DimSim into ~/repos/DimSim.
    """
    home_repo = Path.home() / "repos" / "DimSim"
    home_cli = home_repo / "dimos-cli" / "cli.ts"
    if home_cli.exists():
        return home_cli

    sibling_cli = Path(__file__).resolve().parents[4] / "DimSim" / "dimos-cli" / "cli.ts"
    if sibling_cli.exists():
        return sibling_cli

    if not shutil.which("git"):
        logger.error("git not found on PATH; cannot auto-clone DimSim")
        return None

    home_repo.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{_GITHUB_REPO}.git"
    logger.info(f"Cloning {_GITHUB_REPO} into {home_repo}")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(home_repo)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error(f"git clone {url} failed: {result.stderr.strip()}")
        return None
    return home_cli if home_cli.exists() else None


# DimSim's pub/sub uses these internal channel names. Port names below
# are picked to match these so the default LCM topic resolution lands on
# the same channels — no --topic-remap needed for the typical blueprint.
# If a blueprint explicitly remaps a port, the override mechanism below
# generates a --topic-remap entry for it.
_PORT_TO_INTERNAL: dict[str, str] = {
    "cmd_vel": "/cmd_vel",
    "odom": "/odom",
    "lidar": "/lidar",
    "color_image": "/color_image",
    "depth_image": "/depth_image",
}


class DimSimBridgeConfig(NativeModuleConfig):
    """Configuration for the DimSim native bridge.

    All fields except ``local`` map to DimSim CLI flags. ``cli_name_override``
    handles snake_case → kebab-case translation; ``cli_exclude`` drops fields
    that don't correspond to a DimSim flag.
    """

    # NativeModuleConfig requires `executable`; it's filled in by resolve().
    executable: str = ""

    scene: str = "apt"
    port: int = 8090

    # Sensor publish rates (ms). None = DimSim defaults.
    image_rate_ms: int | None = None
    lidar_rate_ms: int | None = None
    odom_rate_ms: int | None = None

    enable_depth: bool = True
    camera_fov: int | None = None

    # Pinhole-camera intrinsics consumers care about (forwarded to the
    # DimSimAdapter that synthesizes CameraInfo).
    vehicle_height: float = 0.75

    # Source mode: True = run from ~/repos/DimSim source via deno, else
    # installed binary or auto-install.
    local: bool = False

    # LCM URL override (passed through env, not CLI; informational here).
    lcm_url: str = ""

    cli_name_override: dict[str, str] = Field(default_factory=lambda: {
        "image_rate_ms": "image-rate",
        "lidar_rate_ms": "lidar-rate",
        "odom_rate_ms": "odom-rate",
        "camera_fov": "camera-fov",
    })

    # Fields that aren't DimSim CLI flags — exclude from to_cli_args().
    cli_exclude: frozenset[str] = frozenset({
        "local",
        "vehicle_height",
        "lcm_url",
        "enable_depth",  # mapped to --no-depth manually
        "scene",         # passed positionally
        "port",          # passed positionally
    })


class DimSimBridge(NativeModule):
    """NativeModule wrapper around the DimSim simulator subprocess.

    Outputs publish raw DimSim message types (PoseStamped odom, JPEG-encoded
    images). Consumers that need standard nav-stack types should compose
    with ``DimSimAdapter``, which decodes JPEG and converts PoseStamped to
    Odometry with synthesized velocity.

    Ports (named to match DimSim's native LCM channels so default topic
    resolution lands on the right wire — no remapping needed unless a
    blueprint explicitly assigns a different topic):
        cmd_vel (In[Twist]): velocity commands.
        odom (Out[PoseStamped]): raw pose from DimSim's physics step.
        lidar (Out[PointCloud2]): lidar pointcloud.
        color_image (Out[Image]): RGB camera frame; wire format is JPEG, so
            subscribers must use ``JpegLcmTransport`` (set in the blueprint
            via ``.transports({...})``).
        depth_image (Out[Image]): depth frame (if enabled); also JPEG.
    """

    config: DimSimBridgeConfig

    cmd_vel: In[Twist]
    odom: Out[PoseStamped]
    lidar: Out[PointCloud2]
    # JPEG-encoded on the wire; consumers use JpegLcmTransport to decode.
    color_image: Out[Image]
    depth_image: Out[Image]

    @staticmethod
    def rerun_blueprint() -> Any:
        """3D world view for DimSim visualization."""
        import rerun.blueprint as rrb

        return rrb.Blueprint(
            rrb.Vertical(
                rrb.Spatial3DView(
                    origin="world",
                    name="3D",
                    eye_controls=rrb.EyeControls3D(
                        position=(0.0, 0.0, 20.0),
                        look_target=(0.0, 0.0, 0.0),
                        eye_up=(0.0, 0.0, 1.0),
                    ),
                ),
            ),
            collapse_panels=True,
        )

    @staticmethod
    def rerun_static_pinhole(rr: Any) -> list[Any]:
        """Static Pinhole + Transform3D for the DimSim camera."""
        fov = _DEFAULT_FOV_DEG
        fx = (_CAM_W / 2) / math.tan(math.radians(fov / 2))
        return [
            rr.Pinhole(
                resolution=[_CAM_W, _CAM_H],
                focal_length=[fx, fx],
                principal_point=[_CAM_W / 2, _CAM_H / 2],
                camera_xyz=rr.ViewCoordinates.RDF,
            ),
            rr.Transform3D(
                parent_frame="tf#/sensor",
                translation=[0.3, 0.0, 0.0],
                rotation=rr.Quaternion(xyzw=[0.5, -0.5, 0.5, -0.5]),
            ),
        ]

    @staticmethod
    def rerun_suppress_camera_info(_: Any) -> None:
        return None

    @rpc
    def start(self) -> None:
        # Resolve the executable + interpreter prefix dynamically. Stash
        # the result on the config so NativeModule.start() can use them.
        exe, prefix_args = self._resolve_executable()
        self.config.executable = exe
        self.config.extra_args = self._build_extra_args(prefix_args)
        super().start()

    def _build_extra_args(self, prefix_args: list[str]) -> list[str]:
        """Build the CLI args that follow the executable.

        Combines:
          - prefix_args (for `deno run ...` when using local source)
          - DimSim subcommand + positional config (`dev --scene X --port N`)
          - headless/render flags (default headless; DIMSIM_INTERACTIVE=1 opts out)
          - --topic-remap mapping DimSim's internal channels onto the
            blueprint-resolved port topics
          - rate/depth/fov flags via NativeModuleConfig.to_cli_args()
        """
        cfg = self.config
        args = list(prefix_args)

        scene = os.environ.get("DIMSIM_SCENE", "").strip() or cfg.scene
        args.extend(["dev", "--scene", scene, "--port", str(cfg.port)])

        if not cfg.enable_depth or os.environ.get("DIMSIM_DISABLE_DEPTH", "").strip() in ("1", "true"):
            args.append("--no-depth")

        interactive = os.environ.get("DIMSIM_INTERACTIVE", "").strip() in ("1", "true")
        if not interactive:
            explicit_render = os.environ.get("DIMSIM_RENDER", "").strip()
            if explicit_render:
                render = explicit_render
            elif _detect_gpu():
                render = "gpu"
                logger.info("GPU detected — using GPU rendering for headless DimSim")
            else:
                render = "cpu"
                logger.info("No GPU detected — using CPU rendering (SwiftShader)")
            args.extend(["--headless", "--render", render])

        channels = os.environ.get("DIMSIM_CHANNELS", "").strip()
        if channels:
            args.extend(["--channels", channels])

        # Pass through blueprint-resolved port topics via --topic-remap so
        # DimSim publishes/subscribes on the same channels the rest of the
        # blueprint uses. DimSim auto-appends the `#type.Name` suffix to its
        # internal channel before LCM publish, so the remap value must be
        # the BARE topic — otherwise the suffix would be doubled.
        remap_pairs = []
        for port_name, internal_ch in _PORT_TO_INTERNAL.items():
            topic = self._port_topic(port_name)
            if topic is None:
                continue
            bare_topic = topic.split("#", 1)[0]
            if bare_topic != internal_ch:
                remap_pairs.append(f"{internal_ch}={bare_topic}")
        if remap_pairs:
            args.extend(["--topic-remap", ",".join(remap_pairs)])

        return args

    def _port_topic(self, port_name: str) -> str | None:
        """Read the blueprint-resolved LCM channel for a port (or None)."""
        stream = getattr(self, port_name, None)
        if stream is None:
            return None
        transport = getattr(stream, "_transport", None)
        if transport is None:
            return None
        topic = getattr(transport, "topic", None)
        return str(topic) if topic is not None else None

    def _collect_topics(self) -> dict[str, str]:
        """Suppress NativeModule's default per-port `--<name> <topic>` args.

        DimSim's CLI doesn't recognize them — topic routing is handled via
        --topic-remap in :meth:`_build_extra_args` instead.
        """
        return {}

    def _resolve_executable(self) -> tuple[str, list[str]]:
        use_local = self.config.local or os.environ.get("DIMSIM_LOCAL", "").strip() in ("1", "true")

        if use_local:
            cli_ts = _find_local_cli()
            if not cli_ts:
                raise FileNotFoundError(
                    "Local DimSim not found. Expected DimSim/dimos-cli/cli.ts"
                )
            logger.info(f"Using local DimSim: {cli_ts}")
            return _ensure_deno(), ["run", "--allow-all", "--unstable-net", str(cli_ts)]

        # Not local: ensure binary is installed/up-to-date before resolving.
        self._ensure_installed()

        dimsim = _dimsim_bin()
        if dimsim.exists():
            return str(dimsim), []

        path_dimsim = shutil.which("dimsim")
        if path_dimsim:
            return path_dimsim, []

        raise FileNotFoundError(
            "dimsim not found — run `dimsim setup` or install via deno"
        )

    def _ensure_installed(self) -> None:
        """Download binary + setup + scene install if needed."""
        import json
        import platform
        import stat
        import urllib.request

        scene = os.environ.get("DIMSIM_SCENE", "").strip() or self.config.scene
        dimsim = _dimsim_bin()
        dimsim.parent.mkdir(parents=True, exist_ok=True)

        dimsim_path = str(dimsim) if dimsim.exists() else shutil.which("dimsim")
        installed_ver = None
        if dimsim_path:
            try:
                result = subprocess.run(
                    [dimsim_path, "--version"],
                    capture_output=True, text=True, timeout=5,
                )
                installed_ver = result.stdout.strip() if result.returncode == 0 else None
            except Exception:
                pass

        latest_ver = None
        release_tag = None
        try:
            req = urllib.request.Request(
                _RELEASES_API,
                headers={"Accept": "application/vnd.github.v3+json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                release_tag = data["tag_name"]
                latest_ver = release_tag.lstrip("v")
        except Exception:
            pass

        if not dimsim_path or installed_ver != latest_ver:
            downloaded = False
            if release_tag:
                system = platform.system().lower()
                machine = platform.machine().lower()
                if system == "darwin" and machine in ("arm64", "aarch64"):
                    binary_name = "dimsim-darwin-arm64"
                elif system == "darwin":
                    binary_name = "dimsim-darwin-x64"
                elif system == "linux" and machine in ("x86_64", "amd64"):
                    binary_name = "dimsim-linux-x64"
                else:
                    binary_name = None

                if binary_name:
                    url = (
                        f"https://github.com/{_GITHUB_REPO}/releases/download"
                        f"/{release_tag}/{binary_name}"
                    )
                    try:
                        logger.info(f"Downloading dimsim {latest_ver} for {system}/{machine}...")
                        urllib.request.urlretrieve(url, str(dimsim))
                        dimsim.chmod(
                            dimsim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
                        )
                        if system == "darwin":
                            subprocess.run(["xattr", "-c", str(dimsim)], capture_output=True)

                        file_size = dimsim.stat().st_size
                        if file_size < 1_000_000:
                            dimsim.unlink()
                            raise RuntimeError(
                                f"Downloaded binary too small ({file_size} bytes) — likely a bad download"
                            )
                        smoke = subprocess.run(
                            [str(dimsim), "--version"],
                            capture_output=True, text=True, timeout=10,
                        )
                        if smoke.returncode != 0:
                            dimsim.unlink()
                            raise RuntimeError(
                                f"Binary smoketest failed (exit {smoke.returncode}): {smoke.stderr.strip()}"
                            )
                        logger.info(f"dimsim binary installed (v{smoke.stdout.strip()}, {file_size // 1024}KB).")

                        dimsim_path = str(dimsim)
                        downloaded = True
                    except Exception as exc:
                        logger.warning(f"Binary download failed ({exc}), trying deno fallback...")

            if not downloaded and not dimsim_path:
                deno = _ensure_deno()
                logger.info("Installing dimsim via deno...")
                subprocess.run(
                    [deno, "install", "-gAf", "--reload", "--unstable-net", "jsr:@antim/dimsim"],
                    check=True,
                )
                dimsim_path = shutil.which("dimsim") or str(Path.home() / ".deno" / "bin" / "dimsim")
        else:
            logger.info(f"dimsim up-to-date (v{installed_ver})")

        if not dimsim_path:
            raise FileNotFoundError("dimsim not found")

        local_bin = Path.home() / ".local" / "bin"
        local_bin.mkdir(parents=True, exist_ok=True)
        symlink = local_bin / "dimsim"
        try:
            target = Path(dimsim_path).resolve()
            if symlink.is_symlink() and symlink.resolve() != target:
                symlink.unlink()
            if not symlink.exists():
                symlink.symlink_to(target)
        except OSError:
            pass

        logger.info("Checking core assets...")
        subprocess.run([dimsim_path, "setup"], check=True)
        logger.info(f"Checking scene '{scene}'...")
        subprocess.run([dimsim_path, "scene", "install", scene], check=True)


sim_bridge = DimSimBridge.blueprint

__all__ = ["DimSimBridge", "DimSimBridgeConfig", "sim_bridge"]
