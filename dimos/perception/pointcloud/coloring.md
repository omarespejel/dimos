# Pointcloud coloring

For every lidar frame we need the closest-in-time camera frame so that, with
intrinsics + extrinsics, we can project each point into image space and read
back a colour. Lidar runs at ~7Hz, the camera at ~14Hz, and they're captured
independently — so step one is a streaming temporal alignment.

```python session=coloring
from dimos.memory2.store.sqlite import SqliteStore
from dimos.utils.data import get_data

store = SqliteStore(path=get_data("hk_village1.db"))
lidar = store.streams.lidar
color_image = store.streams.color_image
print(lidar.summary())
print(color_image.summary())
```

```results
Stream("lidar"): 957 items, 2026-05-14 10:15:50 — 2026-05-14 10:18:17 (146.4s)
Stream("color_image"): 1984 items, 2026-05-14 10:15:52 — 2026-05-14 10:18:17 (144.9s)
```

`Stream.align` pairs each primary observation with the nearest one from
`other` within `tolerance` seconds. Streams iterate in ts order on both sides
and the matching is a bounded two-pointer merge — no full materialization,
no per-pair queries.

```python session=coloring
aligned = lidar.align(color_image, tolerance=0.05)
print(aligned.summary())
```

```results
Stream("lidar") | order_by(ts) -> FnIterTransformer(fn=_align): 932 items, 2026-05-14 10:15:52 — 2026-05-14 10:18:17 (144.9s)
```

Each output observation's `data` is a namedtuple keyed by source-stream name —
fully addressable both ways:

```python session=coloring
pair = aligned.first().data
print(f"lidar @ {pair.lidar.ts:.3f}  ↔  image @ {pair.color_image.ts:.3f}")
print(f"Δt = {(pair.color_image.ts - pair.lidar.ts) * 1000:.1f} ms")
print(f"index access works too: pair[0] is pair.lidar -> {pair[0] is pair.lidar}")
```

```results
lidar @ 1778753752.548  ↔  image @ 1778753752.551
Δt = 2.5 ms
index access works too: pair[0] is pair.lidar -> True
```

## Projecting 3D points into the image

For coloring we need a `CameraInfo` (intrinsics + distortion model) plus a
camera pose in the same frame as the points we'll project. The Go2's front
fisheye is an equidistant Kannala-Brandt model — calibration ships with the
repo as a YAML.

```python session=coloring
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.perception.pointcloud.projection import Camera

info = CameraInfo.from_yaml("dimos/robot/unitree/go2/front_camera_720.yaml")
camera = Camera(info=info, pose=Pose())  # identity pose: "points are in camera frame"
print(f"sensor: {info.width}x{info.height}, model={info.distortion_model}")
print(f"K[0,0]={info.K[0]:.1f}  K[1,1]={info.K[4]:.1f}  cx={info.K[2]:.1f}  cy={info.K[5]:.1f}")
```

```results
sensor: 1280x720, model=equidistant
K[0,0]=797.5  K[1,1]=796.5  cx=643.5  cy=349.3
```

Forward projection takes `(N,3)` points and returns `(pixels: (N,2), valid: (N,))`.
Invalid = behind the camera or projected outside the image bounds.

```python session=coloring
import numpy as np

# Synthetic points in the camera optical frame (z forward, x right, y down):
# a target ahead, two off-axis, one behind, one way out of frame.
pts = np.array([
    [0.0, 0.0, 3.0],     # straight ahead at 3m       -> hits (cx, cy)
    [0.3, -0.1, 2.0],    # slightly right + up        -> in-frame
    [-0.4, 0.2, 4.0],    # slightly left + down       -> in-frame
    [0.0, 0.0, -1.0],    # behind camera              -> invalid
    [5.0, 0.0, 1.0],     # 79° off-axis right         -> outside image
])
pixels, valid = camera.project(pts)
for p, (u, v), ok in zip(pts, pixels, valid):
    label = f"({u:7.1f}, {v:7.1f})" if ok else "  invalid"
    print(f"{p.tolist()!s:32}  ->  {label}")
```

```results
[0.0, 0.0, 3.0]                   ->  (  643.5,   349.3)
[0.3, -0.1, 2.0]                  ->  (  762.0,   309.9)
[-0.4, 0.2, 4.0]                  ->  (  564.2,   388.9)
[0.0, 0.0, -1.0]                  ->    invalid
[5.0, 0.0, 1.0]                   ->    invalid
```

Back-projection turns a pixel into a `Ray(origin, direction)`. The center
pixel's ray should point along +z (the optical axis) since this camera is
at the origin with identity orientation.

```python session=coloring
ray = camera.ray(info.K[2], info.K[5])  # ray through (cx, cy)
print(f"origin    = {ray.origin}")
print(f"direction = {ray.direction.round(3)}")
print(f"|dir|     = {np.linalg.norm(ray.direction):.6f}")
```

```results
origin    = [0. 0. 0.]
direction = [0. 0. 1.]
|dir|     = 1.000000
```

Real coloring still needs one more thing: a static `T_camera_lidar` extrinsic
so we can express each lidar point in the camera frame before `project()`.
That goes into the coloring transform itself (next step), which takes the
aligned `(lidar, color_image)` pairs and emits a colored pointcloud.
