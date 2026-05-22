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

const canvas = document.getElementById("renderCanvas");
const statusEl = document.getElementById("status");
const engine = new BABYLON.Engine(canvas, true, {
  preserveDrawingBuffer: false,
  stencil: false,
  antialias: false,
});
const scene = new BABYLON.Scene(engine);
scene.useRightHandedSystem = true;
scene.clearColor = new BABYLON.Color4(0.055, 0.063, 0.078, 1);
scene.skipPointerMovePicking = true;
if (BABYLON.ScenePerformancePriority) {
  scene.performancePriority = BABYLON.ScenePerformancePriority.Aggressive;
}

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
const bodyNodeList = [];
const bodyNameList = [];
const sceneMeshes = [];
const collisionMeshes = [];
const robotMeshes = [];
const maxAutoSceneBytes = 2 * 1024 * 1024 * 1024;

// Havok physics. Right-handed, z-up: gravity is -Z.
// `physicsReady` is a promise that resolves once HavokPhysics() returns and
// the plugin is wired. Consumers (collision-mesh wrapper, entity spawn)
// await it before creating aggregates.
const physicsGravity = new BABYLON.Vector3(0, 0, -9.8);
const physicsReady = (async () => {
  if (typeof HavokPhysics !== "function") {
    console.warn("HavokPhysics not loaded — physics disabled");
    return false;
  }
  try {
    const havok = await HavokPhysics();
    scene.enablePhysics(physicsGravity, new BABYLON.HavokPlugin(true, havok));
    console.log("Havok physics enabled");
    return true;
  } catch (error) {
    console.error("Havok init failed:", error);
    return false;
  }
})();

// Entity world. Browser is authoritative for state; Python pushes
// spawn/despawn/teleport via WS, browser reports per-tick state back.
const entities = new Map();              // entity_id -> {mesh, aggregate, descriptor}
let entityBroadcastIntervalMs = 1000 / 30;   // ~30 Hz default
let lastEntityBroadcast = 0;
const params = new URLSearchParams(window.location.search);
const useRobotMesh = params.get("robot") !== "proxy";
const sceneMode = params.get("scene") || "auto";
const showPerfHud = params.get("perf") === "1";
let sceneConfig = null;
let sceneLoadStarted = false;
let pathMesh = null;
let lidarMesh = null;
let lidarMaterial = null;
let lidarPointCount = 0;
let lidarVisible = true;
const lidarPointSizePx = 5.0;
let clickMode = null;
let navGoalMarker = null;
let pointGoalMarker = null;
let spawnMarker = null;
let latestRootPosition = null;
let sceneDepthEnabled = true;
let sceneWireEnabled = false;
let collisionVisible = false;
let collisionMaterial = null;
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
const pointcloudHeaderBytes = 8;
const robotPoseHeaderBytes = 16;
const stateQueueMaxLength = 8;
const stateImmediateDeltaMs = 100;
const stateMaxLagMs = 100;
const stateEarlyThresholdMs = 3;
const statePacingDamping = 0.95;
const robotRootBodyNames = new Set(["pelvis", "torso_link", "body_1"]);
const posePositionEpsilon = 1e-5;
const poseQuaternionEpsilon = 1e-5;
const queuedStateFrames = [];
const lastAppliedRobotPoses = [];
const stateTiming = {
  previousSourceMs: null,
  lastIdealJsMs: null,
  jsMinusSourceMs: Number.POSITIVE_INFINITY,
};
const streamRef = { worker: null, ready: false };
let lastPathVersion = null;
const perfCounters = {
  lastReportMs: performance.now(),
  frames: 0,
  poseFrames: 0,
  poseBodiesUpdated: 0,
  poseBodiesSkipped: 0,
  pointcloudFrames: 0,
  pointcloudBufferUpdates: 0,
  pointcloudRecreates: 0,
};

const vec3 = (values) => new BABYLON.Vector3(values[0], values[1], values[2]);
const quatWxyz = (values) =>
  new BABYLON.Quaternion(values[1], values[2], values[3], values[0]);

function setStatus(message) {
  statusEl.textContent = message;
}

function updatePerfCounters() {
  perfCounters.frames += 1;
  const now = performance.now();
  const elapsedMs = now - perfCounters.lastReportMs;
  if (elapsedMs < 1000) return;

  const elapsedSeconds = elapsedMs / 1000;
  const summary = [
    `fps=${(perfCounters.frames / elapsedSeconds).toFixed(1)}`,
    `pose=${perfCounters.poseFrames}/s`,
    `bodies=${perfCounters.poseBodiesUpdated}/${perfCounters.poseBodiesSkipped}`,
    `pc=${perfCounters.pointcloudFrames}/s`,
    `pcbuf=${perfCounters.pointcloudBufferUpdates}/${perfCounters.pointcloudRecreates}`,
    `q=${queuedStateFrames.length}`,
  ].join(" ");
  statusEl.title = summary;
  if (showPerfHud) statusEl.textContent = summary;

  perfCounters.lastReportMs = now;
  perfCounters.frames = 0;
  perfCounters.poseFrames = 0;
  perfCounters.poseBodiesUpdated = 0;
  perfCounters.poseBodiesSkipped = 0;
  perfCounters.pointcloudFrames = 0;
  perfCounters.pointcloudBufferUpdates = 0;
  perfCounters.pointcloudRecreates = 0;
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

function createCollisionMaterial() {
  if (collisionMaterial) return collisionMaterial;
  collisionMaterial = new BABYLON.StandardMaterial("collisionDebugMaterial", scene);
  collisionMaterial.diffuseColor = new BABYLON.Color3(0.0, 0.95, 1.0);
  collisionMaterial.emissiveColor = new BABYLON.Color3(0.0, 0.45, 0.55);
  collisionMaterial.alpha = 0.45;
  collisionMaterial.wireframe = true;
  collisionMaterial.backFaceCulling = false;
  collisionMaterial.transparencyMode = BABYLON.Material.MATERIAL_ALPHABLEND;
  return collisionMaterial;
}

function setCollisionVisibility(visible) {
  collisionVisible = visible;
  for (const mesh of collisionMeshes) {
    // Only toggle visibility — setEnabled(false) would also stop the
    // PhysicsAggregate from colliding, which defeats the point of
    // running Havok against the cooked collision scene.
    mesh.visibility = visible ? 1 : 0;
  }
  setButtonActive("toggleCollision", visible);
}

function sceneMaterials() {
  const visited = new Set();
  const materials = [];
  for (const mesh of sceneMeshes) {
    const material = mesh.material;
    if (!material || visited.has(material.uniqueId)) continue;
    visited.add(material.uniqueId);
    materials.push(material);
  }
  return materials;
}

function ensureMaterialDefaults(material) {
  material.metadata = material.metadata || {};
  if (material.metadata.dimosDefaults) return material.metadata.dimosDefaults;
  material.metadata.dimosDefaults = {
    alpha: material.alpha,
    backFaceCulling: material.backFaceCulling,
    disableDepthWrite: material.disableDepthWrite,
    forceDepthWrite: material.forceDepthWrite,
    transparencyMode: material.transparencyMode,
    needDepthPrePass: material.needDepthPrePass,
    metallic: material.metallic,
    roughness: material.roughness,
    environmentIntensity: material.environmentIntensity,
  };
  return material.metadata.dimosDefaults;
}

function editMaterial(material, apply) {
  if (material.unfreeze) material.unfreeze();
  apply(material, ensureMaterialDefaults(material));
  if (material.freeze) material.freeze();
}

function configureStaticSceneMaterial(material) {
  editMaterial(material, (mat) => {
    mat.disableDepthWrite = false;
    mat.forceDepthWrite = true;
    mat.needDepthPrePass = false;
  });
}

function setClickMode(mode) {
  clickMode = clickMode === mode ? null : mode;
  setButtonActive("navClick", clickMode === "nav");
  setButtonActive("pointClick", clickMode === "point");
  setButtonActive("spawnClick", clickMode === "spawn");
  if (clickMode === "nav") setStatus("click nav target");
  if (clickMode === "point") setStatus("click point target");
  if (clickMode === "spawn") setStatus("click spawn point");
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
const spawnMarkerMaterial = markerMaterial(
  "spawnMarkerMaterial",
  new BABYLON.Color3(0.14, 1.0, 0.42),
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
  if (!streamRef.worker || !streamRef.ready) return false;
  streamRef.worker.postMessage({ type: "send_json", payload });
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
  for (const material of sceneMaterials()) {
    editMaterial(material, (mat) => {
      mat.disableDepthWrite = !enabled;
      mat.needDepthPrePass = false;
    });
  }
  setButtonActive("toggleDepth", enabled);
}

function setSceneWireframe(enabled) {
  sceneWireEnabled = enabled;
  for (const material of sceneMaterials()) {
    editMaterial(material, (mat) => {
      mat.wireframe = enabled;
    });
  }
  setButtonActive("toggleWire", enabled);
}

function setForceVisible(enabled) {
  forceVisibleEnabled = enabled;
  for (const material of sceneMaterials()) {
    editMaterial(material, (mat, defaults) => {
      if (!enabled) {
        mat.alpha = defaults.alpha;
        mat.backFaceCulling = defaults.backFaceCulling;
        mat.disableDepthWrite = false;
        mat.forceDepthWrite = true;
        mat.transparencyMode = defaults.transparencyMode;
        mat.needDepthPrePass = false;
        return;
      }
      mat.backFaceCulling = false;
      mat.alpha = 1;
      mat.disableDepthWrite = false;
      mat.forceDepthWrite = true;
      mat.needDepthPrePass = false;
      mat.transparencyMode = BABYLON.Material.MATERIAL_OPAQUE;
      if (mat.albedoColor) {
        mat.metallic = 0;
        mat.roughness = 0.9;
        mat.environmentIntensity = 1;
      }
      if (mat.diffuseColor) {
        mat.diffuseColor = mat.diffuseColor || new BABYLON.Color3(0.8, 0.8, 0.8);
      }
    });
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
    min.x = Math.min(min.x, box.minimumWorld.x);
    min.y = Math.min(min.y, box.minimumWorld.y);
    min.z = Math.min(min.z, box.minimumWorld.z);
    max.x = Math.max(max.x, box.maximumWorld.x);
    max.y = Math.max(max.y, box.maximumWorld.y);
    max.z = Math.max(max.z, box.maximumWorld.z);
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
    if (mesh.material) configureStaticSceneMaterial(mesh.material);
    if (mesh.getTotalVertices && mesh.getTotalVertices() > 0) {
      mesh.computeWorldMatrix(true);
      mesh.refreshBoundingInfo(true);
      mesh.freezeWorldMatrix();
      mesh.doNotSyncBoundingInfo = true;
    }
  }
  root.freezeWorldMatrix();
  setSceneDepthWrite(sceneDepthEnabled);
  setSceneWireframe(sceneWireEnabled);
  setForceVisible(forceVisibleEnabled);
  fitCameraToMeshes(sceneMeshes);
}

async function loadCollisionAsset(config) {
  if (!config.collisionSceneFile) return;
  if (config.collisionSceneFile === config.sceneFile) return;
  if (config.collisionSceneBytes > maxAutoSceneBytes) {
    setStatus("collision exceeds browser load guard");
    return;
  }
  setStatus("loading collision");
  const root = new BABYLON.TransformNode("collisionRoot", scene);
  root.position = vec3(config.scenePosition);
  root.scaling = new BABYLON.Vector3(config.sceneScale, config.sceneScale, config.sceneScale);
  root.rotationQuaternion = quatWxyz(config.sceneWxyz);

  const result = await BABYLON.SceneLoader.ImportMeshAsync(
    null,
    "/assets/",
    config.collisionSceneFile,
    scene,
  );
  for (const light of result.lights || []) light.dispose();
  for (const camera of result.cameras || []) camera.dispose();
  for (const mesh of result.meshes) {
    if (mesh.parent === null) mesh.parent = root;
    if (!mesh.getTotalVertices || mesh.getTotalVertices() === 0) continue;
    mesh.isPickable = true;
    mesh.material = createCollisionMaterial();
    mesh.visibility = collisionVisible ? 1 : 0;
    // Don't setEnabled(false) — Havok needs the mesh active to collide.
    // setCollisionVisibility() toggles only the visibility for display.
    mesh.metadata = { dimosCollisionMesh: true };
    collisionMeshes.push(mesh);
  }
  if (sceneMeshes.length === 0) fitCameraToMeshes(collisionMeshes);

  // Wrap each collision mesh with a static (mass=0) PhysicsAggregate so
  // entities + the character controller actually collide with the scene.
  if (await physicsReady) {
    for (const mesh of collisionMeshes) {
      if (mesh.physicsBody) continue; // idempotent
      try {
        new BABYLON.PhysicsAggregate(mesh, BABYLON.PhysicsShapeType.MESH, { mass: 0 }, scene);
      } catch (error) {
        console.warn(`collision physics aggregate failed for ${mesh.name}:`, error);
      }
    }
  }

  setStatus("live");
}

function pickScenePoint() {
  const ray = scene.createPickingRay(
    scene.pointerX,
    scene.pointerY,
    BABYLON.Matrix.Identity(),
    camera,
  );
  const meshes = collisionMeshes.length > 0 ? collisionMeshes : sceneMeshes;
  if (collisionMeshes.length > 0 && !collisionVisible) {
    for (const mesh of collisionMeshes) mesh.setEnabled(true);
  }
  const pick = scene.pickWithRay(ray, (mesh) => meshes.includes(mesh));
  if (collisionMeshes.length > 0 && !collisionVisible) {
    for (const mesh of collisionMeshes) mesh.setEnabled(false);
  }
  if (pick && pick.hit && pick.pickedPoint) return pick.pickedPoint;
  if (Math.abs(ray.direction.z) < 1e-6) return null;
  const distance = -ray.origin.z / ray.direction.z;
  if (distance <= 0) return null;
  return ray.origin.add(ray.direction.scale(distance));
}

async function loadRobot() {
  setStatus("loading robot");
  const response = await fetch("/robot.json", { cache: "no-store" });
  const payload = await response.json();
  bodyNodeList.length = 0;
  bodyNameList.length = 0;
  lastAppliedRobotPoses.length = 0;
  for (const bodyName of payload.bodyNames) {
    bodyNameList.push(bodyName);
    bodyNodeList.push(ensureBodyNode(bodyName));
  }
  if (!useRobotMesh) return;

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

function poseChanged(poses, offset, previous) {
  if (!previous) return true;
  return (
    Math.abs(poses[offset] - previous[0]) > posePositionEpsilon ||
    Math.abs(poses[offset + 1] - previous[1]) > posePositionEpsilon ||
    Math.abs(poses[offset + 2] - previous[2]) > posePositionEpsilon ||
    Math.abs(poses[offset + 3] - previous[3]) > poseQuaternionEpsilon ||
    Math.abs(poses[offset + 4] - previous[4]) > poseQuaternionEpsilon ||
    Math.abs(poses[offset + 5] - previous[5]) > poseQuaternionEpsilon ||
    Math.abs(poses[offset + 6] - previous[6]) > poseQuaternionEpsilon
  );
}

function rememberAppliedPose(bodyIndex, poses, offset) {
  let previous = lastAppliedRobotPoses[bodyIndex];
  if (!previous) {
    previous = new Float32Array(7);
    lastAppliedRobotPoses[bodyIndex] = previous;
  }
  previous[0] = poses[offset];
  previous[1] = poses[offset + 1];
  previous[2] = poses[offset + 2];
  previous[3] = poses[offset + 3];
  previous[4] = poses[offset + 4];
  previous[5] = poses[offset + 5];
  previous[6] = poses[offset + 6];
}

function copyPackedPoseToNode(node, poses, offset) {
  node.position.copyFromFloats(poses[offset], poses[offset + 1], poses[offset + 2]);
  if (!node.rotationQuaternion) {
    node.rotationQuaternion = BABYLON.Quaternion.Identity();
  }
  node.rotationQuaternion.copyFromFloats(
    poses[offset + 4],
    poses[offset + 5],
    poses[offset + 6],
    poses[offset + 3],
  );
}

function updateLatestRootPosition(position) {
  if (!latestRootPosition) latestRootPosition = BABYLON.Vector3.Zero();
  latestRootPosition.copyFrom(position);
}

function updatePath(path, pathVersion) {
  const hasVersion = Number.isFinite(pathVersion);
  if (hasVersion && pathVersion === lastPathVersion) return;
  if (hasVersion) lastPathVersion = pathVersion;

  if (pathMesh) {
    pathMesh.dispose();
    pathMesh = null;
  }
  if (!path || path.length <= 1) return;

  pathMesh = BABYLON.MeshBuilder.CreateLines(
    "navPath",
    { points: path.map((point) => vec3([point[0], point[1], point[2] + 0.08])) },
    scene,
  );
  pathMesh.color = new BABYLON.Color3(0.15, 0.95, 0.68);
  pathMesh.isPickable = false;
}

function applyRobotPose(payload) {
  const count = Math.min(payload.count, bodyNodeList.length);
  let updated = 0;
  let skipped = 0;
  for (let bodyIndex = 0; bodyIndex < count; bodyIndex += 1) {
    const node = bodyNodeList[bodyIndex];
    const bodyName = bodyNameList[bodyIndex];
    const offset = bodyIndex * 7;
    if (poseChanged(payload.poses, offset, lastAppliedRobotPoses[bodyIndex])) {
      copyPackedPoseToNode(node, payload.poses, offset);
      rememberAppliedPose(bodyIndex, payload.poses, offset);
      updated += 1;
    } else {
      skipped += 1;
    }
    if (robotRootBodyNames.has(bodyName)) {
      updateLatestRootPosition(node.position);
    }
  }

  if (!latestRootPosition && count > 1) {
    updateLatestRootPosition(bodyNodeList[1].position);
  }
  perfCounters.poseFrames += 1;
  perfCounters.poseBodiesUpdated += updated;
  perfCounters.poseBodiesSkipped += skipped;
}

function applyStateMetadata(payload) {
  updatePath(payload.path, payload.path_version);
  _updateSlidersFromState(payload.joints);
}

function queueRobotPose(payload) {
  const nowMs = performance.now();
  const sourceMs = Number.isFinite(payload.time) ? payload.time * 1000 : nowMs;
  stateTiming.jsMinusSourceMs = Math.min(
    nowMs - sourceMs,
    stateTiming.jsMinusSourceMs,
  );

  const previousSourceMs = stateTiming.previousSourceMs ?? sourceMs;
  const sourceDeltaMs = sourceMs - previousSourceMs;
  stateTiming.previousSourceMs = sourceMs;

  let targetMs = nowMs;
  const sourceLagMs = nowMs - stateTiming.jsMinusSourceMs - sourceMs;
  if (
    stateTiming.lastIdealJsMs === null ||
    sourceDeltaMs <= 0 ||
    sourceDeltaMs > stateImmediateDeltaMs ||
    sourceLagMs > stateMaxLagMs
  ) {
    stateTiming.lastIdealJsMs = nowMs;
  } else {
    const idealNextMs = stateTiming.lastIdealJsMs + sourceDeltaMs;
    const timeUntilIdealMs = idealNextMs - nowMs;
    if (timeUntilIdealMs > stateEarlyThresholdMs) {
      targetMs = nowMs + timeUntilIdealMs * statePacingDamping;
      stateTiming.lastIdealJsMs += sourceDeltaMs * statePacingDamping;
    } else {
      stateTiming.lastIdealJsMs = nowMs;
    }
  }

  queuedStateFrames.push({ targetMs, payload });
  while (queuedStateFrames.length > stateQueueMaxLength) {
    queuedStateFrames.shift();
  }
}

function applyQueuedState() {
  const nowMs = performance.now();
  let frame = null;
  while (queuedStateFrames.length > 0 && queuedStateFrames[0].targetMs <= nowMs) {
    frame = queuedStateFrames.shift();
  }
  if (frame) applyRobotPose(frame.payload);
}

function createLidarMaterial() {
  if (lidarMaterial) return lidarMaterial;

  const shaders = BABYLON.Effect.ShadersStore;
  shaders.lidarPointVertexShader = `
    precision highp float;
    attribute vec3 position;
    attribute vec4 color;
    uniform mat4 worldViewProjection;
    uniform float pointSize;
    varying vec4 vColor;

    void main(void) {
      gl_Position = worldViewProjection * vec4(position, 1.0);
      gl_PointSize = pointSize;
      vColor = color;
    }
  `;
  shaders.lidarPointFragmentShader = `
    precision highp float;
    varying vec4 vColor;

    void main(void) {
      vec2 delta = gl_PointCoord - vec2(0.5);
      float radiusSquared = dot(delta, delta);
      if (radiusSquared > 0.25) discard;
      float edgeAlpha = smoothstep(0.25, 0.16, radiusSquared);
      gl_FragColor = vec4(vColor.rgb, vColor.a * edgeAlpha);
    }
  `;
  lidarMaterial = new BABYLON.ShaderMaterial(
    "lidarMaterial",
    scene,
    { vertex: "lidarPoint", fragment: "lidarPoint" },
    {
      attributes: ["position", "color"],
      uniforms: ["worldViewProjection", "pointSize"],
      needAlphaBlending: true,
    },
  );
  lidarMaterial.pointsCloud = true;
  lidarMaterial.setFloat("pointSize", lidarPointSizePx);
  lidarMaterial.backFaceCulling = false;
  return lidarMaterial;
}

function updatePointCloud(payload) {
  const count = payload.count || 0;
  if (count === 0 || !payload.positions || !payload.colors) return;
  perfCounters.pointcloudFrames += 1;

  const positions = payload.positions instanceof Float32Array
    ? payload.positions
    : Float32Array.from(payload.positions);
  const packedColors = payload.colors;
  const colors = new Float32Array(count * 4);
  for (let i = 0; i < count; i += 1) {
    colors[i * 4 + 0] = packedColors[i * 3 + 0] / 255;
    colors[i * 4 + 1] = packedColors[i * 3 + 1] / 255;
    colors[i * 4 + 2] = packedColors[i * 3 + 2] / 255;
    colors[i * 4 + 3] = 0.94;
  }

  if (lidarMesh && lidarPointCount === count) {
    lidarMesh.updateVerticesData(
      BABYLON.VertexBuffer.PositionKind,
      positions,
      true,
      false,
    );
    lidarMesh.updateVerticesData(
      BABYLON.VertexBuffer.ColorKind,
      colors,
      true,
      false,
    );
    lidarMesh.setEnabled(lidarVisible);
    perfCounters.pointcloudBufferUpdates += 1;
    return;
  }

  if (lidarMesh) {
    lidarMesh.dispose();
    lidarMesh = null;
    lidarPointCount = 0;
  }

  const nextMesh = new BABYLON.Mesh("lidarCloud", scene);
  const vertexData = new BABYLON.VertexData();
  vertexData.positions = positions;
  vertexData.colors = colors;
  vertexData.applyToMesh(nextMesh, true);
  nextMesh.hasVertexAlpha = true;
  nextMesh.alwaysSelectAsActiveMesh = true;
  nextMesh.isPickable = false;

  nextMesh.material = createLidarMaterial();
  nextMesh.setEnabled(lidarVisible);

  lidarMesh = nextMesh;
  lidarPointCount = count;
  perfCounters.pointcloudRecreates += 1;
}

// ─── Entity world ───────────────────────────────────────────────────────
//
// Python sends spawn/despawn/set_pose/apply_velocity over WS (JSON,
// dispatched in the worker handler below). State is broadcast back per
// Havok physics tick, throttled to ~30 Hz.

function physicsShapeFromHint(hint) {
  switch (hint) {
    case "box":
      return BABYLON.PhysicsShapeType.BOX;
    case "sphere":
      return BABYLON.PhysicsShapeType.SPHERE;
    case "cylinder":
      return BABYLON.PhysicsShapeType.CYLINDER;
    case "mesh":
    default:
      return BABYLON.PhysicsShapeType.MESH;
  }
}

function buildPrimitiveMesh(descriptor) {
  const ext = descriptor.extents || [];
  switch (descriptor.shape_hint) {
    case "box": {
      const w = ext[0] || 1.0;
      const h = ext[1] || 1.0;
      const d = ext[2] || 1.0;
      return BABYLON.MeshBuilder.CreateBox(
        `entity:${descriptor.entity_id}`,
        { width: w, height: h, depth: d },
        scene,
      );
    }
    case "sphere": {
      const r = ext[0] || 0.5;
      return BABYLON.MeshBuilder.CreateSphere(
        `entity:${descriptor.entity_id}`,
        { diameter: 2 * r },
        scene,
      );
    }
    case "cylinder": {
      const r = ext[0] || 0.5;
      const h = ext[1] || 1.0;
      return BABYLON.MeshBuilder.CreateCylinder(
        `entity:${descriptor.entity_id}`,
        { diameterTop: 2 * r, diameterBottom: 2 * r, height: h },
        scene,
      );
    }
    default:
      return null;
  }
}

function applyPoseWire(mesh, pose) {
  mesh.position.set(pose.x || 0, pose.y || 0, pose.z || 0);
  if (!mesh.rotationQuaternion) {
    mesh.rotationQuaternion = BABYLON.Quaternion.Identity();
  }
  mesh.rotationQuaternion.set(
    pose.qx || 0,
    pose.qy || 0,
    pose.qz || 0,
    pose.qw === undefined ? 1 : pose.qw,
  );
}

async function handleEntitySpawn(payload) {
  const desc = payload.descriptor || {};
  const id = desc.entity_id;
  if (!id) return;
  if (entities.has(id)) {
    // Idempotent re-spawn (e.g. reconnect replay): just apply the pose.
    handleEntitySetPose({ entity_id: id, pose: payload.pose || {} });
    return;
  }
  if (!(await physicsReady)) {
    console.warn(`entity_spawn ${id}: physics not ready, dropping`);
    return;
  }
  if (desc.shape_hint === "mesh") {
    // Deferred: GLB-backed entity meshes. MVP supports primitives only.
    console.warn(`entity_spawn ${id}: shape_hint=mesh not yet supported (MVP: primitives)`);
    return;
  }
  const mesh = buildPrimitiveMesh(desc);
  if (!mesh) {
    console.warn(`entity_spawn ${id}: cannot build shape_hint=${desc.shape_hint}`);
    return;
  }
  applyPoseWire(mesh, payload.pose || {});

  // mass=0 → kinematic (program-driven via set_entity_pose). Matches the
  // semantics in entity.py: dynamic with mass>0 actually gets simulated.
  const wantsKinematic = desc.kind !== "dynamic" || (desc.mass || 0) === 0;
  const mass = wantsKinematic ? 0 : desc.mass;
  let aggregate;
  try {
    aggregate = new BABYLON.PhysicsAggregate(
      mesh,
      physicsShapeFromHint(desc.shape_hint),
      { mass },
      scene,
    );
  } catch (error) {
    console.warn(`entity_spawn ${id}: aggregate failed:`, error);
    mesh.dispose();
    return;
  }
  // Kinematic bodies: keep prestep enabled so set_entity_pose teleports
  // are picked up before the next integration.
  if (wantsKinematic && aggregate.body && "disablePreStep" in aggregate.body) {
    aggregate.body.disablePreStep = false;
  }
  entities.set(id, {
    mesh,
    aggregate,
    descriptor: desc,
    kinematic: wantsKinematic,
  });
}

function handleEntityDespawn(payload) {
  const entry = entities.get(payload.entity_id);
  if (!entry) return;
  try {
    entry.aggregate.dispose();
  } catch (_) {
    // best-effort
  }
  try {
    entry.mesh.dispose();
  } catch (_) {
    // best-effort
  }
  entities.delete(payload.entity_id);
}

function handleEntitySetPose(payload) {
  const entry = entities.get(payload.entity_id);
  if (!entry) return;
  applyPoseWire(entry.mesh, payload.pose || {});
  // For dynamic bodies, zero velocities so the teleport doesn't carry
  // stale momentum into the next step.
  if (!entry.kinematic && entry.aggregate.body) {
    try {
      entry.aggregate.body.setLinearVelocity(BABYLON.Vector3.Zero());
      entry.aggregate.body.setAngularVelocity(BABYLON.Vector3.Zero());
    } catch (_) {
      // ignore
    }
  }
}

function handleEntityApplyVelocity(payload) {
  const entry = entities.get(payload.entity_id);
  if (!entry || entry.kinematic) return;
  const t = payload.twist || {};
  try {
    entry.aggregate.body.setLinearVelocity(
      new BABYLON.Vector3(t.lx || 0, t.ly || 0, t.lz || 0),
    );
    entry.aggregate.body.setAngularVelocity(
      new BABYLON.Vector3(t.ax || 0, t.ay || 0, t.az || 0),
    );
  } catch (error) {
    console.warn(`apply_velocity ${payload.entity_id} failed:`, error);
  }
}

function broadcastEntityStates() {
  if (entities.size === 0) return;
  const now = performance.now();
  if (now - lastEntityBroadcast < entityBroadcastIntervalMs) return;
  lastEntityBroadcast = now;
  const ts = Date.now() / 1000.0;
  const states = [];
  for (const [id, entry] of entities) {
    const p = entry.mesh.position;
    const q = entry.mesh.rotationQuaternion || BABYLON.Quaternion.Identity();
    states.push({
      entity_id: id,
      frame_id: "world",
      ts,
      pose: { x: p.x, y: p.y, z: p.z, qw: q.w, qx: q.x, qy: q.y, qz: q.z },
    });
  }
  sendSocketPayload({ type: "entity_states", states });
}

physicsReady.then((ok) => {
  if (ok) scene.onAfterPhysicsObservable.add(broadcastEntityStates);
});

function connectStreamWorker() {
  const worker = new Worker("/static/stream_worker.js");
  streamRef.worker = worker;
  streamRef.ready = false;
  worker.onmessage = (event) => {
    const message = event.data || {};
    switch (message.type) {
      case "status":
        streamRef.ready = Boolean(message.ready);
        setStatus(message.status);
        break;
      case "state":
        if (message.payload.type === "state") {
          applyStateMetadata(message.payload);
        }
        if (message.payload.type === "pointcloud") {
          updatePointCloud(message.payload);
        }
        if (message.payload.type === "entity_spawn") {
          handleEntitySpawn(message.payload);
        }
        if (message.payload.type === "entity_despawn") {
          handleEntityDespawn(message.payload);
        }
        if (message.payload.type === "entity_set_pose") {
          handleEntitySetPose(message.payload);
        }
        if (message.payload.type === "entity_apply_velocity") {
          handleEntityApplyVelocity(message.payload);
        }
        break;
      case "robot_pose":
        queueRobotPose({
          count: message.count,
          time: message.time,
          poses: new Float32Array(
            message.buffer,
            robotPoseHeaderBytes,
            message.count * 7,
          ),
        });
        break;
      case "pointcloud": {
        const positionLength = message.count * 3;
        const colorOffset = pointcloudHeaderBytes + positionLength * 4;
        updatePointCloud({
          count: message.count,
          positions: new Float32Array(
            message.buffer,
            pointcloudHeaderBytes,
            positionLength,
          ),
          colors: new Uint8Array(message.buffer, colorOffset, positionLength),
        });
        break;
      }
      case "camera":
        updateCameraFrame(message.cameraName, message.buffer, message.jpegOffset);
        break;
      case "error":
        console.warn(message.message);
        break;
      default:
        break;
    }
  };
  worker.onerror = (event) => {
    console.error(event);
    streamRef.ready = false;
    setStatus("stream worker error");
  };
  worker.postMessage({ type: "connect" });
  return worker;
}

// Per-camera state. Each entry tracks its <img> element, label, and
// the last object URL so we can revoke it after the next swap.
const _cameraTargets = {
  // First camera ("camera" by default) lives in the primary panel.
  // Any other named camera (e.g. "workspace") lives in the secondary
  // panel. The mapping is by exact name match so dimos-side renames
  // need to be mirrored here.
  primary: {
    img: () => document.getElementById("cameraImg"),
    label: () => document.getElementById("cameraLabel"),
    panel: () => document.getElementById("cameraPanel"),
    lastUrl: null,
  },
  workspace: {
    img: () => document.getElementById("workspaceImg"),
    label: () => document.getElementById("workspaceLabel"),
    panel: () => document.getElementById("workspacePanel"),
    lastUrl: null,
  },
};

function updateCameraFrame(cameraName, buffer, jpegOffset) {
  const jpegBytes = new Uint8Array(buffer, jpegOffset);
  const blob = new Blob([jpegBytes], { type: "image/jpeg" });
  const url = URL.createObjectURL(blob);

  const target = cameraName === "workspace"
    ? _cameraTargets.workspace
    : _cameraTargets.primary;
  const img = target.img();
  const label = target.label();
  const panel = target.panel();
  if (img) {
    img.src = url;
    if (target.lastUrl) URL.revokeObjectURL(target.lastUrl);
    target.lastUrl = url;
  }
  if (label) label.textContent = cameraName;
  if (panel) panel.dataset.hasFrame = "true";
}

function installClickPublisher() {
  scene.onPointerObservable.add((pointerInfo) => {
    if (pointerInfo.type !== BABYLON.PointerEventTypes.POINTERPICK) return;
    const event = pointerInfo.event;
    if (event.target !== canvas) return;

    const publishNav = clickMode === "nav" || event.shiftKey;
    const publishPoint = clickMode === "point" || event.altKey;
    const publishSpawn = clickMode === "spawn";
    if (!publishNav && !publishPoint && !publishSpawn) return;

    if (!streamRef.ready) return;

    if (publishNav || publishSpawn) {
      const point = pickScenePoint();
      if (!point) return;
      if (publishSpawn) {
        spawnMarker = placeMarker(
          spawnMarker,
          "spawnMarker",
          new BABYLON.Vector3(point.x, point.y, point.z + 0.12),
          spawnMarkerMaterial,
          0.28,
        );
        sendSocketPayload({
          type: "respawn_at",
          point: [point.x, point.y, point.z],
        });
        setClickMode(null);
        setStatus("spawn requested");
        return;
      }
      navGoalMarker = placeMarker(
        navGoalMarker,
        "navGoalMarker",
        new BABYLON.Vector3(point.x, point.y, point.z + 0.08),
        navMarkerMaterial,
        0.22,
      );
      sendSocketPayload({
        type: "clicked_point",
        point: [point.x, point.y, point.z],
      });
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
    sendSocketPayload({
      type: "point_goal",
      point: [point.x, point.y, point.z],
    });
    setClickMode(null);
    setStatus("point target sent");
  });
}

document.getElementById("toggleScene").onclick = () => {
  const visible = document.getElementById("toggleScene").dataset.active !== "true";
  setSceneVisibility(visible);
};
document.getElementById("toggleCollision").onclick = () => {
  setCollisionVisibility(!collisionVisible);
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
document.getElementById("spawnClick").onclick = () => setClickMode("spawn");
document.getElementById("toggleDepth").onclick = () => setSceneDepthWrite(!sceneDepthEnabled);
document.getElementById("toggleWire").onclick = () => setSceneWireframe(!sceneWireEnabled);
document.getElementById("forceVisible").onclick = () => setForceVisible(!forceVisibleEnabled);
document.getElementById("focusRobot").onclick = focusRobot;
document.getElementById("loadScene").onclick = () => {
  if (!sceneConfig) return;
  (async () => {
    try {
      await loadSceneAsset(sceneConfig);
      await loadCollisionAsset(sceneConfig);
    } catch (error) {
      console.error(error);
      setStatus("scene load failed");
    }
  })();
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

(async () => {
  try {
    const config = await loadConfig();
    sceneConfig = config;
    await loadRobot();
    connectStreamWorker();
    installClickPublisher();
    setStatus("live");
    if (sceneMode !== "0" && sceneMode !== "manual") {
      window.setTimeout(async () => {
        try {
          await loadSceneAsset(config);
          await loadCollisionAsset(config);
        } catch (error) {
          console.error(error);
          setStatus("scene load failed");
        }
      }, 0);
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
  // Clear any keys the user was holding when the window lost focus
  // so we don't keep walking on alt-tab. Only publish a stop if drive
  // was actually engaged — otherwise we'd send a spurious zero Twist
  // on every page refresh, which clobbers whatever cmd_vel source
  // is currently driving the robot (Quest joysticks, an agent, etc.)
  // for one tick and shows up as the robot going briefly damp.
  pressedKeys.clear();
  if (driveEnabled) sendDriveCommand(true);
});

function renderFrame() {
  applyQueuedState();
  updateKeyboardCamera();
  sendDriveCommand(false);
  scene.render();
  updatePerfCounters();
}

engine.runRenderLoop(renderFrame);
window.addEventListener("resize", () => engine.resize());
