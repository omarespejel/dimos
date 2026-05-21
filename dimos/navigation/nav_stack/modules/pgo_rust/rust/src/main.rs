// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0

// Library-style modules expose a wider API than the binary's main loop
// currently uses (e.g. scan_context's sector helpers, StubOptimizer for tests
// of orchestration only).  Suppress dead-code warnings rather than mask the
// surface — the unused functions are intentionally part of the module API.
#![allow(dead_code)]

mod gtsam_ffi;
mod icp;
mod lcm_conv;
mod local_msgs;
mod optimizer;
mod pgo;
mod scan_context;
mod voxel_grid;

use dimos_lcm::LcmOptions;
use dimos_module::{run, Input, LcmTransport, Module, Output};
use lcm_msgs::geometry_msgs::{Quaternion, Transform, TransformStamped, Vector3};
use lcm_msgs::nav_msgs::Odometry;
use lcm_msgs::sensor_msgs::PointCloud2;
use lcm_msgs::std_msgs::{Header, Time};
use lcm_msgs::tf2_msgs::TFMessage;
use local_msgs::{Graph3D, GraphDelta3D};
use nalgebra::Isometry3;
use optimizer::GtsamOptimizer;
use pgo::PgoState;
use serde::Deserialize;

// Same relinearization threshold as the C++ build (cpp/simple_pgo.cpp).
const ISAM2_RELINEARIZE_THRESHOLD: f64 = 0.01;

// All fields default. Python's `NativeModuleConfig.to_config_dict()` excludes
// fields inherited from `ModuleConfig` (`frame_id`, `tf_transport`, etc.) from
// the stdin JSON, so the Rust side must tolerate their absence. Unknown extra
// keys are also accepted — defensive against future Python-side additions.
#[derive(Debug, Deserialize)]
#[serde(default)]
struct Config {
    frame_id: String,
    child_frame_id: String,
    parent_frame: String,
    body_frame: String,
    tf_channel: String,

    key_pose_delta_deg: f64,
    key_pose_delta_trans: f64,

    loop_search_radius: f64,
    loop_time_thresh: f64,
    loop_frame_gap: u64,
    loop_score_thresh: f32,
    loop_submap_half_range: i32,
    submap_resolution: f64,
    min_loop_detect_duration: f64,
    loop_candidate_max_distance_m: f64,

    unregister_input: bool,

    global_map_voxel_size: f64,
    global_map_publish_rate: f32,

    use_scan_context: bool,
    scan_context_num_rings: i32,
    scan_context_num_sectors: i32,
    scan_context_max_range_m: f64,
    scan_context_top_k: i32,
    scan_context_match_threshold: f32,
    scan_context_lidar_height_m: f64,

    debug: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            // Frame chain: parent_frame → frame_id → child_frame_id → body_frame
            // Identity:           world  →  map     →   odom            (corrected_odometry's child_frame_id is body_frame)
            // pgo publishes parent→world (identity anchor) + world→odom
            // (SLAM correction). Upstream odometry publishes odom→body.
            // Matches cpp/pgo_cpp main's parent/world/local/body model.
            parent_frame: "world".to_string(),
            frame_id: "map".to_string(),
            child_frame_id: "odom".to_string(),
            body_frame: "base_link".to_string(),
            tf_channel: "/tf#tf2_msgs.TFMessage".to_string(),
            key_pose_delta_deg: 10.0,
            key_pose_delta_trans: 0.5,
            loop_search_radius: 1.0,
            loop_time_thresh: 60.0,
            loop_frame_gap: 0,
            loop_score_thresh: 0.15,
            loop_submap_half_range: 5,
            submap_resolution: 0.1,
            min_loop_detect_duration: 5.0,
            loop_candidate_max_distance_m: 30.0,
            unregister_input: true,
            global_map_voxel_size: 0.1,
            global_map_publish_rate: 1.0,
            use_scan_context: true,
            scan_context_num_rings: 20,
            scan_context_num_sectors: 60,
            scan_context_max_range_m: 80.0,
            scan_context_top_k: 10,
            scan_context_match_threshold: 0.4,
            scan_context_lidar_height_m: 2.0,
            debug: false,
        }
    }
}

fn ts_to_time(ts: f64) -> Time {
    let sec = ts.trunc();
    Time {
        sec: sec as i32,
        nsec: ((ts - sec) * 1e9).round() as i32,
    }
}

fn build_tf(
    iso: &Isometry3<f64>,
    ts: f64,
    frame_id: &str,
    child_frame_id: &str,
) -> TransformStamped {
    let t = iso.translation.vector;
    let q = iso.rotation.into_inner();
    TransformStamped {
        header: Header {
            seq: 0,
            stamp: ts_to_time(ts),
            frame_id: frame_id.to_string(),
        },
        child_frame_id: child_frame_id.to_string(),
        transform: Transform {
            translation: Vector3 {
                x: t.x,
                y: t.y,
                z: t.z,
            },
            rotation: Quaternion {
                x: q.i,
                y: q.j,
                z: q.k,
                w: q.w,
            },
        },
    }
}

/// Mirrors pgo_cpp/cpp/main.cpp::build_tf_message — emits two
/// transforms: identity `parent_frame → frame_id` and the SLAM
/// correction `frame_id → child_frame_id`. Downstream odometry is
/// responsible for `child_frame_id → body_frame`.
fn build_tf_message(
    correction: &Isometry3<f64>,
    ts: f64,
    parent_frame: &str,
    world_frame: &str,
    local_frame: &str,
) -> TFMessage {
    TFMessage {
        transforms: vec![
            build_tf(&Isometry3::identity(), ts, parent_frame, world_frame),
            build_tf(correction, ts, world_frame, local_frame),
        ],
    }
}

#[derive(Module)]
#[module(setup = setup)]
struct PgoRust {
    #[input(decode = PointCloud2::decode, handler = on_registered_scan)]
    registered_scan: Input<PointCloud2>,

    #[input(decode = Odometry::decode, handler = on_odometry)]
    odometry: Input<Odometry>,

    #[output(encode = Odometry::encode)]
    corrected_odometry: Output<Odometry>,

    #[output(encode = PointCloud2::encode)]
    global_map: Output<PointCloud2>,

    #[output(encode = Graph3D::encode)]
    pose_graph: Output<Graph3D>,

    #[output(encode = GraphDelta3D::encode)]
    loop_closure_event: Output<GraphDelta3D>,

    #[output(encode = TFMessage::encode)]
    tf: Output<TFMessage>,

    #[config]
    config: Config,

    state: Option<PgoState>,
    /// Last received odometry, with the message timestamp it arrived under.
    /// We pair scans with this only when the timestamps match — LCM is UDP
    /// multicast and drops messages under load; if we paired by "latest
    /// odom" alone, a dropped odom would attach an older pose to a newer
    /// scan and break the ground-truth position invariant the benchmark
    /// relies on. (See lcm_conv.rs::odometry_to_isometry and the
    /// playback.py odom/scan pair — playback sends both with the same `ts`.)
    last_odometry: Option<(f64, Isometry3<f64>)>,
    /// Monotonic count of registered_scans received. Stored on each keyframe
    /// so loop-eligibility can use frame-index gap (robust to dataset-specific
    /// timestamp regimes) instead of timestamp deltas.
    scan_count: u64,
}

impl PgoRust {
    async fn setup(&mut self) {
        let mut pgo_config = pgo::Config::default();
        pgo_config.key_pose_delta_deg = self.config.key_pose_delta_deg;
        pgo_config.key_pose_delta_trans = self.config.key_pose_delta_trans;
        pgo_config.loop_search_radius = self.config.loop_search_radius;
        pgo_config.loop_time_thresh = self.config.loop_time_thresh;
        pgo_config.loop_frame_gap = self.config.loop_frame_gap;
        pgo_config.loop_score_thresh = self.config.loop_score_thresh;
        pgo_config.loop_submap_half_range = self.config.loop_submap_half_range.max(0) as usize;
        pgo_config.submap_resolution = self.config.submap_resolution;
        pgo_config.min_loop_detect_duration = self.config.min_loop_detect_duration;
        pgo_config.loop_candidate_max_distance_m = self.config.loop_candidate_max_distance_m;
        pgo_config.use_scan_context = self.config.use_scan_context;
        pgo_config.scan_context = scan_context::Config {
            n_rings: self.config.scan_context_num_rings.max(0) as usize,
            n_sectors: self.config.scan_context_num_sectors.max(0) as usize,
            max_range_m: self.config.scan_context_max_range_m,
            candidate_top_k: self.config.scan_context_top_k.max(0) as usize,
            match_threshold: self.config.scan_context_match_threshold,
            lidar_height_m: self.config.scan_context_lidar_height_m,
        };
        pgo_config.global_map_voxel_size = self.config.global_map_voxel_size;
        self.state = Some(PgoState::new(
            pgo_config,
            Box::new(GtsamOptimizer::new(ISAM2_RELINEARIZE_THRESHOLD)),
        ));
        if self.config.debug {
            eprintln!("pgo_rust: initialized with GtsamOptimizer (iSAM2 via cxx FFI)");
        }
        // Override the auto-derived `Output::topic` so TF publishes go to
        // the configured channel (default `/tf#tf2_msgs.TFMessage`). The
        // Python side intentionally doesn't declare `tf: Out[TFMessage]`
        // because that would shadow `Module.tf`, so the macro's
        // default-topic-from-port-name path doesn't reach the right
        // channel. pgo_cpp publishes the same channel via raw
        // `lcm.publish(tf_channel, ...)`.
        self.tf.topic = self.config.tf_channel.clone();

        // Seed an identity `world → odom` so consumers querying
        // `world → body` get an immediate result before the first loop
        // closure fires. Matches pgo_cpp main.cpp behavior.
        let seed_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let seed = build_tf_message(
            &Isometry3::identity(),
            seed_ts,
            &self.config.parent_frame,
            &self.config.frame_id,
            &self.config.child_frame_id,
        );
        let _ = self.tf.publish(&seed).await;

        // Marker the Python NativeModule.start() waits for before declaring
        // the subprocess ready (see ready_timeout_sec). Without this the
        // host races the binary's LCM subscribes, and the first publisher's
        // messages get dropped. C++ pgo emits the same marker from main.cpp.
        eprintln!("[DIMOS_NATIVE_READY]");
    }

    async fn on_odometry(&mut self, msg: Odometry) {
        let pose = lcm_conv::odometry_to_isometry(&msg);
        let ts = msg.header.stamp.sec as f64 + msg.header.stamp.nsec as f64 * 1e-9;
        self.last_odometry = Some((ts, pose));
        let (corrected, pose_offset) = {
            let Some(state) = self.state.as_ref() else {
                return;
            };
            (state.correct(pose), state.pose_offset())
        };

        // Republish corrected odometry — pose_offset is identity until
        // the first loop fires.
        let mut out = msg.clone();
        out.pose.pose.position.x = corrected.translation.vector.x;
        out.pose.pose.position.y = corrected.translation.vector.y;
        out.pose.pose.position.z = corrected.translation.vector.z;
        let rotation = corrected.rotation.into_inner();
        out.pose.pose.orientation.x = rotation.i;
        out.pose.pose.orientation.y = rotation.j;
        out.pose.pose.orientation.z = rotation.k;
        out.pose.pose.orientation.w = rotation.w;
        let _ = self.corrected_odometry.publish(&out).await;

        // Publish the SLAM correction TF (parent→world identity + world→odom
        // correction) alongside corrected_odometry. Downstream consumers that
        // query `world → body` via the TF graph rely on this; pgo_cpp does
        // the same on every scan callback.
        let tf_msg = build_tf_message(
            &pose_offset,
            ts,
            &self.config.parent_frame,
            &self.config.frame_id,
            &self.config.child_frame_id,
        );
        let _ = self.tf.publish(&tf_msg).await;
    }

    async fn on_registered_scan(&mut self, msg: PointCloud2) {
        // Increment per received scan, even when this scan doesn't become a
        // keyframe — the counter is a stable index into the playback stream
        // that the GT criterion (min_frame_gap) also uses.
        let scan_index = self.scan_count;
        self.scan_count += 1;
        let Some(state) = self.state.as_mut() else {
            return;
        };
        let Some((odom_ts, raw_pose)) = self.last_odometry else {
            return;
        };
        // Require the odom and scan timestamps to match within 1 ms — KITTI
        // playback emits both with the SAME `ts` for each frame; an LCM drop
        // that mispairs them would attach the wrong pose to this scan and
        // break the world-frame invariant. The pgo_rust binary is then
        // robust to UDP packet loss instead of silently consuming bad data.
        let scan_ts = msg.header.stamp.sec as f64 + msg.header.stamp.nsec as f64 * 1e-9;
        if (odom_ts - scan_ts).abs() > 1e-3 {
            return;
        }
        if !state.should_add_keyframe(&raw_pose) {
            return;
        }
        let timestamp = scan_ts;
        let body_cloud = if self.config.unregister_input {
            // Scans arrive in world frame; transform back into body frame.
            lcm_conv::point_cloud_to_xyz(&msg)
                .into_iter()
                .map(|point| {
                    let body =
                        raw_pose.inverse() * nalgebra::Point3::new(point[0], point[1], point[2]);
                    [body.x, body.y, body.z]
                })
                .collect()
        } else {
            lcm_conv::point_cloud_to_xyz(&msg)
        };
        let index = state.add_keyframe(body_cloud, raw_pose, timestamp, scan_index);
        if let Some(pair) = state.search_loop_candidate() {
            state.enqueue_loop(pair);
            let _ = state.flush();
            if self.config.debug {
                eprintln!(
                    "pgo_rust: loop closure detected — query {} ↔ candidate {} (score {:.3})",
                    pair.source_index, pair.target_index, pair.score
                );
            }
            let delta = build_loop_closure_event(state, pair);
            let _ = self.loop_closure_event.publish(&delta).await;
        }
        // Publish pose_graph on every keyframe so the scoring module's
        // `id_to_node_ts` map stays populated for every keyframe id —
        // loop edges can reference any historic keyframe, so omitting
        // intermediate node entries would cause the scorer to drop the
        // edge during timestamp→frame_id lookup.
        let graph = build_pose_graph(state, timestamp);
        let _ = self.pose_graph.publish(&graph).await;

        // Publish a downsampled global map periodically — once every N
        // keyframes to keep wire traffic bounded.  N = 5 mirrors the C++
        // default at the 1Hz publish rate when keyframes arrive at ~5Hz.
        if index % 5 == 0 {
            let cloud = build_global_map(state, &msg, &self.config);
            let _ = self.global_map.publish(&cloud).await;
        }

        if self.config.debug && index % 10 == 0 {
            eprintln!("pgo_rust: keyframes = {}", index + 1);
        }
    }
}

fn build_pose_graph(state: &PgoState, timestamp: f64) -> Graph3D {
    let mut graph = Graph3D {
        ts: timestamp,
        nodes: Vec::new(),
        edges: Vec::new(),
    };
    for (index, keyframe) in state.keyframes().iter().enumerate() {
        let pose = state.correct(keyframe.raw_pose);
        let translation = pose.translation.vector;
        let rotation = pose.rotation.into_inner();
        graph.nodes.push(local_msgs::Node3D {
            pose: local_msgs::PoseStamped {
                ts: keyframe.timestamp,
                frame_id: "map".to_string(),
                position: [translation.x, translation.y, translation.z],
                orientation: [rotation.i, rotation.j, rotation.k, rotation.w],
            },
            id: index as u64,
            metadata_id: 0,
        });
    }
    // Odometry edges (consecutive-keyframe `metadata_id=0` edges) were
    // previously emitted on every pose_graph publish for graph completeness.
    // The scoring module filters by `metadata_id == EDGE_LOOP_CLOSURE` and
    // skips odometry edges, so emitting them is pure cost on its callback
    // hot path — at N keyframes the callback iterates N edges PER message,
    // turning the receiver into an O(N^2) bottleneck that starves its LCM
    // subscriber. Drop them; loop edges below carry all the scoring-
    // relevant information.
    for (start_id, end_id) in state.history_pairs() {
        graph.edges.push(local_msgs::Edge {
            start_id: *start_id as u64,
            end_id: *end_id as u64,
            timestamp,
            metadata_id: 1, // loop-closure metadata, matches Python's far_planner enum
        });
    }
    graph
}

fn build_loop_closure_event(state: &PgoState, pair: pgo::LoopPair) -> GraphDelta3D {
    // A single keyframe-with-delta payload: the closed-loop node and the
    // SE(3) transform iSAM2 just applied to it.  Mirrors the C++ shape.
    let mut delta = GraphDelta3D::default();
    if let Some(keyframe) = state.keyframes().get(pair.source_index) {
        let pose = state.correct(keyframe.raw_pose);
        let translation = pose.translation.vector;
        let rotation = pose.rotation.into_inner();
        delta.ts = keyframe.timestamp;
        delta.nodes.push(local_msgs::Node3D {
            pose: local_msgs::PoseStamped {
                ts: keyframe.timestamp,
                frame_id: "map".to_string(),
                position: [translation.x, translation.y, translation.z],
                orientation: [rotation.i, rotation.j, rotation.k, rotation.w],
            },
            id: pair.source_index as u64,
            metadata_id: 0,
        });
        let relative = pair.relative_pose;
        let translation = relative.translation.vector;
        let rotation = relative.rotation.into_inner();
        delta.transforms.push(local_msgs::Transform {
            translation: [translation.x, translation.y, translation.z],
            rotation: [rotation.i, rotation.j, rotation.k, rotation.w],
        });
    }
    delta
}

fn build_global_map(state: &PgoState, template: &PointCloud2, config: &Config) -> PointCloud2 {
    // Concatenate every keyframe's body cloud into world frame, then voxel-
    // downsample at the configured resolution.  Re-uses the template's
    // PointField layout / header for transport-side consistency.
    let mut all_points: Vec<[f64; 3]> = Vec::new();
    for keyframe in state.keyframes() {
        let world_pose = state.correct(keyframe.raw_pose);
        for point in &keyframe.body_cloud {
            let p = world_pose * nalgebra::Point3::new(point[0], point[1], point[2]);
            all_points.push([p.x, p.y, p.z]);
        }
    }
    let downsampled =
        voxel_grid::VoxelGrid::new(config.global_map_voxel_size).downsample(&all_points);
    let mut data = Vec::with_capacity(downsampled.len() * 12);
    for point in &downsampled {
        data.extend_from_slice(&(point[0] as f32).to_le_bytes());
        data.extend_from_slice(&(point[1] as f32).to_le_bytes());
        data.extend_from_slice(&(point[2] as f32).to_le_bytes());
    }
    let mut cloud = template.clone();
    cloud.width = downsampled.len() as i32;
    cloud.height = 1;
    cloud.point_step = 12;
    cloud.row_step = data.len() as i32;
    cloud.data = data;
    // Override fields to a clean xyz-f32 layout independent of input cloud.
    cloud.fields = vec![
        lcm_msgs::sensor_msgs::PointField {
            name: "x".into(),
            offset: 0,
            datatype: 7,
            count: 1,
        },
        lcm_msgs::sensor_msgs::PointField {
            name: "y".into(),
            offset: 4,
            datatype: 7,
            count: 1,
        },
        lcm_msgs::sensor_msgs::PointField {
            name: "z".into(),
            offset: 8,
            datatype: 7,
            count: 1,
        },
    ];
    cloud.is_dense = true;
    cloud
}

#[tokio::main]
async fn main() {
    // 64 MB requested SO_RCVBUF — the kernel doubles this internally so
    // the actual usable buffer is ~128 MB. We explicitly set it (rather
    // than relying on net.core.rmem_default) because setsockopt with a
    // smaller value would otherwise be the only effective change, and
    // we want a known-large buffer regardless of host sysctl state.
    // BufferConfiguratorLinux already sets rmem_max=64 MB so this clamp
    // succeeds on a configured host.
    const RECV_BUF_SIZE_BYTES: usize = 64 * 1024 * 1024;
    let mut opts = LcmOptions::default();
    opts.recv_buf_size = Some(RECV_BUF_SIZE_BYTES);
    let transport = LcmTransport::with_options(opts)
        .await
        .expect("failed to create LCM transport");
    run::<PgoRust, _>(transport)
        .await
        .expect("pgo_rust run failed");
}
