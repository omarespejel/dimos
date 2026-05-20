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

use dimos_module::{run, Input, LcmTransport, Module, Output};
use lcm_msgs::nav_msgs::Odometry;
use lcm_msgs::sensor_msgs::PointCloud2;
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
            frame_id: "map".to_string(),
            child_frame_id: "start_point".to_string(),
            parent_frame: "world".to_string(),
            body_frame: "current_point".to_string(),
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
        let Some(state) = self.state.as_ref() else { return };
        let corrected = state.correct(pose);
        // Republish corrected odometry — pose_offset is identity under the
        // stub optimizer, so this is a pass-through until Phase 3 lands.
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
    }

    async fn on_registered_scan(&mut self, msg: PointCloud2) {
        // Increment per received scan, even when this scan doesn't become a
        // keyframe — the counter is a stable index into the playback stream
        // that the GT criterion (min_frame_gap) also uses.
        let scan_index = self.scan_count;
        self.scan_count += 1;
        let Some(state) = self.state.as_mut() else { return };
        let Some((odom_ts, raw_pose)) = self.last_odometry else { return };
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
                    let body = raw_pose.inverse() * nalgebra::Point3::new(point[0], point[1], point[2]);
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
    let mut graph = Graph3D { ts: timestamp, nodes: Vec::new(), edges: Vec::new() };
    let mut last_index: Option<u64> = None;
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
        if let Some(previous) = last_index {
            graph.edges.push(local_msgs::Edge {
                start_id: previous,
                end_id: index as u64,
                timestamp: keyframe.timestamp,
                metadata_id: 0,
            });
        }
        last_index = Some(index as u64);
    }
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
    let downsampled = voxel_grid::VoxelGrid::new(config.global_map_voxel_size).downsample(&all_points);
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
        lcm_msgs::sensor_msgs::PointField { name: "x".into(), offset: 0, datatype: 7, count: 1 },
        lcm_msgs::sensor_msgs::PointField { name: "y".into(), offset: 4, datatype: 7, count: 1 },
        lcm_msgs::sensor_msgs::PointField { name: "z".into(), offset: 8, datatype: 7, count: 1 },
    ];
    cloud.is_dense = true;
    cloud
}

#[tokio::main]
async fn main() {
    let transport = LcmTransport::new()
        .await
        .expect("failed to create LCM transport");
    run::<PgoRust, _>(transport).await.expect("pgo_rust run failed");
}
