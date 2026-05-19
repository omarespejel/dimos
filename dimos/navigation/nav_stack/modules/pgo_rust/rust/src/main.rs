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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
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
    loop_score_thresh: f32,
    loop_submap_half_range: i32,
    submap_resolution: f64,
    min_loop_detect_duration: f64,

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
    last_odometry: Option<Isometry3<f64>>,
}

impl PgoRust {
    async fn setup(&mut self) {
        let mut pgo_config = pgo::Config::default();
        pgo_config.key_pose_delta_deg = self.config.key_pose_delta_deg;
        pgo_config.key_pose_delta_trans = self.config.key_pose_delta_trans;
        pgo_config.loop_search_radius = self.config.loop_search_radius;
        pgo_config.loop_time_thresh = self.config.loop_time_thresh;
        pgo_config.loop_score_thresh = self.config.loop_score_thresh;
        pgo_config.loop_submap_half_range = self.config.loop_submap_half_range.max(0) as usize;
        pgo_config.submap_resolution = self.config.submap_resolution;
        pgo_config.min_loop_detect_duration = self.config.min_loop_detect_duration;
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
    }

    async fn on_odometry(&mut self, msg: Odometry) {
        let pose = lcm_conv::odometry_to_isometry(&msg);
        self.last_odometry = Some(pose);
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
        let Some(state) = self.state.as_mut() else { return };
        let Some(raw_pose) = self.last_odometry else { return };
        if !state.should_add_keyframe(&raw_pose) {
            return;
        }
        let timestamp = msg.header.stamp.sec as f64 + msg.header.stamp.nsec as f64 * 1e-9;
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
        let index = state.add_keyframe(body_cloud, raw_pose, timestamp);
        if let Some(pair) = state.search_loop_candidate() {
            state.enqueue_loop(pair);
            let _ = state.flush();
            if self.config.debug {
                eprintln!(
                    "pgo_rust: loop closure detected — query {} ↔ candidate {} (score {:.3})",
                    pair.source_index, pair.target_index, pair.score
                );
            }
        }
        if self.config.debug && index % 10 == 0 {
            eprintln!("pgo_rust: keyframes = {}", index + 1);
        }
    }
}

#[tokio::main]
async fn main() {
    let transport = LcmTransport::new()
        .await
        .expect("failed to create LCM transport");
    run::<PgoRust, _>(transport).await.expect("pgo_rust run failed");
}
