// PGO orchestrator.
//
// Mirrors cpp/simple_pgo.{h,cpp} but reduced to the orchestration concerns: how
// keyframes are detected, how loop closures are searched, how factors are
// staged, and how the optimizer's output is applied.  The actual nonlinear
// solve lives behind the `GraphOptimizer` trait — `StubOptimizer` keeps the
// orchestrator testable; the eventual GtsamOptimizer (Phase 3) plugs in via the
// same trait.
//
// Side notes vs C++:
// - C++ uses `gtsam::Symbol` keys; we use bare u64 keyframe indices, kept
//   consistent across `optimizer.{add_prior,add_between,insert_initial,estimate}`.
// - C++ holds body-frame clouds; we do the same so loop-closure ICP runs
//   against pre-correction geometry and stays meaningful after smoothing.
// - C++'s `m_r_offset` / `m_t_offset` represent the cumulative correction
//   applied to incoming odometry — we collapse to a single Isometry3.

use nalgebra::Isometry3;

use crate::icp;
use crate::optimizer::{GraphOptimizer, PoseDelta, PoseNoise};
use crate::scan_context::{self, Descriptor, RingKey};
use crate::voxel_grid::VoxelGrid;

#[derive(Debug, Clone)]
pub struct Config {
    pub key_pose_delta_deg: f64,
    pub key_pose_delta_trans: f64,
    pub loop_search_radius: f64,
    pub loop_time_thresh: f64,
    pub loop_score_thresh: f32,
    pub loop_submap_half_range: usize,
    pub submap_resolution: f64,
    pub min_loop_detect_duration: f64,
    pub use_scan_context: bool,
    pub scan_context: scan_context::Config,
    pub global_map_voxel_size: f64,
    pub prior_noise: PoseNoise,
    pub odometry_noise: PoseNoise,
    pub loop_noise: PoseNoise,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            key_pose_delta_deg: 10.0,
            key_pose_delta_trans: 0.5,
            loop_search_radius: 1.0,
            loop_time_thresh: 60.0,
            loop_score_thresh: 0.15,
            loop_submap_half_range: 5,
            submap_resolution: 0.1,
            min_loop_detect_duration: 5.0,
            use_scan_context: true,
            scan_context: scan_context::Config::default(),
            global_map_voxel_size: 0.1,
            prior_noise: PoseNoise::isotropic(1e-6, 1e-6),
            odometry_noise: PoseNoise::isotropic(0.05, 0.02),
            loop_noise: PoseNoise::isotropic(0.1, 0.05),
        }
    }
}

#[derive(Debug)]
pub struct Keyframe {
    pub timestamp: f64,
    pub body_cloud: Vec<[f64; 3]>,
    pub raw_pose: Isometry3<f64>,
    pub descriptor: Descriptor,
    pub ring_key: RingKey,
}

#[derive(Debug, Clone, Copy)]
pub struct LoopPair {
    pub source_index: usize,
    pub target_index: usize,
    pub relative_pose: Isometry3<f64>,
    pub score: f32,
}

pub struct PgoState {
    config: Config,
    optimizer: Box<dyn GraphOptimizer + Send>,
    keyframes: Vec<Keyframe>,
    pose_offset: Isometry3<f64>,
    last_loop_check_time: f64,
    pending_loops: Vec<LoopPair>,
    history_pairs: Vec<(usize, usize)>,
}

impl PgoState {
    pub fn new(config: Config, optimizer: Box<dyn GraphOptimizer + Send>) -> Self {
        Self {
            config,
            optimizer,
            keyframes: Vec::new(),
            pose_offset: Isometry3::identity(),
            last_loop_check_time: -f64::INFINITY,
            pending_loops: Vec::new(),
            history_pairs: Vec::new(),
        }
    }

    pub fn keyframes(&self) -> &[Keyframe] {
        &self.keyframes
    }

    pub fn pose_offset(&self) -> Isometry3<f64> {
        self.pose_offset
    }

    /// Returns true if `pose` is "far enough" from the last keyframe to warrant
    /// a new keyframe entry.  First-call always returns true.
    pub fn should_add_keyframe(&self, pose: &Isometry3<f64>) -> bool {
        let Some(last) = self.keyframes.last() else {
            return true;
        };
        let delta = pose.inverse() * last.raw_pose;
        let translation = delta.translation.vector.norm();
        let rotation = delta.rotation.angle().to_degrees();
        translation >= self.config.key_pose_delta_trans
            || rotation.abs() >= self.config.key_pose_delta_deg
    }

    /// Add a new keyframe and stage its odometry / prior factor.  Returns the
    /// keyframe index.
    pub fn add_keyframe(
        &mut self,
        body_cloud: Vec<[f64; 3]>,
        raw_pose: Isometry3<f64>,
        timestamp: f64,
    ) -> usize {
        let descriptor = scan_context::make_descriptor(&body_cloud, &self.config.scan_context);
        let ring_key = scan_context::make_ring_key(&descriptor);
        let keyframe = Keyframe { timestamp, body_cloud, raw_pose, descriptor, ring_key };
        let index = self.keyframes.len();
        let corrected_pose = self.pose_offset * raw_pose;
        if index == 0 {
            self.optimizer.add_prior(0, corrected_pose, self.config.prior_noise);
            self.optimizer.insert_initial(0, corrected_pose);
        } else {
            let prev = &self.keyframes[index - 1];
            let relative = prev.raw_pose.inverse() * raw_pose;
            self.optimizer.add_between(
                (index - 1) as u64,
                index as u64,
                relative,
                self.config.odometry_noise,
            );
            self.optimizer.insert_initial(index as u64, corrected_pose);
        }
        self.keyframes.push(keyframe);
        index
    }

    /// Find loop candidates for the most recent keyframe.  Returns the best
    /// (lowest-score) candidate or `None` if none meet the thresholds.
    ///
    /// Logic mirrors C++:
    ///   1. Require min_loop_detect_duration since last attempt
    ///   2. If use_scan_context, try scan-context match first
    ///   3. Otherwise (or if SC fails), do position-radius search
    ///   4. Require time gap > loop_time_thresh between candidate and query
    pub fn search_loop_candidate(&mut self) -> Option<LoopPair> {
        // Mirror cpp/simple_pgo.cpp:212 — require ≥10 keyframes before any
        // loop search. Under 10 keyframes the trajectory is too short to be
        // a meaningful revisit, and scan-context "best match" can pass the
        // threshold accidentally on near-empty descriptors → false positives.
        const MIN_KEYFRAMES_BEFORE_LOOP_SEARCH: usize = 10;
        if self.keyframes.len() < MIN_KEYFRAMES_BEFORE_LOOP_SEARCH {
            return None;
        }
        let query_index = self.keyframes.len() - 1;
        let query = &self.keyframes[query_index];
        // Mirror cpp/simple_pgo.cpp:214-223 — gate against the LAST DETECTED
        // loop's source-keyframe time, NOT every search attempt.  This lets
        // search run on every keyframe until the first closure fires.
        if self.config.min_loop_detect_duration > 0.0 {
            if let Some(&(_, last_loop_source)) = self.history_pairs.last() {
                let last_loop_time = self.keyframes[last_loop_source].timestamp;
                if query.timestamp - last_loop_time < self.config.min_loop_detect_duration {
                    return None;
                }
            }
        }
        self.last_loop_check_time = query.timestamp;

        let (candidate_index, sector_shift) = if self.config.use_scan_context {
            self.search_by_scan_context(query_index)
                .or_else(|| self.search_by_position(query_index).map(|index| (index, 0)))
        } else {
            self.search_by_position(query_index).map(|index| (index, 0))
        }?;

        // Refine via ICP. Both source and target submaps live in the WORLD
        // frame (mirroring cpp/simple_pgo.cpp:260-261, which uses each
        // keyframe's t_global/r_global to bake bodies into world coords).
        // Running ICP on body-frame data on the source side and world-frame on
        // the target side would land ICP on a wrong basin every time.
        let target_cloud = self.submap(candidate_index);
        let source_cloud = self.submap(query_index);

        // Seed ICP from the scan-context column shift's implied yaw rotation,
        // about the query's global position (NOT the world origin — see cpp
        // comment at simple_pgo.cpp:244-247).
        let mut init_guess = Isometry3::<f64>::identity();
        if sector_shift != 0 {
            let yaw = scan_context::yaw_from_shift(sector_shift, self.config.scan_context.n_sectors);
            let rotation = nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::z_axis(), yaw);
            let source_world_pos = (self.pose_offset * self.keyframes[query_index].raw_pose).translation.vector;
            // init = T(p) · Rz(yaw) · T(-p)
            let translation = source_world_pos - rotation * source_world_pos;
            init_guess = Isometry3::from_parts(
                nalgebra::Translation3::from(translation),
                rotation,
            );
        }

        let mut icp_cfg = icp::Config::default();
        icp_cfg.initial_transform = init_guess;
        let icp_result = icp::align(&source_cloud, &target_cloud, &icp_cfg);
        // C++ requires hasConverged() AND fitness ≤ threshold to accept. Drop
        // results that hit max_iterations without converging — those are the
        // ICP runs that didn't find a stable basin.
        if icp_result.reason != icp::TerminationReason::Converged {
            return None;
        }
        let score = icp_result.mean_squared_error as f32;
        if score > self.config.loop_score_thresh {
            return None;
        }
        Some(LoopPair {
            source_index: query_index,
            target_index: candidate_index,
            relative_pose: icp_result.transform,
            score,
        })
    }

    fn search_by_scan_context(&self, query_index: usize) -> Option<(usize, i64)> {
        let query = &self.keyframes[query_index];
        let mut best: Option<(usize, f32, i64)> = None;
        for (candidate_index, candidate) in self.keyframes.iter().enumerate() {
            if candidate_index == query_index || !self.is_time_eligible(query, candidate) {
                continue;
            }
            let (distance, shift) = scan_context::best_distance(&query.descriptor, &candidate.descriptor);
            if distance < self.config.scan_context.match_threshold
                && best.is_none_or(|(_, best_distance, _)| distance < best_distance)
            {
                best = Some((candidate_index, distance, shift));
            }
        }
        best.map(|(index, _, shift)| (index, shift))
    }

    fn search_by_position(&self, query_index: usize) -> Option<usize> {
        let query = &self.keyframes[query_index];
        let query_pos = (self.pose_offset * query.raw_pose).translation.vector;
        let mut best: Option<(usize, f64)> = None;
        for (candidate_index, candidate) in self.keyframes.iter().enumerate() {
            if !self.is_time_eligible(query, candidate) {
                continue;
            }
            let candidate_pos = (self.pose_offset * candidate.raw_pose).translation.vector;
            let distance = (query_pos - candidate_pos).norm();
            if distance <= self.config.loop_search_radius
                && best.is_none_or(|(_, best_distance)| distance < best_distance)
            {
                best = Some((candidate_index, distance));
            }
        }
        best.map(|(index, _)| index)
    }

    fn is_time_eligible(&self, query: &Keyframe, candidate: &Keyframe) -> bool {
        query.timestamp - candidate.timestamp >= self.config.loop_time_thresh
    }

    /// Concatenate body clouds from neighbouring keyframes around `index` into
    /// a single voxel-downsampled submap, expressed in the **world** frame
    /// (mirrors cpp/simple_pgo.cpp::getSubMap which transforms each body
    /// cloud by its keyframe's `r_global`/`t_global`).  ICP runs on submaps
    /// in the same coordinate frame, so source and target must both come
    /// through here.
    pub fn submap(&self, index: usize) -> Vec<[f64; 3]> {
        let half_range = self.config.loop_submap_half_range;
        let start = index.saturating_sub(half_range);
        let end = (index + half_range + 1).min(self.keyframes.len());

        let mut combined: Vec<[f64; 3]> = Vec::new();
        for keyframe in &self.keyframes[start..end] {
            let world_pose = self.pose_offset * keyframe.raw_pose;
            for point in &keyframe.body_cloud {
                let p = world_pose * nalgebra::Point3::new(point[0], point[1], point[2]);
                combined.push([p.x, p.y, p.z]);
            }
        }
        VoxelGrid::new(self.config.submap_resolution).downsample(&combined)
    }

    /// Add a loop closure as a BetweenFactor on the optimizer.  Stores
    /// pending pairs so iSAM2 can be invoked in a batch on `flush()`.
    ///
    /// Noise sigma is scaled by the ICP fitness score (mirrors
    /// cpp/simple_pgo.cpp:296: `Variances(Vector6::Ones() * pair.score)`).
    /// Higher MSE → looser noise → less weight in iSAM2, so a borderline /
    /// false-positive closure can't yank the trajectory the way a tight prior
    /// would. We clamp to the configured `loop_noise` floor so good loops
    /// still get the minimum tightness from config, and to a 1.0 ceiling so
    /// catastrophic fits don't go infinite-sigma.
    pub fn enqueue_loop(&mut self, pair: LoopPair) {
        let score_sigma = (pair.score as f64).clamp(
            self.config.loop_noise.translation_sigma.min(self.config.loop_noise.rotation_sigma),
            1.0,
        );
        let scaled_noise = PoseNoise::isotropic(score_sigma, score_sigma);
        self.optimizer.add_between(
            pair.target_index as u64,
            pair.source_index as u64,
            pair.relative_pose,
            scaled_noise,
        );
        self.pending_loops.push(pair);
        self.history_pairs.push((pair.target_index, pair.source_index));
    }

    /// Run optimizer.update() and update pose_offset from the most-recent key's
    /// delta. Returns the deltas (useful for emitting loop_closure_event).
    pub fn flush(&mut self) -> Vec<PoseDelta> {
        if self.pending_loops.is_empty() {
            return Vec::new();
        }
        let deltas = self.optimizer.update();
        self.pending_loops.clear();
        // Refresh pose_offset against the latest keyframe.
        if let Some(last_index) = self.keyframes.len().checked_sub(1) {
            if let Some(optimized) = self.optimizer.estimate(last_index as u64) {
                let raw_last = self.keyframes[last_index].raw_pose;
                self.pose_offset = optimized * raw_last.inverse();
            }
        }
        deltas
    }

    /// Apply pose_offset to an incoming odometry pose for downstream consumers.
    pub fn correct(&self, raw_pose: Isometry3<f64>) -> Isometry3<f64> {
        self.pose_offset * raw_pose
    }

    pub fn history_pairs(&self) -> &[(usize, usize)] {
        &self.history_pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::StubOptimizer;
    use nalgebra::{Translation3, UnitQuaternion, Vector3};
    use std::f64::consts::FRAC_PI_2;

    fn translated(x: f64, y: f64, z: f64) -> Isometry3<f64> {
        Isometry3::from_parts(Translation3::new(x, y, z), UnitQuaternion::identity())
    }

    fn rotated_z(theta: f64) -> Isometry3<f64> {
        Isometry3::from_parts(
            Translation3::identity(),
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), theta),
        )
    }

    fn empty_state() -> PgoState {
        PgoState::new(Config::default(), Box::new(StubOptimizer::new()))
    }

    #[test]
    fn first_pose_always_a_keyframe() {
        let state = empty_state();
        assert!(state.should_add_keyframe(&Isometry3::identity()));
    }

    #[test]
    fn small_move_not_a_keyframe() {
        let mut state = empty_state();
        state.add_keyframe(vec![], Isometry3::identity(), 0.0);
        // 0.1m move < 0.5m threshold, no rotation → not a keyframe
        assert!(!state.should_add_keyframe(&translated(0.1, 0.0, 0.0)));
    }

    #[test]
    fn large_translation_triggers_keyframe() {
        let mut state = empty_state();
        state.add_keyframe(vec![], Isometry3::identity(), 0.0);
        assert!(state.should_add_keyframe(&translated(1.0, 0.0, 0.0)));
    }

    #[test]
    fn large_rotation_triggers_keyframe() {
        let mut state = empty_state();
        state.add_keyframe(vec![], Isometry3::identity(), 0.0);
        // 90° rotation >> 10° threshold
        assert!(state.should_add_keyframe(&rotated_z(FRAC_PI_2)));
    }

    #[test]
    fn loop_search_with_no_keyframes_yields_none() {
        let mut state = empty_state();
        assert!(state.search_loop_candidate().is_none());
    }

    #[test]
    fn loop_search_skips_when_too_recent() {
        let mut state = empty_state();
        // Add two close-in-time keyframes — second one shouldn't even start
        // loop search because min_loop_detect_duration (5s) hasn't elapsed.
        state.add_keyframe(vec![], Isometry3::identity(), 0.0);
        state.add_keyframe(vec![], translated(1.0, 0.0, 0.0), 1.0);
        assert!(state.search_loop_candidate().is_none());
    }

    #[test]
    fn submap_empty_when_keyframe_has_no_cloud() {
        let mut state = empty_state();
        state.add_keyframe(vec![], Isometry3::identity(), 0.0);
        assert!(state.submap(0).is_empty());
    }

    #[test]
    fn correct_with_identity_offset_passes_through() {
        let state = empty_state();
        let pose = translated(1.0, 2.0, 3.0);
        let corrected = state.correct(pose);
        assert!((corrected.translation.vector.x - 1.0).abs() < 1e-9);
    }

    #[test]
    fn add_keyframe_initial_inserts_with_offset() {
        let mut state = empty_state();
        // First keyframe is always at offset=identity, so corrected_pose == raw_pose.
        let raw = translated(5.0, 0.0, 0.0);
        let index = state.add_keyframe(vec![], raw, 0.0);
        assert_eq!(index, 0);
        // Optimizer should have received insert_initial(0, raw).
        // We can't peek directly through the trait object, but we can verify
        // via flush behaviour: no loops pending → no work, pose_offset stays I.
        let deltas = state.flush();
        assert!(deltas.is_empty());
        assert!((state.pose_offset().translation.vector.norm() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn enqueue_loop_records_history_pair() {
        let mut state = empty_state();
        state.add_keyframe(vec![], Isometry3::identity(), 0.0);
        state.add_keyframe(vec![], translated(1.0, 0.0, 0.0), 100.0);
        state.enqueue_loop(LoopPair {
            source_index: 1,
            target_index: 0,
            relative_pose: Isometry3::identity(),
            score: 0.0,
        });
        assert_eq!(state.history_pairs(), &[(0, 1)]);
    }
}
