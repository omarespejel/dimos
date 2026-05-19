// Pose-graph optimizer abstraction.
//
// Two impls live alongside this trait:
//   - `StubOptimizer` — records inserts, no-op update.  Used in tests and as
//     the placeholder while bringing up the orchestrator.
//   - `GtsamOptimizer` — wraps the cxx::bridge into libgtsam iSAM2.  This is
//     the production path used in the running binary.

use crate::gtsam_ffi::GtsamBackend;
use nalgebra::Isometry3;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub struct PoseNoise {
    /// Isotropic translation σ in metres.
    pub translation_sigma: f64,
    /// Isotropic rotation σ in radians.
    pub rotation_sigma: f64,
}

impl PoseNoise {
    pub const fn isotropic(translation_sigma: f64, rotation_sigma: f64) -> Self {
        Self { translation_sigma, rotation_sigma }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PoseDelta {
    pub key: u64,
    pub before: Isometry3<f64>,
    pub after: Isometry3<f64>,
}

pub trait GraphOptimizer {
    fn add_prior(&mut self, key: u64, pose: Isometry3<f64>, noise: PoseNoise);
    fn add_between(
        &mut self,
        key_from: u64,
        key_to: u64,
        relative_pose: Isometry3<f64>,
        noise: PoseNoise,
    );
    fn insert_initial(&mut self, key: u64, pose: Isometry3<f64>);

    /// Run one incremental optimization step. Returns the per-key pose deltas
    /// from before → after, for any keys whose estimate changed.
    fn update(&mut self) -> Vec<PoseDelta>;

    fn estimate(&self, key: u64) -> Option<Isometry3<f64>>;
}

/// No-op optimizer.  `update()` always returns an empty delta vector; estimates
/// return whatever was inserted.  Used to test orchestration logic when no real
/// iSAM2 solver is wired up yet.
#[derive(Debug, Default)]
pub struct StubOptimizer {
    values: HashMap<u64, Isometry3<f64>>,
}

impl StubOptimizer {
    pub fn new() -> Self {
        Self::default()
    }
}

impl GraphOptimizer for StubOptimizer {
    fn add_prior(&mut self, _key: u64, _pose: Isometry3<f64>, _noise: PoseNoise) {
        // no-op — stub does not maintain a factor graph
    }

    fn add_between(
        &mut self,
        _key_from: u64,
        _key_to: u64,
        _relative_pose: Isometry3<f64>,
        _noise: PoseNoise,
    ) {
        // no-op
    }

    fn insert_initial(&mut self, key: u64, pose: Isometry3<f64>) {
        self.values.insert(key, pose);
    }

    fn update(&mut self) -> Vec<PoseDelta> {
        Vec::new()
    }

    fn estimate(&self, key: u64) -> Option<Isometry3<f64>> {
        self.values.get(&key).copied()
    }
}

/// iSAM2-backed optimizer.  All add/insert calls stage in a scratch graph + values
/// that `update()` flushes into the solver.  `estimate()` reads from the cached
/// estimate that the solver refreshes after each `update`.
pub struct GtsamOptimizer {
    backend: GtsamBackend,
    last_estimates: HashMap<u64, Isometry3<f64>>,
}

impl GtsamOptimizer {
    pub fn new(relinearize_threshold: f64) -> Self {
        Self {
            backend: GtsamBackend::new(relinearize_threshold),
            last_estimates: HashMap::new(),
        }
    }
}

impl GraphOptimizer for GtsamOptimizer {
    fn add_prior(&mut self, key: u64, pose: Isometry3<f64>, noise: PoseNoise) {
        self.backend.add_prior(key, pose, noise.translation_sigma, noise.rotation_sigma);
    }

    fn add_between(
        &mut self,
        key_from: u64,
        key_to: u64,
        relative_pose: Isometry3<f64>,
        noise: PoseNoise,
    ) {
        self.backend.add_between(
            key_from,
            key_to,
            relative_pose,
            noise.translation_sigma,
            noise.rotation_sigma,
        );
    }

    fn insert_initial(&mut self, key: u64, pose: Isometry3<f64>) {
        self.backend.insert_initial(key, pose);
    }

    fn update(&mut self) -> Vec<PoseDelta> {
        self.backend.update();
        let mut deltas = Vec::new();
        for (key, after) in self.backend.estimate_all() {
            let before = self.last_estimates.insert(key, after).unwrap_or(after);
            // Only emit a delta if the pose actually changed.  Tiny epsilon
            // — 1e-12 m + 1e-12 rad — guards against float jitter on keys
            // that the solver didn't touch this round.
            let translation_delta = (after.translation.vector - before.translation.vector).norm();
            let rotation_delta = (after.rotation * before.rotation.inverse()).angle();
            if translation_delta > 1e-12 || rotation_delta.abs() > 1e-12 {
                deltas.push(PoseDelta { key, before, after });
            }
        }
        deltas
    }

    fn estimate(&self, key: u64) -> Option<Isometry3<f64>> {
        self.last_estimates.get(&key).copied().or_else(|| self.backend.estimate(key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Translation3, UnitQuaternion};

    #[test]
    fn stub_remembers_inserts() {
        let mut optimizer = StubOptimizer::new();
        let pose = Isometry3::from_parts(
            Translation3::new(1.0, 2.0, 3.0),
            UnitQuaternion::identity(),
        );
        optimizer.insert_initial(42, pose);
        let recovered = optimizer.estimate(42).unwrap();
        assert!((recovered.translation.vector.x - 1.0).abs() < 1e-9);
    }

    #[test]
    fn stub_update_returns_no_deltas() {
        let mut optimizer = StubOptimizer::new();
        optimizer.insert_initial(0, Isometry3::identity());
        optimizer.insert_initial(1, Isometry3::identity());
        let deltas = optimizer.update();
        assert!(deltas.is_empty());
    }

    #[test]
    fn stub_estimate_missing_key_is_none() {
        let optimizer = StubOptimizer::new();
        assert!(optimizer.estimate(99).is_none());
    }

    #[test]
    fn add_prior_and_between_are_noops_on_stub() {
        let mut optimizer = StubOptimizer::new();
        // Should not panic, even though no actual graph is maintained.
        optimizer.add_prior(0, Isometry3::identity(), PoseNoise::isotropic(0.1, 0.05));
        optimizer.add_between(
            0,
            1,
            Isometry3::identity(),
            PoseNoise::isotropic(0.1, 0.05),
        );
    }

    // GTSAM-backed tests below require `nix develop` to set up
    // GTSAM_INCLUDE_DIR / GTSAM_LIB_DIR / etc.  build.rs forces libcephes-gtsam
    // + libm into DT_NEEDED via --no-as-needed; without it the IFUNC for `sin`
    // resolves against libm before cephes initializes, segfaulting at startup.
    #[test]
    fn gtsam_solves_trivial_chain() {
        // Three poses linked by odometry, with a tight prior at key 0.
        // After one update, estimate(0) should be near the prior, and
        // estimate(1)/estimate(2) should chain forward through the betweens.
        let mut optimizer = GtsamOptimizer::new(0.01);
        let prior_noise = PoseNoise::isotropic(1e-6, 1e-6);
        let odom_noise = PoseNoise::isotropic(0.05, 0.02);

        let pose_0 = Isometry3::identity();
        let pose_1 = Isometry3::from_parts(
            Translation3::new(1.0, 0.0, 0.0),
            UnitQuaternion::identity(),
        );
        let pose_2 = Isometry3::from_parts(
            Translation3::new(2.0, 0.0, 0.0),
            UnitQuaternion::identity(),
        );

        // Initial guesses match the truth to keep this a sanity check.
        optimizer.add_prior(0, pose_0, prior_noise);
        optimizer.insert_initial(0, pose_0);
        optimizer.add_between(0, 1, pose_1, odom_noise);
        optimizer.insert_initial(1, pose_1);
        optimizer.add_between(1, 2, Isometry3::from_parts(Translation3::new(1.0, 0.0, 0.0), UnitQuaternion::identity()), odom_noise);
        optimizer.insert_initial(2, pose_2);

        let _deltas = optimizer.update();

        let estimate_0 = optimizer.estimate(0).expect("missing 0");
        let estimate_1 = optimizer.estimate(1).expect("missing 1");
        let estimate_2 = optimizer.estimate(2).expect("missing 2");

        assert!(estimate_0.translation.vector.norm() < 1e-4, "pose 0 = {:?}", estimate_0.translation);
        assert!((estimate_1.translation.vector.x - 1.0).abs() < 1e-4, "pose 1.x = {}", estimate_1.translation.vector.x);
        assert!((estimate_2.translation.vector.x - 2.0).abs() < 1e-4, "pose 2.x = {}", estimate_2.translation.vector.x);
    }

    #[test]
    fn gtsam_loop_closure_corrects_drift() {
        // Three-pose chain with deliberate drift between pose 0 → 2 via odometry,
        // then close a loop with a BetweenFactor (2 → 0) saying they should
        // actually coincide.  After update, pose 2 should be pulled back toward 0.
        let mut optimizer = GtsamOptimizer::new(0.01);
        let prior_noise = PoseNoise::isotropic(1e-6, 1e-6);
        let odom_noise = PoseNoise::isotropic(0.5, 0.2);  // loose
        let loop_noise = PoseNoise::isotropic(0.01, 0.01);  // tight

        let identity = Isometry3::identity();
        let drift = Isometry3::from_parts(Translation3::new(1.0, 0.0, 0.0), UnitQuaternion::identity());

        // Odometry says 0 → 1 → 2 (so 2 should be at (2,0,0))
        optimizer.add_prior(0, identity, prior_noise);
        optimizer.insert_initial(0, identity);
        optimizer.insert_initial(1, drift);
        optimizer.add_between(0, 1, drift, odom_noise);
        optimizer.insert_initial(2, Isometry3::from_parts(Translation3::new(2.0, 0.0, 0.0), UnitQuaternion::identity()));
        optimizer.add_between(1, 2, drift, odom_noise);

        // Tight loop-closure factor saying 2 ≡ 0
        optimizer.add_between(2, 0, identity, loop_noise);

        let _deltas = optimizer.update();

        let estimate_2 = optimizer.estimate(2).expect("missing 2");
        // With a tight loop closure (sigma=0.01) competing against loose odometry
        // (sigma=0.5), the solver should pull pose 2 substantially toward 0.
        assert!(
            estimate_2.translation.vector.x < 1.0,
            "expected pose 2 to be pulled toward 0, got x = {}",
            estimate_2.translation.vector.x
        );
    }
}
