// Point-to-point ICP for loop-closure refinement.
//
// The C++ PGO uses PCL's IterativeClosestPoint<PointXYZ> with:
//   - max 50 iterations
//   - 10m maximum correspondence distance
//   - point-to-point (Arun/Horn SVD-based rigid alignment)
//
// We replicate the algorithm against `nalgebra` and `kiddo` here.  The
// math is small and tractable; KISS-ICP was considered but its Rust port
// (ulagbulag/kiss-icp-rs) is minimally documented and effectively a research
// prototype, so we hand-roll instead.

use kiddo::float::kdtree::KdTree;
use kiddo::SquaredEuclidean;
use nalgebra::{Isometry3, Matrix3, Translation3, UnitQuaternion, Vector3};

// kiddo panics when more than 2 × bucket_size points share the same value on
// any one axis (= fall on the same splitting plane). LiDAR scans routinely
// have thousands of ground-plane points with identical z, so the default 32
// triggers the panic on real data. 4096 is large enough that no realistic
// scan hits the limit, and the per-node memory cost is small (a few KB).
const KD_BUCKET_SIZE: usize = 4096;

#[derive(Debug, Clone)]
pub struct Config {
    pub max_iterations: u32,
    pub max_correspondence_distance: f64,
    pub transform_epsilon: f64,
    pub min_correspondences: usize,
    /// Initial alignment transform applied to the source cloud before the
    /// iteration loop starts. Defaults to identity. The PGO loop search seeds
    /// this from the scan-context column-shift's implied yaw rotation, which
    /// dramatically improves convergence on revisits at different headings.
    pub initial_transform: Isometry3<f64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            max_correspondence_distance: 10.0,
            transform_epsilon: 1e-6,
            min_correspondences: 10,
            initial_transform: Isometry3::identity(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationReason {
    Converged,
    MaxIterations,
    TooFewCorrespondences,
    Empty,
}

#[derive(Debug, Clone)]
pub struct IcpResult {
    pub transform: Isometry3<f64>,
    pub iterations: u32,
    pub correspondences: usize,
    pub mean_squared_error: f64,
    pub reason: TerminationReason,
}

pub fn align(source: &[[f64; 3]], target: &[[f64; 3]], config: &Config) -> IcpResult {
    if source.is_empty() || target.is_empty() {
        return IcpResult {
            transform: Isometry3::identity(),
            iterations: 0,
            correspondences: 0,
            mean_squared_error: 0.0,
            reason: TerminationReason::Empty,
        };
    }

    // kiddo panics if more than 2 × bucket_size points share the same
    // coordinate on any single axis (ground-plane scans easily blow the
    // default 32-point limit). KD_BUCKET_SIZE is bumped above; we also
    // dedupe at a 1 µm grid (well below sensor noise) so identical-XYZ
    // duplicates from voxel pre-downsampling don't burn bucket capacity.
    let mut tree: KdTree<f64, u32, 3, KD_BUCKET_SIZE, u32> = KdTree::with_capacity(target.len());
    let mut seen: std::collections::HashSet<(i64, i64, i64)> = std::collections::HashSet::with_capacity(target.len());
    let scale = 1.0e6_f64;
    for (index, point) in target.iter().enumerate() {
        let key = (
            (point[0] * scale).round() as i64,
            (point[1] * scale).round() as i64,
            (point[2] * scale).round() as i64,
        );
        if seen.insert(key) {
            tree.add(point, index as u32);
        }
    }
    let max_sq_dist = config.max_correspondence_distance * config.max_correspondence_distance;

    let mut current = config.initial_transform;
    let mut last_correspondences = 0usize;
    let mut last_mse = f64::INFINITY;
    let mut termination = TerminationReason::MaxIterations;

    for iter in 0..config.max_iterations {
        let mut src_pairs: Vec<Vector3<f64>> = Vec::new();
        let mut tgt_pairs: Vec<Vector3<f64>> = Vec::new();
        let mut squared_error_sum = 0.0;

        for source_point in source {
            let transformed = current * Vector3::new(source_point[0], source_point[1], source_point[2]).into_point();
            let q = [transformed.x, transformed.y, transformed.z];
            let nearest = tree.nearest_one::<SquaredEuclidean>(&q);
            if nearest.distance > max_sq_dist {
                continue;
            }
            let target_point = target[nearest.item as usize];
            src_pairs.push(transformed.coords);
            tgt_pairs.push(Vector3::new(target_point[0], target_point[1], target_point[2]));
            squared_error_sum += nearest.distance;
        }

        last_correspondences = src_pairs.len();
        if last_correspondences < config.min_correspondences {
            termination = TerminationReason::TooFewCorrespondences;
            break;
        }
        last_mse = squared_error_sum / last_correspondences as f64;

        let delta = solve_rigid(&src_pairs, &tgt_pairs);
        current = delta * current;

        let translation_delta = delta.translation.vector.norm();
        let rotation_delta = delta.rotation.angle();
        if translation_delta < config.transform_epsilon && rotation_delta < config.transform_epsilon {
            termination = TerminationReason::Converged;
            return IcpResult {
                transform: current,
                iterations: iter + 1,
                correspondences: last_correspondences,
                mean_squared_error: last_mse,
                reason: termination,
            };
        }
    }

    IcpResult {
        transform: current,
        iterations: config.max_iterations,
        correspondences: last_correspondences,
        mean_squared_error: last_mse,
        reason: termination,
    }
}

// Arun / Horn closed-form rigid alignment via SVD.
// Returns the transform `T` such that T * src[i] ≈ tgt[i] minimizing MSE.
fn solve_rigid(src: &[Vector3<f64>], tgt: &[Vector3<f64>]) -> Isometry3<f64> {
    debug_assert_eq!(src.len(), tgt.len());
    let n = src.len() as f64;
    let src_centroid: Vector3<f64> = src.iter().sum::<Vector3<f64>>() / n;
    let tgt_centroid: Vector3<f64> = tgt.iter().sum::<Vector3<f64>>() / n;

    let mut covariance = Matrix3::zeros();
    for (s, t) in src.iter().zip(tgt.iter()) {
        let ds = s - src_centroid;
        let dt = t - tgt_centroid;
        covariance += ds * dt.transpose();
    }

    let svd = covariance.svd(true, true);
    let u = svd.u.expect("SVD U missing");
    let v_t = svd.v_t.expect("SVD V^T missing");

    let mut s_sign = Matrix3::identity();
    let det_sign = (v_t.transpose() * u.transpose()).determinant();
    if det_sign < 0.0 {
        s_sign[(2, 2)] = -1.0;
    }
    let rotation_matrix = v_t.transpose() * s_sign * u.transpose();
    let rotation = UnitQuaternion::from_matrix(&rotation_matrix);
    let translation = tgt_centroid - rotation * src_centroid;
    Isometry3::from_parts(Translation3::from(translation), rotation)
}

// Helper trait to coerce Vector3 into Point3 without an extra import dance.
trait IntoPoint {
    fn into_point(self) -> nalgebra::Point3<f64>;
}
impl IntoPoint for Vector3<f64> {
    fn into_point(self) -> nalgebra::Point3<f64> {
        nalgebra::Point3::from(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_4;

    // Deterministic LCG so tests don't pull rand as a dep.  Generates a cloud
    // of unique random-ish points — ICP needs unambiguous correspondences for
    // small initial misalignments; a lattice would alias the nearest match.
    fn pseudo_random_cloud(n: usize) -> Vec<[f64; 3]> {
        let mut state: u64 = 0xDEADBEEFCAFEBABE;
        let mut next = || -> f64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / (u32::MAX as f64) * 10.0 - 5.0  // ∈ [-5, 5]
        };
        (0..n).map(|_| [next(), next(), next()]).collect()
    }

    fn apply(transform: &Isometry3<f64>, cloud: &[[f64; 3]]) -> Vec<[f64; 3]> {
        cloud
            .iter()
            .map(|p| {
                let v = transform * nalgebra::Point3::new(p[0], p[1], p[2]);
                [v.x, v.y, v.z]
            })
            .collect()
    }

    #[test]
    fn identity_converges_at_zero() {
        let cloud = pseudo_random_cloud(100);
        let result = align(&cloud, &cloud, &Config::default());
        // Both translation and rotation should be near identity.
        assert!(
            result.transform.translation.vector.norm() < 1e-6,
            "got translation {:?}", result.transform.translation
        );
        let angle = result.transform.rotation.angle();
        assert!(angle.abs() < 1e-6, "angle = {angle}");
    }

    #[test]
    fn pure_translation_recovered() {
        let target = pseudo_random_cloud(100);
        let truth = Isometry3::from_parts(Translation3::new(0.2, -0.3, 0.05), UnitQuaternion::identity());
        let source = apply(&truth.inverse(), &target);
        let result = align(&source, &target, &Config::default());
        let translation = result.transform.translation.vector;
        assert!((translation.x - 0.2).abs() < 1e-4, "tx = {}", translation.x);
        assert!((translation.y - (-0.3)).abs() < 1e-4, "ty = {}", translation.y);
        assert!((translation.z - 0.05).abs() < 1e-4, "tz = {}", translation.z);
    }

    #[test]
    fn small_rotation_recovered() {
        let target = pseudo_random_cloud(100);
        let true_angle = FRAC_PI_4 / 8.0;  // ~5.6°
        let truth = Isometry3::from_parts(
            Translation3::identity(),
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), true_angle),
        );
        let source = apply(&truth.inverse(), &target);
        let result = align(&source, &target, &Config::default());
        let recovered_angle = result.transform.rotation.angle();
        assert!(
            (recovered_angle - true_angle).abs() < 1e-4,
            "expected ~{true_angle}, got {recovered_angle}"
        );
    }

    #[test]
    fn far_apart_clouds_return_too_few() {
        let source = pseudo_random_cloud(20);
        let target: Vec<[f64; 3]> = pseudo_random_cloud(20)
            .into_iter()
            .map(|p| [p[0] + 100.0, p[1], p[2]])
            .collect();
        let result = align(&source, &target, &Config::default());
        assert_eq!(result.reason, TerminationReason::TooFewCorrespondences);
    }

    #[test]
    fn empty_inputs() {
        let result = align(&[], &[[0.0; 3]], &Config::default());
        assert_eq!(result.reason, TerminationReason::Empty);
        let result = align(&[[0.0; 3]], &[], &Config::default());
        assert_eq!(result.reason, TerminationReason::Empty);
    }
}
