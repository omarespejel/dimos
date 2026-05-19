// Voxel-grid downsampler — matches PCL's VoxelGrid<PointXYZ> behaviour at the
// resolution the C++ PGO uses for global_map (0.1m).  Each voxel's output point
// is the centroid of input points falling in it.
//
// Pure Rust, no external deps beyond std.  ahash would be marginally faster on
// large clouds but isn't worth a build-dep here — global_map is published at
// 1Hz, not per-frame.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct VoxelGrid {
    voxel_size: f64,
}

impl VoxelGrid {
    pub fn new(voxel_size: f64) -> Self {
        assert!(voxel_size > 0.0, "voxel_size must be > 0, got {voxel_size}");
        Self { voxel_size }
    }

    pub fn downsample(&self, cloud: &[[f64; 3]]) -> Vec<[f64; 3]> {
        let mut accum: HashMap<(i64, i64, i64), ([f64; 3], u32)> = HashMap::new();
        let inv = 1.0 / self.voxel_size;
        for point in cloud {
            let key = (
                (point[0] * inv).floor() as i64,
                (point[1] * inv).floor() as i64,
                (point[2] * inv).floor() as i64,
            );
            let entry = accum.entry(key).or_insert(([0.0; 3], 0));
            entry.0[0] += point[0];
            entry.0[1] += point[1];
            entry.0[2] += point[2];
            entry.1 += 1;
        }
        accum
            .into_values()
            .map(|(sum, count)| {
                let c = count as f64;
                [sum[0] / c, sum[1] / c, sum[2] / c]
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_point_in_returns_self() {
        let grid = VoxelGrid::new(0.1);
        let out = grid.downsample(&[[1.234, 5.678, -2.0]]);
        assert_eq!(out.len(), 1);
        let p = out[0];
        assert!((p[0] - 1.234).abs() < 1e-12);
        assert!((p[1] - 5.678).abs() < 1e-12);
        assert!((p[2] - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn two_points_same_voxel_centroid() {
        let grid = VoxelGrid::new(1.0);
        let out = grid.downsample(&[[0.2, 0.0, 0.0], [0.8, 0.0, 0.0]]);
        // Both points in voxel (0,0,0) at voxel_size=1.0; centroid is x=0.5.
        assert_eq!(out.len(), 1);
        assert!((out[0][0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn two_points_different_voxels_both_kept() {
        let grid = VoxelGrid::new(1.0);
        let out = grid.downsample(&[[0.2, 0.0, 0.0], [1.8, 0.0, 0.0]]);
        // x=0.2 → voxel (0,0,0); x=1.8 → voxel (1,0,0)
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn many_points_collapse_to_one() {
        let grid = VoxelGrid::new(1.0);
        let cloud: Vec<[f64; 3]> = (0..1000).map(|i| [i as f64 * 0.0005, 0.0, 0.0]).collect();
        // All within x ∈ [0, 0.4995) → voxel (0,0,0)
        let out = grid.downsample(&cloud);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn empty_input_empty_output() {
        let grid = VoxelGrid::new(0.1);
        assert!(grid.downsample(&[]).is_empty());
    }

    #[test]
    #[should_panic(expected = "voxel_size must be > 0")]
    fn zero_voxel_size_panics() {
        let _ = VoxelGrid::new(0.0);
    }

    #[test]
    fn negative_coordinate_handled() {
        let grid = VoxelGrid::new(1.0);
        // x = -0.5 → floor(-0.5) = -1 → voxel (-1, 0, 0)
        // x = -1.5 → floor(-1.5) = -2 → voxel (-2, 0, 0)
        // These should be separate voxels.
        let out = grid.downsample(&[[-0.5, 0.0, 0.0], [-1.5, 0.0, 0.0]]);
        assert_eq!(out.len(), 2);
    }
}
