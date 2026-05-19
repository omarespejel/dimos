// Scan Context — polar-binned lidar place-recognition descriptor.
//
// Rust port of cpp/scan_context.{h,cpp}.  Faithful — same field layout, same
// no-point convention, same column-shifted cosine distance.  Kim & Kim 2018,
// reference impl github.com/irapkaist/scancontext.

use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone)]
pub struct Config {
    pub n_rings: usize,
    pub n_sectors: usize,
    pub max_range_m: f64,
    pub candidate_top_k: usize,
    pub match_threshold: f32,
    pub lidar_height_m: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_rings: 20,
            n_sectors: 60,
            max_range_m: 80.0,
            candidate_top_k: 10,
            match_threshold: 0.4,
            lidar_height_m: 2.0,
        }
    }
}

pub type Descriptor = DMatrix<f32>;
pub type RingKey = DVector<f32>;
pub type SectorKey = DVector<f32>;

pub fn make_descriptor(cloud: &[[f64; 3]], config: &Config) -> Descriptor {
    let rings = config.n_rings;
    let sectors = config.n_sectors;
    let mut descriptor = Descriptor::from_element(rings, sectors, 0.0);
    if rings == 0 || sectors == 0 || config.max_range_m <= 0.0 {
        return descriptor;
    }

    let ring_step = config.max_range_m / rings as f64;
    let sector_step = 2.0 * std::f64::consts::PI / sectors as f64;
    let height_offset = config.lidar_height_m as f32;

    for point in cloud {
        let (x, y, z) = (point[0], point[1], point[2]);
        let range = (x * x + y * y).sqrt();
        if range >= config.max_range_m || range <= 1e-6 {
            continue;
        }

        let ring = (range / ring_step).floor() as i64;
        if ring < 0 || ring as usize >= rings {
            continue;
        }
        let ring = ring as usize;

        let mut azimuth = y.atan2(x);
        if azimuth < 0.0 {
            azimuth += 2.0 * std::f64::consts::PI;
        }
        let mut sector = (azimuth / sector_step).floor() as i64;
        if sector < 0 {
            sector = 0;
        }
        if sector as usize >= sectors {
            sector = sectors as i64 - 1;
        }
        let sector = sector as usize;

        let shifted_z = z as f32 + height_offset;
        let cell_value = if shifted_z > 0.0 { shifted_z } else { 0.0 };
        let cell = descriptor.index_mut((ring, sector));
        if cell_value > *cell {
            *cell = cell_value;
        }
    }
    descriptor
}

pub fn make_ring_key(descriptor: &Descriptor) -> RingKey {
    let rows = descriptor.nrows();
    let cols = descriptor.ncols();
    let mut key = RingKey::zeros(rows);
    if cols == 0 {
        return key;
    }
    for i in 0..rows {
        let row = descriptor.row(i);
        key[i] = row.iter().sum::<f32>() / cols as f32;
    }
    key
}

pub fn make_sector_key(descriptor: &Descriptor) -> SectorKey {
    let rows = descriptor.nrows();
    let cols = descriptor.ncols();
    let mut key = SectorKey::zeros(cols);
    if rows == 0 {
        return key;
    }
    for j in 0..cols {
        let col = descriptor.column(j);
        key[j] = col.iter().sum::<f32>() / rows as f32;
    }
    key
}

pub fn column_cosine_distance(query: &Descriptor, candidate: &Descriptor, shift: i64) -> f32 {
    if query.nrows() != candidate.nrows() || query.ncols() != candidate.ncols() {
        return 2.0;
    }
    let cols = query.ncols() as i64;
    if cols == 0 {
        return 2.0;
    }

    let mut total = 0.0f32;
    let mut valid_cols = 0u32;
    for j in 0..cols {
        let shifted_j = (((j + shift) % cols) + cols) % cols;
        let query_col = query.column(j as usize);
        let candidate_col = candidate.column(shifted_j as usize);
        let query_norm = query_col.norm();
        let candidate_norm = candidate_col.norm();
        if query_norm <= 1e-6 || candidate_norm <= 1e-6 {
            continue;
        }
        let cos_sim = query_col.dot(&candidate_col) / (query_norm * candidate_norm);
        total += 1.0 - cos_sim;
        valid_cols += 1;
    }
    if valid_cols == 0 {
        return 2.0;
    }
    total / valid_cols as f32
}

pub fn best_distance(query: &Descriptor, candidate: &Descriptor) -> (f32, i64) {
    let cols = query.ncols() as i64;
    let mut min_distance = 2.0f32;
    let mut best_shift = 0i64;
    for shift in 0..cols {
        let distance = column_cosine_distance(query, candidate, shift);
        if distance < min_distance {
            min_distance = distance;
            best_shift = shift;
        }
    }
    (min_distance, best_shift)
}

pub fn yaw_from_shift(shift: i64, n_sectors: usize) -> f64 {
    let mut yaw = -2.0 * std::f64::consts::PI * shift as f64 / n_sectors as f64;
    if yaw < -std::f64::consts::PI {
        yaw += 2.0 * std::f64::consts::PI;
    }
    yaw
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config { n_rings: 20, n_sectors: 60, max_range_m: 80.0, ..Default::default() }
    }

    #[test]
    fn empty_cloud_zero_descriptor() {
        let descriptor = make_descriptor(&[], &cfg());
        assert_eq!(descriptor.nrows(), 20);
        assert_eq!(descriptor.ncols(), 60);
        assert!(descriptor.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn point_outside_max_range_skipped() {
        // 100m away, outside 80m max_range
        let descriptor = make_descriptor(&[[100.0, 0.0, 0.0]], &cfg());
        assert!(descriptor.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn point_at_origin_skipped() {
        // range <= 1e-6
        let descriptor = make_descriptor(&[[0.0, 0.0, 0.0]], &cfg());
        assert!(descriptor.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn known_point_lands_in_expected_bin() {
        // Point at (4, 0, 0), 80m max with 20 rings → 4m per ring → ring 1
        // azimuth = 0 → sector 0
        // z = 0 → shifted_z = 0 + 2.0 = 2.0
        let descriptor = make_descriptor(&[[4.0, 0.0, 0.0]], &cfg());
        assert!((descriptor[(1, 0)] - 2.0).abs() < 1e-5);
        // Verify no other cell is set
        for i in 0..20 {
            for j in 0..60 {
                if !(i == 1 && j == 0) {
                    assert_eq!(descriptor[(i, j)], 0.0, "unexpected cell ({i},{j})");
                }
            }
        }
    }

    #[test]
    fn max_z_wins_in_same_cell() {
        let descriptor = make_descriptor(
            &[[4.0, 0.0, 0.5], [4.0, 0.0, 1.5], [4.0, 0.0, -0.5]],
            &cfg(),
        );
        // Expect 1.5 + 2.0 height offset = 3.5 (max wins)
        assert!((descriptor[(1, 0)] - 3.5).abs() < 1e-5);
    }

    #[test]
    fn ring_key_mean_per_row() {
        let mut descriptor = Descriptor::zeros(3, 4);
        descriptor.row_mut(0).fill(2.0);
        descriptor.row_mut(1).fill(0.0);
        descriptor.row_mut(2).fill(4.0);
        let key = make_ring_key(&descriptor);
        assert!((key[0] - 2.0).abs() < 1e-5);
        assert!((key[1] - 0.0).abs() < 1e-5);
        assert!((key[2] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn identical_descriptors_zero_distance() {
        let mut descriptor = Descriptor::zeros(3, 4);
        descriptor[(0, 0)] = 1.0;
        descriptor[(1, 2)] = 2.0;
        let distance = column_cosine_distance(&descriptor, &descriptor, 0);
        assert!(distance.abs() < 1e-5);
    }

    #[test]
    fn shifted_descriptor_zero_distance_at_correct_shift() {
        // Build a descriptor with one populated column, then circularly
        // shift it.  best_distance should report distance ~0 at the shift.
        let mut query = Descriptor::zeros(3, 4);
        query[(0, 0)] = 1.0;
        query[(1, 0)] = 2.0;
        query[(2, 0)] = 3.0;

        let mut candidate = Descriptor::zeros(3, 4);
        candidate[(0, 2)] = 1.0;
        candidate[(1, 2)] = 2.0;
        candidate[(2, 2)] = 3.0;

        // Shifting query by +2 should produce candidate's column-0 perspective.
        // best_distance should find a shift where distance ≈ 0.
        let (distance, _shift) = best_distance(&query, &candidate);
        assert!(distance < 1e-5, "expected near-zero distance, got {distance}");
    }

    #[test]
    fn yaw_from_shift_wraps_to_pm_pi() {
        let pi = std::f64::consts::PI;
        // shift 0 → 0
        assert!((yaw_from_shift(0, 60) - 0.0).abs() < 1e-9);
        // shift 15 of 60 → -2pi*15/60 = -pi/2 → stays as -pi/2
        assert!((yaw_from_shift(15, 60) - (-pi / 2.0)).abs() < 1e-9);
        // shift 45 of 60 → -2pi*45/60 = -3pi/2 → wraps to +pi/2
        assert!((yaw_from_shift(45, 60) - (pi / 2.0)).abs() < 1e-9);
    }
}
