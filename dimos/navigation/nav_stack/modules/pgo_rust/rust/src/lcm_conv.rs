// Conversions between lcm-msgs types and our internal representations.
//
// PointCloud2: extract Vec<[f64;3]> from the packed `data` blob by interpreting
// the FieldOffset values for x/y/z. Datatype 7 = FLOAT32, datatype 8 = FLOAT64
// per sensor_msgs/PointField.h.
//
// Odometry: extract translation+quaternion from the nested pose field.

use lcm_msgs::nav_msgs::Odometry;
use lcm_msgs::sensor_msgs::PointCloud2;
use nalgebra::{Isometry3, Translation3, UnitQuaternion};

const DATATYPE_FLOAT32: u8 = 7;
const DATATYPE_FLOAT64: u8 = 8;

#[derive(Debug, Clone, Copy)]
struct FieldOffset {
    offset: usize,
    datatype: u8,
}

fn find_xyz_offsets(cloud: &PointCloud2) -> Option<(FieldOffset, FieldOffset, FieldOffset)> {
    let mut x: Option<FieldOffset> = None;
    let mut y: Option<FieldOffset> = None;
    let mut z: Option<FieldOffset> = None;
    for field in &cloud.fields {
        let entry = FieldOffset { offset: field.offset as usize, datatype: field.datatype };
        match field.name.as_str() {
            "x" => x = Some(entry),
            "y" => y = Some(entry),
            "z" => z = Some(entry),
            _ => {}
        }
    }
    Some((x?, y?, z?))
}

fn read_field(blob: &[u8], offset_in_point: usize, field: FieldOffset) -> Option<f64> {
    let abs_offset = offset_in_point + field.offset;
    match field.datatype {
        DATATYPE_FLOAT32 => {
            let bytes = blob.get(abs_offset..abs_offset + 4)?;
            Some(f32::from_le_bytes(bytes.try_into().ok()?) as f64)
        }
        DATATYPE_FLOAT64 => {
            let bytes = blob.get(abs_offset..abs_offset + 8)?;
            Some(f64::from_le_bytes(bytes.try_into().ok()?))
        }
        _ => None,
    }
}

pub fn point_cloud_to_xyz(cloud: &PointCloud2) -> Vec<[f64; 3]> {
    let Some((fx, fy, fz)) = find_xyz_offsets(cloud) else {
        return Vec::new();
    };
    let point_step = cloud.point_step as usize;
    if point_step == 0 {
        return Vec::new();
    }
    let n_points = cloud.data.len() / point_step;
    let mut points = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let base = i * point_step;
        let Some(x) = read_field(&cloud.data, base, fx) else { continue };
        let Some(y) = read_field(&cloud.data, base, fy) else { continue };
        let Some(z) = read_field(&cloud.data, base, fz) else { continue };
        if x.is_finite() && y.is_finite() && z.is_finite() {
            points.push([x, y, z]);
        }
    }
    points
}

pub fn odometry_to_isometry(odom: &Odometry) -> Isometry3<f64> {
    let position = &odom.pose.pose.position;
    let orientation = &odom.pose.pose.orientation;
    let translation = Translation3::new(position.x, position.y, position.z);
    let rotation = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
        orientation.w,
        orientation.x,
        orientation.y,
        orientation.z,
    ));
    Isometry3::from_parts(translation, rotation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lcm_msgs::sensor_msgs::PointField;

    fn pc_with_xyz_f32(points: &[[f32; 3]]) -> PointCloud2 {
        let mut data = Vec::with_capacity(points.len() * 12);
        for point in points {
            data.extend_from_slice(&point[0].to_le_bytes());
            data.extend_from_slice(&point[1].to_le_bytes());
            data.extend_from_slice(&point[2].to_le_bytes());
        }
        let mut cloud = PointCloud2::default();
        cloud.point_step = 12;
        cloud.row_step = data.len() as i32;
        cloud.width = points.len() as i32;
        cloud.height = 1;
        cloud.data = data;
        cloud.fields = vec![
            PointField { name: "x".into(), offset: 0, datatype: DATATYPE_FLOAT32, count: 1 },
            PointField { name: "y".into(), offset: 4, datatype: DATATYPE_FLOAT32, count: 1 },
            PointField { name: "z".into(), offset: 8, datatype: DATATYPE_FLOAT32, count: 1 },
        ];
        cloud
    }

    #[test]
    fn extract_f32_xyz() {
        let pc = pc_with_xyz_f32(&[[1.0, 2.0, 3.0], [-1.5, 0.0, 4.25]]);
        let out = point_cloud_to_xyz(&pc);
        assert_eq!(out.len(), 2);
        assert!((out[0][0] - 1.0).abs() < 1e-6);
        assert!((out[0][1] - 2.0).abs() < 1e-6);
        assert!((out[0][2] - 3.0).abs() < 1e-6);
        assert!((out[1][0] - (-1.5)).abs() < 1e-6);
        assert!((out[1][2] - 4.25).abs() < 1e-6);
    }

    #[test]
    fn empty_cloud_returns_empty() {
        let pc = PointCloud2::default();
        let out = point_cloud_to_xyz(&pc);
        assert!(out.is_empty());
    }

    #[test]
    fn nonfinite_points_dropped() {
        let pc = pc_with_xyz_f32(&[[1.0, 2.0, 3.0], [f32::NAN, 0.0, 0.0], [4.0, 5.0, 6.0]]);
        let out = point_cloud_to_xyz(&pc);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn odometry_round_trip() {
        let mut odom = Odometry::default();
        odom.pose.pose.position.x = 1.0;
        odom.pose.pose.position.y = 2.0;
        odom.pose.pose.position.z = 3.0;
        odom.pose.pose.orientation.x = 0.0;
        odom.pose.pose.orientation.y = 0.0;
        odom.pose.pose.orientation.z = 0.0;
        odom.pose.pose.orientation.w = 1.0;
        let pose = odometry_to_isometry(&odom);
        assert!((pose.translation.vector.x - 1.0).abs() < 1e-9);
        assert!((pose.translation.vector.y - 2.0).abs() < 1e-9);
        assert!((pose.translation.vector.z - 3.0).abs() < 1e-9);
        assert!(pose.rotation.angle().abs() < 1e-9);
    }
}
