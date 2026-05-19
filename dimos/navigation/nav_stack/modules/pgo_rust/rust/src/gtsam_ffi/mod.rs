// Safe Rust wrapper around the gtsam cxx::bridge.
//
// Wire format: poses cross the FFI as 7 doubles (tx, ty, tz, qx, qy, qz, qw),
// passed as individual args (FactorGraph::add_*, Values::insert) or as
// rust::Vec<f64> (Solver::estimate_*).  Keeping shim.h struct-free dodges the
// cxx-generated-header include-order circularity.

use nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion};

#[cxx::bridge(namespace = "gtsam_shim")]
mod ffi {
    unsafe extern "C++" {
        include!("dimos-pgo/src/gtsam_ffi/shim.h");

        type FactorGraph;
        type Values;
        type Solver;

        fn new_factor_graph() -> UniquePtr<FactorGraph>;
        fn new_values() -> UniquePtr<Values>;
        fn new_solver(relinearize_threshold: f64) -> UniquePtr<Solver>;

        fn add_prior(
            self: Pin<&mut FactorGraph>,
            key: u64,
            tx: f64,
            ty: f64,
            tz: f64,
            qx: f64,
            qy: f64,
            qz: f64,
            qw: f64,
            translation_sigma: f64,
            rotation_sigma: f64,
        );
        fn add_between(
            self: Pin<&mut FactorGraph>,
            key_from: u64,
            key_to: u64,
            tx: f64,
            ty: f64,
            tz: f64,
            qx: f64,
            qy: f64,
            qz: f64,
            qw: f64,
            translation_sigma: f64,
            rotation_sigma: f64,
        );
        fn size(self: &FactorGraph) -> usize;

        fn insert(
            self: Pin<&mut Values>,
            key: u64,
            tx: f64,
            ty: f64,
            tz: f64,
            qx: f64,
            qy: f64,
            qz: f64,
            qw: f64,
        );
        fn clear(self: Pin<&mut Values>);
        fn size(self: &Values) -> usize;

        fn update(self: Pin<&mut Solver>, graph: Pin<&mut FactorGraph>, initial: Pin<&mut Values>);
        fn estimate_all(
            self: &Solver,
            keys: &mut Vec<u64>,
            pose_components: &mut Vec<f64>,
        );
        fn estimate_one(
            self: &Solver,
            key: u64,
            pose_components: &mut Vec<f64>,
        ) -> bool;
    }
}

fn isometry_to_components(pose: &Isometry3<f64>) -> [f64; 7] {
    let translation = pose.translation.vector;
    let rotation = pose.rotation.into_inner();
    [translation.x, translation.y, translation.z, rotation.i, rotation.j, rotation.k, rotation.w]
}

fn components_to_isometry(c: &[f64]) -> Isometry3<f64> {
    Isometry3::from_parts(
        Translation3::new(c[0], c[1], c[2]),
        UnitQuaternion::from_quaternion(Quaternion::new(c[6], c[3], c[4], c[5])),
    )
}

/// Safe owning wrapper over a `Solver` + scratch `FactorGraph` + scratch `Values`.
pub struct GtsamBackend {
    solver: cxx::UniquePtr<ffi::Solver>,
    graph: cxx::UniquePtr<ffi::FactorGraph>,
    initial: cxx::UniquePtr<ffi::Values>,
}

// gtsam::ISAM2 / NonlinearFactorGraph / Values have no thread-local state.
// We exercise the backend strictly from the dimos-module handler task, never
// shared between threads, so Send is sound. Not Sync — concurrent access to
// the solver is not supported by gtsam without external locking.
unsafe impl Send for GtsamBackend {}

impl GtsamBackend {
    pub fn new(relinearize_threshold: f64) -> Self {
        Self {
            solver: ffi::new_solver(relinearize_threshold),
            graph: ffi::new_factor_graph(),
            initial: ffi::new_values(),
        }
    }

    pub fn add_prior(
        &mut self,
        key: u64,
        pose: Isometry3<f64>,
        translation_sigma: f64,
        rotation_sigma: f64,
    ) {
        let c = isometry_to_components(&pose);
        self.graph.pin_mut().add_prior(
            key, c[0], c[1], c[2], c[3], c[4], c[5], c[6], translation_sigma, rotation_sigma,
        );
    }

    pub fn add_between(
        &mut self,
        key_from: u64,
        key_to: u64,
        relative: Isometry3<f64>,
        translation_sigma: f64,
        rotation_sigma: f64,
    ) {
        let c = isometry_to_components(&relative);
        self.graph.pin_mut().add_between(
            key_from,
            key_to,
            c[0],
            c[1],
            c[2],
            c[3],
            c[4],
            c[5],
            c[6],
            translation_sigma,
            rotation_sigma,
        );
    }

    pub fn insert_initial(&mut self, key: u64, pose: Isometry3<f64>) {
        let c = isometry_to_components(&pose);
        self.initial.pin_mut().insert(key, c[0], c[1], c[2], c[3], c[4], c[5], c[6]);
    }

    pub fn update(&mut self) {
        self.solver.pin_mut().update(self.graph.pin_mut(), self.initial.pin_mut());
    }

    pub fn estimate(&self, key: u64) -> Option<Isometry3<f64>> {
        let mut buffer: Vec<f64> = Vec::with_capacity(7);
        if self.solver.estimate_one(key, &mut buffer) {
            Some(components_to_isometry(&buffer))
        } else {
            None
        }
    }

    pub fn estimate_all(&self) -> Vec<(u64, Isometry3<f64>)> {
        let mut keys: Vec<u64> = Vec::new();
        let mut components: Vec<f64> = Vec::new();
        self.solver.estimate_all(&mut keys, &mut components);
        keys.into_iter()
            .enumerate()
            .map(|(index, key)| {
                let chunk = &components[index * 7..(index + 1) * 7];
                (key, components_to_isometry(chunk))
            })
            .collect()
    }
}
