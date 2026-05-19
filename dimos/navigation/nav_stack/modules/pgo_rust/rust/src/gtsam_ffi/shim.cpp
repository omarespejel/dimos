#include "dimos-pgo/src/gtsam_ffi/shim.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <Eigen/Geometry>

namespace gtsam_shim {

namespace {

gtsam::Pose3 make_pose(double tx, double ty, double tz, double qx, double qy, double qz, double qw) {
    Eigen::Quaterniond q(qw, qx, qy, qz);
    q.normalize();
    return gtsam::Pose3(gtsam::Rot3(q), gtsam::Point3(tx, ty, tz));
}

void push_pose(rust::Vec<double>& dest, const gtsam::Pose3& pose) {
    Eigen::Quaterniond q(pose.rotation().matrix());
    q.normalize();
    dest.push_back(pose.translation().x());
    dest.push_back(pose.translation().y());
    dest.push_back(pose.translation().z());
    dest.push_back(q.x());
    dest.push_back(q.y());
    dest.push_back(q.z());
    dest.push_back(q.w());
}

gtsam::SharedNoiseModel make_pose_noise(double translation_sigma, double rotation_sigma) {
    // gtsam::Pose3::Logmap orders components as (rotation xyz, translation xyz).
    gtsam::Vector6 sigmas;
    sigmas << rotation_sigma, rotation_sigma, rotation_sigma,
              translation_sigma, translation_sigma, translation_sigma;
    return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
}

}  // namespace

struct FactorGraph::Impl {
    gtsam::NonlinearFactorGraph graph;
};

FactorGraph::FactorGraph() : impl_(std::make_unique<Impl>()) {}
FactorGraph::~FactorGraph() = default;

void FactorGraph::add_prior(uint64_t key,
                            double tx, double ty, double tz,
                            double qx, double qy, double qz, double qw,
                            double translation_sigma, double rotation_sigma) {
    impl_->graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        static_cast<gtsam::Key>(key),
        make_pose(tx, ty, tz, qx, qy, qz, qw),
        make_pose_noise(translation_sigma, rotation_sigma));
}

void FactorGraph::add_between(uint64_t key_from, uint64_t key_to,
                              double tx, double ty, double tz,
                              double qx, double qy, double qz, double qw,
                              double translation_sigma, double rotation_sigma) {
    impl_->graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        static_cast<gtsam::Key>(key_from),
        static_cast<gtsam::Key>(key_to),
        make_pose(tx, ty, tz, qx, qy, qz, qw),
        make_pose_noise(translation_sigma, rotation_sigma));
}

size_t FactorGraph::size() const {
    return impl_->graph.size();
}

struct Values::Impl {
    gtsam::Values values;
};

Values::Values() : impl_(std::make_unique<Impl>()) {}
Values::~Values() = default;

void Values::insert(uint64_t key,
                    double tx, double ty, double tz,
                    double qx, double qy, double qz, double qw) {
    impl_->values.insert(static_cast<gtsam::Key>(key), make_pose(tx, ty, tz, qx, qy, qz, qw));
}

void Values::clear() {
    impl_->values.clear();
}

size_t Values::size() const {
    return impl_->values.size();
}

struct Solver::Impl {
    gtsam::ISAM2 isam2;
    gtsam::Values current_estimate;

    explicit Impl(double relinearize_threshold) {
        gtsam::ISAM2Params params;
        params.relinearizeThreshold = relinearize_threshold;
        params.relinearizeSkip = 1;
        isam2 = gtsam::ISAM2(params);
    }
};

Solver::Solver(double relinearize_threshold)
    : impl_(std::make_unique<Impl>(relinearize_threshold)) {}
Solver::~Solver() = default;

void Solver::update(FactorGraph& graph, Values& initial) {
    impl_->isam2.update(graph.impl_->graph, initial.impl_->values);
    graph.impl_->graph.resize(0);
    initial.impl_->values.clear();
    impl_->current_estimate = impl_->isam2.calculateEstimate();
}

void Solver::estimate_all(rust::Vec<uint64_t>& keys, rust::Vec<double>& pose_components) const {
    keys.reserve(impl_->current_estimate.size());
    pose_components.reserve(impl_->current_estimate.size() * 7);
    for (const auto& key_value : impl_->current_estimate) {
        const auto& pose = key_value.value.cast<gtsam::Pose3>();
        keys.push_back(static_cast<uint64_t>(key_value.key));
        push_pose(pose_components, pose);
    }
}

bool Solver::estimate_one(uint64_t key, rust::Vec<double>& pose_components) const {
    pose_components.clear();
    const auto gkey = static_cast<gtsam::Key>(key);
    if (!impl_->current_estimate.exists(gkey)) {
        return false;
    }
    push_pose(pose_components, impl_->current_estimate.at<gtsam::Pose3>(gkey));
    return true;
}

std::unique_ptr<FactorGraph> new_factor_graph() { return std::make_unique<FactorGraph>(); }
std::unique_ptr<Values> new_values() { return std::make_unique<Values>(); }
std::unique_ptr<Solver> new_solver(double relinearize_threshold) {
    return std::make_unique<Solver>(relinearize_threshold);
}

}  // namespace gtsam_shim
