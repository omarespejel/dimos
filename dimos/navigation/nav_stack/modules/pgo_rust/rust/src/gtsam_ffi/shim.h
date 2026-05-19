#pragma once

#include "rust/cxx.h"

#include <cstdint>
#include <memory>

namespace gtsam_shim {

// Pose representation at the FFI boundary: 7 doubles (tx, ty, tz, qx, qy, qz, qw).
// We pass individual fields rather than a shared struct to keep shim.h
// independent of the cxx::bridge-generated header (the generated header
// includes this one first, which would make a shared-struct param undefined).

class FactorGraph {
public:
    FactorGraph();
    ~FactorGraph();

    void add_prior(uint64_t key,
                   double tx, double ty, double tz,
                   double qx, double qy, double qz, double qw,
                   double translation_sigma, double rotation_sigma);

    void add_between(uint64_t key_from, uint64_t key_to,
                     double tx, double ty, double tz,
                     double qx, double qy, double qz, double qw,
                     double translation_sigma, double rotation_sigma);

    size_t size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class Solver;
};

class Values {
public:
    Values();
    ~Values();

    void insert(uint64_t key,
                double tx, double ty, double tz,
                double qx, double qy, double qz, double qw);

    void clear();
    size_t size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class Solver;
};

class Solver {
public:
    explicit Solver(double relinearize_threshold);
    ~Solver();

    void update(FactorGraph& graph, Values& initial);

    // Reads `count` keys back; returns parallel vectors for key and per-key 7
    // pose components (translation + quaternion).  We avoid sharing a struct
    // across the FFI boundary; the bridge zips these into KeyedPose Rust-side.
    void estimate_all(rust::Vec<uint64_t>& keys, rust::Vec<double>& pose_components) const;

    // Best estimate for `key`. Writes the 7 components into `pose_components`
    // (cleared first). Returns true if `key` was present.
    bool estimate_one(uint64_t key, rust::Vec<double>& pose_components) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<FactorGraph> new_factor_graph();
std::unique_ptr<Values> new_values();
std::unique_ptr<Solver> new_solver(double relinearize_threshold);

}  // namespace gtsam_shim
