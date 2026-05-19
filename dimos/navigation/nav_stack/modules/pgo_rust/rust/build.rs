// Compiles the gtsam cxx shim. Header / library locations are sourced from
// env vars set by `flake.nix` (`nix develop` or `nix build`) — falling back to
// pkg-config and finally to /usr defaults for non-nix dev environments.

use std::env;
use std::path::PathBuf;

fn env_var(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.is_empty())
}

fn resolve_include(env_name: &str, pkg: &str, fallback: &str) -> PathBuf {
    if let Some(value) = env_var(env_name) {
        return PathBuf::from(value);
    }
    if let Ok(probe) = pkg_config::probe_library(pkg) {
        if let Some(path) = probe.include_paths.first() {
            return path.clone();
        }
    }
    PathBuf::from(fallback)
}

fn resolve_lib(env_name: &str, fallback: &str) -> PathBuf {
    env_var(env_name).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(fallback))
}

fn main() {
    println!("cargo:rerun-if-changed=src/gtsam_ffi/shim.h");
    println!("cargo:rerun-if-changed=src/gtsam_ffi/shim.cpp");
    println!("cargo:rerun-if-changed=src/gtsam_ffi/mod.rs");
    println!("cargo:rerun-if-env-changed=GTSAM_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=GTSAM_LIB_DIR");
    println!("cargo:rerun-if-env-changed=EIGEN_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=BOOST_INCLUDE_DIR");

    let gtsam_include = resolve_include("GTSAM_INCLUDE_DIR", "gtsam", "/usr/include");
    let gtsam_lib = resolve_lib("GTSAM_LIB_DIR", "/usr/lib");
    let eigen_include = resolve_include("EIGEN_INCLUDE_DIR", "eigen3", "/usr/include/eigen3");
    let boost_include = resolve_include("BOOST_INCLUDE_DIR", "boost", "/usr/include");

    cxx_build::bridge("src/gtsam_ffi/mod.rs")
        .file("src/gtsam_ffi/shim.cpp")
        .include(&gtsam_include)
        .include(&eigen_include)
        .include(&boost_include)
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-Wno-deprecated-declarations")
        .flag_if_supported("-Wno-unused-parameter")
        .compile("dimos-pgo-gtsam-shim");

    println!("cargo:rustc-link-search=native={}", gtsam_lib.display());
    println!("cargo:rustc-link-lib=dylib=gtsam");
    // Force libcephes-gtsam AND libm into our DT_NEEDED list. Rust code never
    // refs cephes symbols, so without --no-as-needed the linker drops it. That
    // leaves cephes only transitively reachable via libgtsam, in which case
    // the dynamic linker resolves cephes's IFUNC `sin` against libm BEFORE
    // cephes itself initializes, triggering "Relink" warnings + SIGSEGV at
    // process startup.  Pinning cephes-gtsam (and libm, which cephes itself
    // needs but the linker can't otherwise prove without --no-as-needed) into
    // DT_NEEDED mirrors how the C++ pgo binary's CMake build links them.
    println!(
        "cargo:rustc-link-arg=-Wl,--push-state,--no-as-needed,-lcephes-gtsam,-lm,--pop-state"
    );

    // GTSAM 4.3a1 is built against TBB for ISAM2's concurrent containers.
    let tbb_lib = resolve_lib("TBB_LIB_DIR", "/usr/lib");
    println!("cargo:rustc-link-search=native={}", tbb_lib.display());
    println!("cargo:rustc-link-lib=dylib=tbb");
}

// Tiny in-build pkg-config probe — avoids adding an external `pkg-config` crate
// dependency just for two `probe_library` calls.
mod pkg_config {
    use std::path::PathBuf;
    use std::process::Command;

    pub struct Library {
        pub include_paths: Vec<PathBuf>,
    }

    pub fn probe_library(name: &str) -> Result<Library, ()> {
        let output = Command::new("pkg-config").arg("--cflags-only-I").arg(name).output().map_err(|_| ())?;
        if !output.status.success() {
            return Err(());
        }
        let cflags = String::from_utf8(output.stdout).map_err(|_| ())?;
        let include_paths = cflags
            .split_whitespace()
            .filter_map(|token| token.strip_prefix("-I"))
            .map(PathBuf::from)
            .collect();
        Ok(Library { include_paths })
    }
}
