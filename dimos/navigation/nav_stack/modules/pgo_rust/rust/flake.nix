{
  description = "DimOS PGO native module — Rust port (iSAM2 via cxx FFI, KISS-style ICP)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    gtsam-extended = {
      url = "github:jeff-hykin/gtsam-extended";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs = { self, nixpkgs, flake-utils, gtsam-extended, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Mirror pgo_cpp/cpp/flake.nix: same gtsam commit so iSAM2 numerics
        # match exactly across the C++ and Rust modules.
        gtsam-base = gtsam-extended.packages.${system}.gtsam-cpp;
        gtsam = gtsam-base.overrideAttrs (_old: {
          src = pkgs.fetchFromGitHub {
            owner = "borglab";
            repo = "gtsam";
            rev = "1a9792a7ede244850a413739557635b606f295c0";
            sha256 = "sha256-zxm5TGVPW1vipFVpw01zcvKRw4mkh+5ZBCR1n6G466o=";
          };
          env.NIX_CFLAGS_COMPILE = "-Wno-error=array-bounds";
        });

        rustToolchain = pkgs.rust-bin or null;

        commonInputs = [
          gtsam
          pkgs.eigen
          pkgs.boost
          pkgs.glib
          pkgs.onetbb
        ];
      in {
        # `nix build .#default` produces the compiled binary alongside the
        # C++ pgo's output path convention. `cargo build` under the dev
        # shell also works against the same headers/libs.
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "dimos-pgo-rust";
          version = "0.1.0";
          src = ./.;
          cargoLock = {
            lockFile = ./Cargo.lock;
            allowBuiltinFetchGit = true;
          };

          nativeBuildInputs = [ pkgs.pkg-config pkgs.cmake ];
          buildInputs = commonInputs;

          # Forward gtsam's location into build.rs via env vars so the cxx
          # shim picks up the right include / lib paths.
          GTSAM_INCLUDE_DIR = "${gtsam}/include";
          GTSAM_LIB_DIR = "${gtsam}/lib";
          EIGEN_INCLUDE_DIR = "${pkgs.eigen}/include/eigen3";
          BOOST_INCLUDE_DIR = "${pkgs.boost.dev}/include";
          TBB_LIB_DIR = "${pkgs.onetbb}/lib";

          doCheck = false;  # unit tests run separately via `cargo test`
        };

        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [
            pkgs.pkg-config
            pkgs.cmake
            pkgs.cargo
            pkgs.rustc
            pkgs.rustfmt
            pkgs.clippy
          ];
          buildInputs = commonInputs;

          GTSAM_INCLUDE_DIR = "${gtsam}/include";
          GTSAM_LIB_DIR = "${gtsam}/lib";
          EIGEN_INCLUDE_DIR = "${pkgs.eigen}/include/eigen3";
          BOOST_INCLUDE_DIR = "${pkgs.boost.dev}/include";
          TBB_LIB_DIR = "${pkgs.onetbb}/lib";

          shellHook = ''
            echo "dimos-pgo-rust dev shell"
            echo "  GTSAM=$GTSAM_INCLUDE_DIR"
            echo "  EIGEN=$EIGEN_INCLUDE_DIR"
          '';
        };
      });
}
