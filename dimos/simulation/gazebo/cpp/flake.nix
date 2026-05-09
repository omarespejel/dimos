{
  description = "Gazebo Harmonic (gz-sim 8) – standalone, no ROS";

  inputs = {
    # Use 24.05 as base: has freeimage + protobuf v24 (string-returning API)
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    # nixos-23.05 still ships ogre1_10. gz-rendering 8 was developed against
    # OGRE 1.10/1.11 and uses APIs (Camera::yaw/pitch/roll/setDirection,
    # SceneManager::_suppressRenderStateChanges, Light::setDirection, etc.)
    # that were removed in OGRE 13+. We pin only ogre1_10 from this side
    # channel; everything else stays on 24.05.
    nixpkgs-old.url = "github:NixOS/nixpkgs/nixos-23.05";
    flake-utils.url = "github:numtide/flake-utils";
    dimos-lcm = {
      url = "github:dimensionalOS/dimos-lcm/main";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, nixpkgs-old, flake-utils, dimos-lcm, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.permittedInsecurePackages = [
            "freeimage-unstable-2021-11-01"
          ];
        };

        # ogre1_10 — last release of the legacy 1.x API line gz-rendering 8
        # still expects. Pulled from 23.05 because 24.05 only ships OGRE 13/14.
        pkgs-old = import nixpkgs-old {
          inherit system;
          config.permittedInsecurePackages = [
            "freeimage-unstable-2021-11-01"
            "freeimage-3.18.0-unstable-2024-04-18"
          ];
        };
        ogre = pkgs-old.ogre1_10;

        # ---------- helper: every gz lib is a cmake project --------
        mkGzPkg = { pname, version, src, buildInputs ? [], cmakeFlags ? [],
                     nativeBuildInputs ? [], preConfigure ? "", postInstall ? "",
                     preFixup ? "", patches ? [], ... }:
          pkgs.stdenv.mkDerivation {
            inherit pname version src patches preConfigure postInstall;
            nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config ] ++ nativeBuildInputs;
            buildInputs = buildInputs;
            cmakeFlags = [
              "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
              "-DBUILD_TESTING=OFF"
            ] ++ cmakeFlags;
            # Strip /build/ refs that some plugins leave in their RPATH
            preFixup = ''
              find $out -type f \( -name '*.so' -o -name '*.so.*' \) -print0 \
                | while IFS= read -r -d "" f; do
                    rp=$(patchelf --print-rpath "$f" 2>/dev/null || true)
                    if [ -n "$rp" ] && echo "$rp" | grep -q '/build/'; then
                      new_rp=$(echo "$rp" | tr ':' '\n' | grep -v '/build/' | paste -sd: -)
                      patchelf --set-rpath "$new_rp" "$f"
                    fi
                  done
            '' + preFixup;
          };

        # Transitive deps that gz cmake configs require at configure time
        transitiveDeps = [
          pkgs.protobuf pkgs.python3 pkgs.tinyxml-2 pkgs.zeromq
          pkgs.cppzmq pkgs.libuuid pkgs.zlib pkgs.eigen pkgs.gdal
          pkgs.freeimage pkgs.curl pkgs.jsoncpp pkgs.libzip
        ];

        # ==================== sources ==============================

        gz-cmake-src = pkgs.fetchFromGitHub {
          owner = "gazebosim"; repo = "gz-cmake";
          rev = "gz-cmake3_3.5.5";
          hash = "sha256-GeVmrcIYzAma7NdeEQUs5VHyCMagj2HYghT0crY4zIc=";
        };

        gz-utils-src = pkgs.fetchFromGitHub {
          owner = "gazebosim"; repo = "gz-utils";
          rev = "gz-utils2_2.2.0";
          hash = "sha256-dNoDOZtk/zseHuOM5mOPHkXKU7wqxxKrFnh7e09bjRA=";
        };

        gz-math-src = pkgs.fetchFromGitHub {
          owner = "gazebosim"; repo = "gz-math";
          rev = "gz-math7_7.5.1";
          hash = "sha256-RxCZiU0h0skVPBSn+PMtkdwEabmTKl+0PYDpl9SQoq8=";
        };

        sdformat-src = pkgs.fetchFromGitHub {
          owner = "gazebosim"; repo = "sdformat";
          rev = "sdformat14_14.7.0";
          hash = "sha256-p2e01bCoMpDhia1yOFa5wIP2ritBiWNT5jYbp/bg1+g=";
        };

        gz-common-src = pkgs.fetchFromGitHub {
          owner = "gazebosim"; repo = "gz-common";
          rev = "gz-common5_5.7.0";
          hash = "sha256-RBu49rxjzo4mc7ma4WpabUxUT7cvabJRinR98it10r4=";
        };

        gz-plugin-src = pkgs.fetchFromGitHub {
          owner = "gazebosim"; repo = "gz-plugin";
          rev = "gz-plugin2_2.0.3";
          hash = "sha256-9t6vcnBbfRWu6ptmqYAhmWKDoKAaK631JD9u1C0G0mY=";
        };

        gz-msgs-src = pkgs.fetchFromGitHub {
          owner = "gazebosim"; repo = "gz-msgs";
          rev = "gz-msgs10_10.3.2";
          hash = "sha256-gxhRqLzBCaDmK67T5RryDpxbDR3WLgV9DFs7w6ieMxQ=";
        };

        gz-transport-src = pkgs.fetchFromGitHub {
          owner = "gazebosim"; repo = "gz-transport";
          rev = "gz-transport13_13.4.0";
          hash = "sha256-2Akd3vKr07IdgoJppvUV1nZlHE4RdQfI2R18ihHTDHk=";
        };

        gz-fuel-tools-src = pkgs.fetchFromGitHub {
          owner = "gazebosim"; repo = "gz-fuel-tools";
          rev = "gz-fuel-tools9_9.1.1";
          hash = "sha256-XQoBcCtzwzzPypS1kIeTCIbjtxrzaW3JvZLCYbwXAOk=";
        };

        gz-rendering-src = pkgs.fetchFromGitHub {
          owner = "gazebosim"; repo = "gz-rendering";
          rev = "gz-rendering8_8.2.3";
          hash = "sha256-5zqEHt7+69Qbp6I+JcY7h2CYzLKnvl1HHcnM3BYpqr4=";
        };

        gz-gui-src = pkgs.fetchFromGitHub {
          owner = "gazebosim"; repo = "gz-gui";
          rev = "gz-gui8_8.4.0";
          hash = "sha256-gf9XZzAX2g6r9ThIA0v2H2X/+uu9VnwvyvrdL5ZazM0=";
        };

        gz-physics-src = pkgs.fetchFromGitHub {
          owner = "gazebosim"; repo = "gz-physics";
          rev = "gz-physics7_7.4.0";
          hash = "sha256-14/P/xoZSqqqf9krgqDKVcO/rTZOEhLni8ZUR3W9ey4=";
        };

        gz-sensors-src = pkgs.fetchFromGitHub {
          owner = "gazebosim"; repo = "gz-sensors";
          rev = "gz-sensors8_8.2.0";
          hash = "sha256-j/8kS+Bvaim2gtsZcp+/u8CAE+N24/5qZhciFR0Q8+M=";
        };

        gz-sim-src = pkgs.fetchFromGitHub {
          owner = "gazebosim"; repo = "gz-sim";
          rev = "gz-sim8_8.7.0";
          hash = "sha256-Kalt+UwFiL1D+A5pkM/aZyEmBenqPo9U4jlAmqLze3c=";
        };

        # ==================== packages (build order) ===============

        gz-cmake = mkGzPkg {
          pname = "gz-cmake";
          version = "3.5.5";
          src = gz-cmake-src;
        };

        gz-utils = mkGzPkg {
          pname = "gz-utils";
          version = "2.2.0";
          src = gz-utils-src;
          buildInputs = [ gz-cmake ];
          cmakeFlags = [ "-DCMAKE_PREFIX_PATH=${gz-cmake}" ];
        };

        gz-math = mkGzPkg {
          pname = "gz-math";
          version = "7.5.1";
          src = gz-math-src;
          buildInputs = [ gz-cmake gz-utils pkgs.eigen ];
          cmakeFlags = [
            "-DCMAKE_PREFIX_PATH=${gz-cmake};${gz-utils}"
          ];
        };

        sdformat = mkGzPkg {
          pname = "sdformat";
          version = "14.7.0";
          src = sdformat-src;
          nativeBuildInputs = [ pkgs.python3 ];
          buildInputs = [ gz-cmake gz-utils gz-math pkgs.tinyxml-2 pkgs.urdfdom ];
          cmakeFlags = [
            "-DCMAKE_PREFIX_PATH=${gz-cmake};${gz-utils};${gz-math}"
          ];
        };

        gz-common = mkGzPkg {
          pname = "gz-common";
          version = "5.7.0";
          src = gz-common-src;
          buildInputs = [
            gz-cmake gz-utils gz-math
            pkgs.tinyxml-2 pkgs.freeimage pkgs.ffmpeg pkgs.gts
            pkgs.libuuid pkgs.libzip pkgs.curl pkgs.jsoncpp
            pkgs.gdal pkgs.assimp
          ];
          cmakeFlags = [
            "-DCMAKE_PREFIX_PATH=${gz-cmake};${gz-utils};${gz-math}"
          ];
        };

        gz-plugin = mkGzPkg {
          pname = "gz-plugin";
          version = "2.0.3";
          src = gz-plugin-src;
          buildInputs = [ gz-cmake gz-utils ];
          cmakeFlags = [
            "-DCMAKE_PREFIX_PATH=${gz-cmake};${gz-utils}"
          ];
        };

        gz-msgs = mkGzPkg {
          pname = "gz-msgs";
          version = "10.3.2";
          src = gz-msgs-src;
          nativeBuildInputs = [ pkgs.protobuf pkgs.python3 ];
          buildInputs = [
            gz-cmake gz-utils gz-math
            pkgs.protobuf pkgs.tinyxml-2
          ];
          cmakeFlags = [
            "-DCMAKE_PREFIX_PATH=${gz-cmake};${gz-utils};${gz-math}"
          ];
        };

        gz-transport = mkGzPkg {
          pname = "gz-transport";
          version = "13.4.0";
          src = gz-transport-src;
          nativeBuildInputs = [ pkgs.python3 ];
          buildInputs = [
            gz-cmake gz-utils gz-math gz-msgs
            pkgs.protobuf pkgs.zeromq pkgs.cppzmq
            pkgs.sqlite pkgs.libuuid pkgs.zlib pkgs.tinyxml-2
          ];
          cmakeFlags = [
            "-DCMAKE_PREFIX_PATH=${gz-cmake};${gz-utils};${gz-math};${gz-msgs}"
          ];
        };

        gz-fuel-tools = mkGzPkg {
          pname = "gz-fuel-tools";
          version = "9.1.1";
          src = gz-fuel-tools-src;
          nativeBuildInputs = [ pkgs.python3 ];
          buildInputs = [
            gz-cmake gz-utils gz-math gz-common gz-msgs gz-transport
            pkgs.curl pkgs.jsoncpp pkgs.libzip pkgs.tinyxml-2
            pkgs.protobuf pkgs.zeromq pkgs.cppzmq pkgs.libyaml
            pkgs.libuuid pkgs.gdal
          ];
          cmakeFlags = [
            "-DCMAKE_PREFIX_PATH=${gz-cmake};${gz-utils};${gz-math};${gz-common};${gz-msgs};${gz-transport}"
          ];
        };

        gz-rendering = mkGzPkg {
          pname = "gz-rendering";
          version = "8.2.3";
          src = gz-rendering-src;
          buildInputs = [
            gz-cmake gz-utils gz-math gz-common gz-plugin
            ogre pkgs.freeimage pkgs.xorg.libX11
            pkgs.libglvnd pkgs.mesa pkgs.eigen
            pkgs.libuuid pkgs.gdal
            pkgs.boost      # ogre1_10's threading headers include boost/thread/tss.hpp
            pkgs.libGL pkgs.libGLU  # OGRE's RenderSystems/GL needs <GL/glu.h>
          ];
          cmakeFlags = [
            "-DCMAKE_PREFIX_PATH=${gz-cmake};${gz-utils};${gz-math};${gz-common};${gz-plugin}"
          ];
          # Two source patches:
          #
          # (1) Modern libglvnd's <GL/glxext.h> uses GLintptr/GLsizeiptr
          #     without forward-declaring them — those typedefs live in
          #     <GL/glext.h>. Inject <GL/gl.h>+<GL/glext.h> before every
          #     <GL/glx.h>.
          #
          # (2) gz-rendering's `OgreRenderEngine::CreateRenderWindow()`
          #     (the no-arg dummy variant) hands OGRE the X11 ID of a 1x1
          #     window it created on its own X display connection. OGRE
          #     opens a SEPARATE display connection and validates the
          #     handle via XGetWindowAttributes, which can fail with
          #     "Invalid parentWindowHandle (wrong server or screen)" on
          #     some X servers (any setup where OGRE's display has a
          #     different DefaultRootWindow value than gz-rendering's).
          #     OGRE explicitly skips that validation when parentWindow
          #     equals DefaultRootWindow — so pass that instead.
          preConfigure = ''
            for f in $(grep -rl '<GL/glx.h>' ogre/ || true); do
              sed -i 's|<GL/glx.h>|<GL/gl.h>\n# include <GL/glext.h>\n# include <GL/glx.h>|' "$f"
            done

            # Replace the dummyWindowId arg with a runtime-computed
            # DefaultRootWindow so OGRE skips its parentWindowHandle check.
            sed -i 's|this->CreateRenderWindow(std::to_string(this->dummyWindowId), 1, 1,|this->CreateRenderWindow(std::to_string(static_cast<unsigned long>(DefaultRootWindow(static_cast<Display*>(this->dummyDisplay)))), 1, 1,|' \
              ogre/src/OgreRenderEngine.cc
            grep -q "DefaultRootWindow(static_cast<Display\*>" ogre/src/OgreRenderEngine.cc \
              || { echo "[gz-rendering patch] dummy window patch did not apply" >&2; exit 1; }
          '';
        };

        gz-gui = mkGzPkg {
          pname = "gz-gui";
          version = "8.4.0";
          src = gz-gui-src;
          nativeBuildInputs = [ pkgs.qt5.wrapQtAppsHook pkgs.python3 ];
          buildInputs = [
            gz-cmake gz-utils gz-math gz-common gz-plugin
            gz-msgs gz-transport gz-rendering
            pkgs.qt5.qtbase pkgs.qt5.qtquickcontrols2 pkgs.qt5.qtdeclarative
          ] ++ transitiveDeps;
          cmakeFlags = [
            "-DCMAKE_PREFIX_PATH=${gz-cmake};${gz-utils};${gz-math};${gz-common};${gz-plugin};${gz-msgs};${gz-transport};${gz-rendering}"
          ];
        };

        gz-physics = mkGzPkg {
          pname = "gz-physics";
          version = "7.4.0";
          src = gz-physics-src;
          nativeBuildInputs = [ pkgs.python3 ];
          buildInputs = [
            gz-cmake gz-utils gz-math gz-common gz-plugin
            sdformat
            pkgs.bullet pkgs.eigen pkgs.libuuid pkgs.gdal
          ];
          cmakeFlags = [
            "-DCMAKE_PREFIX_PATH=${gz-cmake};${gz-utils};${gz-math};${gz-common};${gz-plugin};${sdformat}"
          ];
        };

        gz-sensors = mkGzPkg {
          pname = "gz-sensors";
          version = "8.2.0";
          src = gz-sensors-src;
          nativeBuildInputs = [ pkgs.python3 ];
          buildInputs = [
            gz-cmake gz-utils gz-math gz-common gz-plugin
            gz-msgs gz-transport gz-rendering sdformat
          ] ++ transitiveDeps;
          cmakeFlags = [
            "-DCMAKE_PREFIX_PATH=${gz-cmake};${gz-utils};${gz-math};${gz-common};${gz-plugin};${gz-msgs};${gz-transport};${gz-rendering};${sdformat}"
          ];
        };

        gz-sim = mkGzPkg {
          pname = "gz-sim";
          version = "8.7.0";
          src = gz-sim-src;
          nativeBuildInputs = [ pkgs.qt5.wrapQtAppsHook pkgs.protobuf pkgs.python3 ];
          buildInputs = [
            gz-cmake gz-utils gz-math gz-common gz-plugin
            gz-msgs gz-transport gz-rendering gz-gui
            gz-physics gz-sensors sdformat gz-fuel-tools
            pkgs.qt5.qtbase pkgs.qt5.qtquickcontrols2 pkgs.qt5.qtdeclarative
            pkgs.bullet
            pkgs.ffmpeg  # gz-common5-av re-finds SWSCALE/AV* when consumers load it
          ] ++ transitiveDeps;
          cmakeFlags = [
            "-DCMAKE_PREFIX_PATH=${gz-cmake};${gz-utils};${gz-math};${gz-common};${gz-plugin};${gz-msgs};${gz-transport};${gz-rendering};${gz-gui};${gz-physics};${gz-sensors};${sdformat};${gz-fuel-tools}"
          ];
        };

        # ==================== gazebo_native bridge =================
        # Shared dimos NativeModule helpers + LCM headers live alongside
        # the hardware sensors. Pin a relative path so devShells work too.
        dimos-common = ../../../hardware/sensors/lidar/common;

        gazebo_native = pkgs.stdenv.mkDerivation {
          pname = "gazebo_native";
          version = "0.1.0";
          src = ./.;

          # Bridge is a CLI tool; Qt comes in via gz-sim's transitive deps.
          dontWrapQtApps = true;

          nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config pkgs.makeWrapper ];
          buildInputs = [
            gz-cmake gz-utils gz-math gz-common gz-plugin
            gz-msgs gz-transport gz-rendering gz-physics gz-sensors
            gz-fuel-tools gz-gui gz-sim
            sdformat
            pkgs.lcm pkgs.glib pkgs.protobuf
            pkgs.qt5.qtbase pkgs.qt5.qtquickcontrols2 pkgs.qt5.qtdeclarative
            pkgs.bullet pkgs.ffmpeg pkgs.assimp ogre
            pkgs.libyaml pkgs.urdfdom
          ] ++ transitiveDeps;

          cmakeFlags = [
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
            "-DFETCHCONTENT_SOURCE_DIR_DIMOS_LCM=${dimos-lcm}"
            "-DDIMOS_COMMON_DIR=${dimos-common}"
            "-DCMAKE_PREFIX_PATH=${gz-cmake};${gz-utils};${gz-math};${gz-common};${gz-plugin};${gz-msgs};${gz-transport};${gz-rendering};${gz-physics};${gz-sensors};${gz-fuel-tools};${gz-gui};${gz-sim};${sdformat}"
          ];

          # Tell gz-sim where its plugin .so's live, otherwise the embedded
          # Server can't load gz-sim-physics-system, sensors-system, etc.
          # Also point gz-rendering at the Ogre1 render engine plugin.
          postInstall = ''
            wrapProgram $out/bin/gazebo_native \
              --prefix GZ_SIM_SYSTEM_PLUGIN_PATH    : ${gz-sim}/lib/gz-sim-8/plugins \
              --prefix GZ_SIM_RESOURCE_PATH         : ${gz-sim}/share/gz/gz-sim8 \
              --prefix GZ_SIM_PHYSICS_ENGINE_PATH   : ${gz-physics}/lib \
              --prefix GZ_GUI_PLUGIN_PATH           : ${gz-gui}/lib/gz-gui-8/plugins \
              --prefix GZ_RENDERING_PLUGIN_PATH     : ${gz-rendering}/lib/gz-rendering-8/engine-plugins \
              --prefix GZ_RENDERING_RESOURCE_PATH   : ${gz-rendering}/share/gz/gz-rendering8 \
              --prefix OGRE_RESOURCE_PATH           : ${ogre}/lib/OGRE \
              --set    GZ_CONFIG_PATH                 ${gz-cmake}/share/gz
          '';
        };

      in {
        packages = {
          default = gazebo_native;
          inherit
            gz-cmake gz-utils gz-math sdformat
            gz-common gz-plugin gz-msgs gz-transport
            gz-fuel-tools gz-rendering gz-gui
            gz-physics gz-sensors gz-sim
            gazebo_native;
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [ gz-sim gazebo_native ];
        };
      });
}
