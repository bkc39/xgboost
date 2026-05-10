{
  description = "xgboost - Racket XGBoost bindings (scaffolding)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      version = "0.1.0";
    in
    {
      packages = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          cpp = pkgs.stdenv.mkDerivation {
            pname = "xgboost-cpp";
            inherit version;
            src = ./cpp;

            nativeBuildInputs = [
              pkgs.cmake
              pkgs.ninja
            ];

            buildInputs = [
              pkgs.xgboost
              pkgs.gtest
            ];

            cmakeFlags = [
              "-DBUILD_TESTING=ON"
              "-DCMAKE_CXX_STANDARD=26"
            ];

            doCheck = true;
            checkPhase = ''
              runHook preCheck
              ctest --output-on-failure
              runHook postCheck
            '';
          };

          racket = pkgs.stdenv.mkDerivation {
            pname = "xgboost";
            inherit version;
            src = ./.;

            nativeBuildInputs = [ pkgs.racket pkgs.makeWrapper ];
            buildInputs = [ cpp ];

            buildPhase = ''
              runHook preBuild

              export PLTUSERHOME=$TMPDIR/racket-home
              export XGBOOST_NATIVE_LIB_PATH=${cpp}
              mkdir -p $PLTUSERHOME

              # Pre-populate native-libs/ so define-runtime-path works during testing
              mkdir -p ./xgboost/native-libs
              cp ${cpp}/lib/libxgbcompat.* ./xgboost/native-libs/

              raco pkg install --batch --deps fail --no-setup --copy --scope user \
                --name xgboost ./xgboost

              raco setup --no-docs --pkgs xgboost

              runHook postBuild
            '';

            doCheck = true;
            checkPhase = ''
              runHook preCheck
              raco test ./xgboost/
              raco test \
                examples/11-global-apis.rkt \
                examples/12-dmatrix-constructors.rkt \
                examples/13-high-level-root-api.rkt \
                examples/14-dmatrix-metadata.rkt \
                examples/15-dmatrix-slicing-binary.rkt \
                examples/16-quantile-cuts.rkt \
                examples/17-booster-lifecycle-config.rkt \
                examples/18-booster-attrs.rkt \
                examples/19-booster-dumps-feature-scores.rkt \
                examples/20-inplace-predict-dense.rkt \
                examples/21-inplace-predict-csr.rkt \
                examples/22-inplace-predict-columnar.rkt \
                examples/23-custom-objective.rkt
              runHook postCheck
            '';

            installPhase = ''
              runHook preInstall

              mkdir -p $out/share $out/bin
              cp -r $PLTUSERHOME $out/share/racket-home

              makeWrapper ${pkgs.racket}/bin/racket $out/bin/xgboost \
                --set PLTUSERHOME $out/share/racket-home \
                --add-flags "-l xgboost"

              runHook postInstall
            '';
          };

          copy-native-libs = pkgs.writeShellApplication {
            name = "copy-native-libs";
            runtimeInputs = [ pkgs.patchelf ];
            text = ''
              DEST="$(pwd)/xgboost/native-libs"
              mkdir -p "$DEST"
              cp -v --no-preserve=mode ${cpp}/lib/libxgbcompat.* "$DEST/"
              cp -v --no-preserve=mode ${pkgs.xgboost}/lib/libxgboost.so "$DEST/"
              patchelf --set-rpath '$ORIGIN' "$DEST/libxgbcompat.so"
              echo "Native libraries copied to $DEST"
              ls -la "$DEST"
            '';
          };

          hasAarch64Cross = system == "x86_64-linux";
          pkgsCross-aarch64 = import nixpkgs {
            localSystem = system;
            crossSystem = nixpkgs.lib.systems.examples.aarch64-multiplatform;
          };
          cpp-aarch64 = if hasAarch64Cross then
            pkgsCross-aarch64.stdenv.mkDerivation {
              pname = "xgboost-cpp-aarch64";
              inherit version;
              src = ./cpp;
              nativeBuildInputs = [
                pkgsCross-aarch64.buildPackages.cmake
                pkgsCross-aarch64.buildPackages.ninja
              ];
              buildInputs = [ pkgsCross-aarch64.xgboost ];
              cmakeFlags = [
                "-DBUILD_TESTING=OFF"
                "-DCMAKE_CXX_STANDARD=26"
              ];
              doCheck = false;
            }
          else throw "cpp-aarch64 requires an x86_64-linux host";

          hasCuda = system == "x86_64-linux";
          pkgs-cuda = import nixpkgs {
            inherit system;
            config = { cudaSupport = true; allowUnfree = true; };
          };
          cpp-cuda = pkgs-cuda.stdenv.mkDerivation {
            pname = "xgboost-cpp-cuda";
            inherit version;
            src = ./cpp;
            nativeBuildInputs = [ pkgs-cuda.cmake pkgs-cuda.ninja ];
            buildInputs = [ pkgs-cuda.xgboost pkgs-cuda.gtest ];
            cmakeFlags = [ "-DBUILD_TESTING=ON" "-DCMAKE_CXX_STANDARD=26" ];
            doCheck = true;
            checkPhase = ''
              runHook preCheck
              ctest --output-on-failure
              runHook postCheck
            '';
          };
          copy-native-libs-cuda = pkgs-cuda.writeShellApplication {
            name = "copy-native-libs-cuda";
            runtimeInputs = [ pkgs-cuda.patchelf ];
            text = ''
              DEST="$(pwd)/xgboost/native-libs"
              mkdir -p "$DEST"
              cp -v --no-preserve=mode ${cpp-cuda}/lib/libxgbcompat.* "$DEST/"
              cp -v --no-preserve=mode ${pkgs-cuda.xgboost}/lib/libxgboost.so "$DEST/"
              patchelf --set-rpath '$ORIGIN' "$DEST/libxgbcompat.so"
              echo "CUDA native libraries copied to $DEST"
              ls -la "$DEST"
            '';
          };
        in
        {
          default = racket;
          inherit cpp racket copy-native-libs;
        } // nixpkgs.lib.optionalAttrs hasAarch64Cross {
          inherit cpp-aarch64;
        } // nixpkgs.lib.optionalAttrs hasCuda {
          inherit cpp-cuda copy-native-libs-cuda;
        });

      apps = forAllSystems (system: {
        copy-native-libs = {
          type = "app";
          program = "${self.packages.${system}.copy-native-libs}/bin/copy-native-libs";
        };
      } // nixpkgs.lib.optionalAttrs (system == "x86_64-linux") {
        copy-native-libs-cuda = {
          type = "app";
          program = "${self.packages.${system}.copy-native-libs-cuda}/bin/copy-native-libs-cuda";
        };
      });

      checks = forAllSystems (system: {
        inherit (self.packages.${system}) cpp racket;
      });

      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          cpp = self.packages.${system}.cpp;
          hasCuda = system == "x86_64-linux";
          pkgs-cuda = import nixpkgs {
            inherit system;
            config = { cudaSupport = true; allowUnfree = true; };
          };
          cpp-cuda = self.packages.${system}.cpp-cuda;
        in
        {
          default = pkgs.mkShell {
            buildInputs = [
              pkgs.cmake
              pkgs.gtest
              pkgs.ninja
              pkgs.racket
              pkgs.xgboost
              pkgs.stdenv.cc
            ];

            shellHook = ''
              export XGBOOST_NATIVE_LIB_PATH="${cpp}"
              export PLTUSERHOME="$PWD/.racket-user"
              _rkt_ver=$(racket --version 2>&1 | grep -oP 'v\d+\.\d+' | tr -d 'v' | tr '.' '-')
              deps_stamp="$PLTUSERHOME/.deps-installed-''${_rkt_ver}"
              if [ ! -f "$deps_stamp" ]; then
                echo "Installing Racket package (link mode, Racket ''${_rkt_ver})..."
                mkdir -p "$PLTUSERHOME"
                mkdir -p ./xgboost/native-libs
                cp ${cpp}/lib/libxgbcompat.* ./xgboost/native-libs/ 2>/dev/null || true
                raco pkg install --batch --auto --no-setup --link --scope user --skip-installed \
                  --name xgboost "$PWD/xgboost"
                raco setup --no-docs --pkgs xgboost
                touch "$deps_stamp"
                echo "Done. Run: raco test xgboost/"
              fi
            '';
          };
        } // nixpkgs.lib.optionalAttrs hasCuda {
          cuda = pkgs-cuda.mkShell {
            buildInputs = [
              pkgs-cuda.cmake
              pkgs-cuda.gtest
              pkgs-cuda.ninja
              pkgs-cuda.racket
              pkgs-cuda.xgboost
              pkgs-cuda.stdenv.cc
            ];

            shellHook = ''
              export XGBOOST_NATIVE_LIB_PATH="${cpp-cuda}"
              export PLTUSERHOME="$PWD/.racket-user-cuda"
              # Expose host NVIDIA driver libs (libcuda.so etc.) without
              # overriding Nix-packaged glibc. We symlink only the NVIDIA/CUDA
              # files into a temp dir so the full system lib dir isn't on the path.
              if [ -d /usr/lib/x86_64-linux-gnu ]; then
                _nvidia_stub=$(mktemp -d)
                ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so* "$_nvidia_stub/" 2>/dev/null || true
                ln -sf /usr/lib/x86_64-linux-gnu/libnvidia*.so* "$_nvidia_stub/" 2>/dev/null || true
                export LD_LIBRARY_PATH="$_nvidia_stub''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
                unset _nvidia_stub
              fi
              _rkt_ver=$(racket --version 2>&1 | grep -oP 'v\d+\.\d+' | tr -d 'v' | tr '.' '-')
              deps_stamp="$PLTUSERHOME/.deps-installed-''${_rkt_ver}"
              if [ ! -f "$deps_stamp" ]; then
                echo "Installing Racket package (link mode, CUDA build, Racket ''${_rkt_ver})..."
                mkdir -p "$PLTUSERHOME"
                mkdir -p ./xgboost/native-libs
                cp ${cpp-cuda}/lib/libxgbcompat.* ./xgboost/native-libs/ 2>/dev/null || true
                raco pkg install --batch --auto --no-setup --link --scope user --skip-installed \
                  --name xgboost "$PWD/xgboost"
                raco setup --no-docs --pkgs xgboost
                touch "$deps_stamp"
                echo "Done. Run: racket examples/24-cuda-regression.rkt"
              fi
            '';
          };
        });
    };
}
