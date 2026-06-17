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

          # polyfill-glibc rewrites ELF binaries built against a newer glibc
          # so they resolve only symbols available on a chosen older target.
          # Used by scripts/build-so.sh to make our Linux candidates portable
          # back to glibc 2.35 (Ubuntu 22.04 / pkg-build.racket-lang.org).
          # Not in nixpkgs; pinned to a known-good upstream commit.
          polyfill-glibc = pkgs.stdenv.mkDerivation {
            pname = "polyfill-glibc";
            version = "unstable-2025-dd59051";
            src = pkgs.fetchFromGitHub {
              owner = "corsix";
              repo = "polyfill-glibc";
              rev = "dd59051faaa10ee63c1b96f1b47bf9fcd3770ee2";
              hash = "sha256-Qkzy33dIGnv9BOmRwql+LpYaEukZZIADSux09Fz3h7E=";
            };
            nativeBuildInputs = [ pkgs.ninja ];
            dontConfigure = true;
            buildPhase = ''
              runHook preBuild
              ninja polyfill-glibc
              runHook postBuild
            '';
            installPhase = ''
              runHook preInstall
              install -Dm755 polyfill-glibc $out/bin/polyfill-glibc
              runHook postInstall
            '';
            meta = {
              description = "Patch ELF binaries to require an older glibc version";
              homepage = "https://github.com/corsix/polyfill-glibc";
              license = pkgs.lib.licenses.mit;
              platforms = [ "x86_64-linux" "aarch64-linux" ];
            };
          };

          cpp = pkgs.stdenv.mkDerivation {
            pname = "xgboost-cpp";
            inherit version;
            src = ./cpp;

            nativeBuildInputs = [
              pkgs.cmake
              pkgs.clang-tools
              pkgs.ninja
            ];

            buildInputs = [
              pkgs.xgboost
              pkgs.gtest
            ];

            cmakeFlags = [
              "-DBUILD_TESTING=ON"
              "-DCMAKE_CXX_STANDARD=20"
            ];

            doCheck = true;
            checkPhase = ''
              runHook preCheck
              ctest --output-on-failure
              runHook postCheck
            '';
          };

          cpp-format = pkgs.stdenv.mkDerivation {
            pname = "xgboost-cpp-format";
            inherit version;
            src = ./cpp;

            nativeBuildInputs = [
              pkgs.cmake
              pkgs.clang-tools
              pkgs.ninja
            ];

            buildInputs = [
              pkgs.xgboost
              pkgs.gtest
            ];

            cmakeFlags = [
              "-DBUILD_TESTING=ON"
              "-DCMAKE_CXX_STANDARD=20"
            ];

            buildPhase = ''
              runHook preBuild
              cmake --build . --target format-check
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall
              touch $out
              runHook postInstall
            '';
          };

          cpp-tidy = pkgs.stdenv.mkDerivation {
            pname = "xgboost-cpp-tidy";
            inherit version;
            src = ./cpp;

            nativeBuildInputs = [
              pkgs.cmake
              pkgs.clang-tools
              pkgs.ninja
            ];

            buildInputs = [
              pkgs.xgboost
              pkgs.gtest
            ];

            cmakeFlags = [
              "-DBUILD_TESTING=ON"
              "-DCMAKE_CXX_STANDARD=20"
            ];

            buildPhase = ''
              runHook preBuild
              cmake --build . --target tidy
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall
              touch $out
              runHook postInstall
            '';
          };

          cpp-line-count = pkgs.stdenv.mkDerivation {
            pname = "xgboost-cpp-line-count";
            inherit version;
            src = ./cpp;

            dontConfigure = true;
            dontBuild = true;

            installPhase = ''
              runHook preInstall
              failed=0
              while IFS= read -r file; do
                lines=$(wc -l < "$file")
                if [ "$lines" -gt 500 ]; then
                  echo "ERROR: $file has $lines lines; limit is 500" >&2
                  failed=1
                fi
              done < <(find . -type f \( -name '*.c' -o -name '*.h' -o -name '*.hpp' -o -name '*.cpp' \))
              if [ "$failed" -ne 0 ]; then
                exit 1
              fi
              touch $out
              runHook postInstall
            '';
          };

          # The `polars` runtime dependency (and its transitive Racket deps:
          # gregor-lib, threading-lib, tzinfo, memoize-lib, cldr-*) are not in
          # nixpkgs, and the build sandbox has no network to fetch them from the
          # package catalog. This fixed-output derivation installs them into a
          # user scope *with* network access (FODs are granted it); the result
          # is a populated PLTUSERHOME that the `racket` build copies in before
          # installing xgboost. No Rust build is needed — polars stages its
          # prebuilt `libcompat` candidate during its own pre-install.
          #
          # Bump `version` (or update `outputHash`) when the catalog versions of
          # polars or its deps change.
          polarsScope = pkgs.stdenvNoCC.mkDerivation {
            pname = "rkt-polars-scope";
            version = "unstable-2026-06-17";
            dontUnpack = true;
            nativeBuildInputs = [ pkgs.racket pkgs.cacert ];
            buildCommand = ''
              export PLTUSERHOME="$out"
              export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
              mkdir -p "$out"
              # `tzdata` ships the zoneinfo database as a Racket collection
              # (`tzdata/zoneinfo`), which gregor's `tzinfo` needs at load time;
              # the Nix sandbox has no system /usr/share/zoneinfo.
              #
              # `--no-setup` installs sources only (no machine-code .zo, no
              # native-lib staging), so the output is platform-independent and a
              # single outputHash works across Linux/macOS. The consuming
              # `racket` build runs `raco setup` to compile and to fire polars'
              # pre-install (which stages the right libcompat for the platform).
              raco pkg install --no-setup --batch --auto --scope user --no-docs \
                polars tzdata
            '';
            outputHashMode = "recursive";
            outputHashAlgo = "sha256";
            outputHash = "sha256-UYsJEXP3+Kwl4OxbkCqxg/nBU8Lrf9Q/EuLhrtFUvKI=";
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

              # Seed the user scope with polars + its deps (built in the
              # network-enabled polarsScope FOD), so xgboost's `--deps fail`
              # install is satisfied offline. A relocated raco addon tree
              # resolves fine, native lib included.
              cp -r ${polarsScope}/. $PLTUSERHOME/
              chmod -R u+w $PLTUSERHOME

              # Pre-populate native-libs/ so define-runtime-path works during testing
              mkdir -p ./xgboost/native-libs
              cp ${cpp}/lib/libxgbcompat.* ./xgboost/native-libs/

              raco pkg install --batch --deps fail --no-setup --copy --scope user \
                --name xgboost ./xgboost

              # Compile xgboost, polars, and tzdata (the FOD installed their
              # sources only). Setting up polars fires its pre-install, which
              # stages the platform's libcompat; tzdata must be set up so its
              # committed zoneinfo collection is resolvable (gregor/tzinfo need
              # it at load time, and the sandbox has no /usr/share/zoneinfo).
              # gregor/tzinfo/cldr-* compile on first use.
              raco setup --no-docs --pkgs xgboost polars tzdata

              runHook postBuild
            '';

            doCheck = true;
            checkPhase = ''
              runHook preCheck
              raco test ./xgboost/
              # Each xgboost/examples/NN-name.rkt is a literate scribble/lp2
              # program; its runner + RackUnit checks live in
              # xgboost/examples/test/NN-name.rkt.
              # The CUDA harnesses self-skip when no GPU is available.
              raco test xgboost/examples/test/
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
              patchelf --set-rpath "\$ORIGIN" "$DEST/libxgbcompat.so"
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
                "-DCMAKE_CXX_STANDARD=20"
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
            cmakeFlags = [ "-DBUILD_TESTING=ON" "-DCMAKE_CXX_STANDARD=20" ];
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
              patchelf --set-rpath "\$ORIGIN" "$DEST/libxgbcompat.so"
              echo "CUDA native libraries copied to $DEST"
              ls -la "$DEST"
            '';
          };
        in
        {
          default = racket;
          inherit cpp cpp-format cpp-line-count cpp-tidy racket copy-native-libs;
        } // nixpkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          inherit polyfill-glibc;
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
        inherit (self.packages.${system}) cpp cpp-format cpp-line-count cpp-tidy racket;
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
              pkgs.clang-tools
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
                echo "Installing Racket linters (Resyntax + racket-review)..."
                # No --no-setup here: let `raco pkg install` run its own setup,
                # which is scoped to the new user-scope packages.  A bare
                # `raco setup` would instead try to recompile the read-only
                # nix-store collections and fail with permission errors.
                raco pkg install --batch --auto --scope user --skip-installed \
                  resyntax review
                touch "$deps_stamp"
                echo "Done. Lint: resyntax analyze --directory xgboost  |  raco review <files>"
              fi
              # Expose user-scope Racket launchers (e.g. `resyntax`) on PATH.
              export PATH="$(racket -e '(require setup/dirs)(display (path->string (find-user-console-bin-dir)))'):$PATH"
            '';
          };
        } // nixpkgs.lib.optionalAttrs hasCuda {
          cuda = pkgs-cuda.mkShell {
            buildInputs = [
              pkgs-cuda.cmake
              pkgs-cuda.clang-tools
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
              export LD_LIBRARY_PATH="${pkgs-cuda.stdenv.cc.cc.lib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              if [ -d /usr/lib/x86_64-linux-gnu ]; then
                _nvidia_stub=$(mktemp -d)
                ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so* "$_nvidia_stub/" 2>/dev/null || true
                ln -sf /usr/lib/x86_64-linux-gnu/libnvidia*.so* "$_nvidia_stub/" 2>/dev/null || true
                export LD_LIBRARY_PATH="$_nvidia_stub:$LD_LIBRARY_PATH"
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
                echo "Done. Run: racket xgboost/examples/24-cuda-regression.rkt"
              fi
            '';
          };
        });
    };
}
