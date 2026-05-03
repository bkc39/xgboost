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
              raco test examples/e2e/
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
            text = ''
              DEST="$(pwd)/xgboost/native-libs"
              mkdir -p "$DEST"
              cp -v ${cpp}/lib/libxgbcompat.* "$DEST/"
              echo "Native libraries copied to $DEST"
              ls -la "$DEST"
            '';
          };
        in
        {
          default = racket;
          inherit cpp racket copy-native-libs;
        });

      apps = forAllSystems (system: {
        copy-native-libs = {
          type = "app";
          program = "${self.packages.${system}.copy-native-libs}/bin/copy-native-libs";
        };
      });

      checks = forAllSystems (system: {
        inherit (self.packages.${system}) cpp racket;
      });

      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          cpp = self.packages.${system}.cpp;
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
              deps_stamp="$PLTUSERHOME/.deps-installed-v1"
              if [ ! -f "$deps_stamp" ]; then
                echo "Installing Racket package (link mode)..."
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
        });
    };
}
