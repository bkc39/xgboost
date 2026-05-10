# Darwin Agent Bootstrap Instructions

**After completing all steps, remove this file:**
```bash
git rm DARWIN_BOOTSTRAP.md
git commit -m "Remove darwin bootstrap instructions"
git push
```

## Context

On the `cuda` branch, `candidates/linux-cpu/` and `candidates/linux-aarch64/`
are already committed with self-contained `.so` bundles.  Each bundle contains
`libxgbcompat`, `libxgboost`, and `libgomp` with `$ORIGIN` RPATHs so no Nix
paths leak to non-Nix systems.

You need to do the same for `candidates/darwin/`.  No CUDA — CPU only.  The
macOS equivalents of the Linux steps are:

- `otool -L` instead of `ldd` / `readelf -d`
- `install_name_tool -change <old> <new>` to rewrite embedded Nix store paths
- `install_name_tool -id @rpath/libxgboost.dylib` to fix the library's own ID
- `install_name_tool -delete_rpath <nix-path>` to strip Nix RPATHs
- `install_name_tool -add_rpath @loader_path/.` to set loader-relative lookup

## Step 1 — Run the build script

```bash
./scripts/build-so.sh darwin
```

This calls `nix build .#cpp`, copies `libxgbcompat.dylib` to
`candidates/darwin/`, bundles `libxgboost.dylib`, and adds
`@loader_path/.` to `libxgbcompat.dylib`'s rpath.

## Step 2 — Audit the dylibs for Nix store paths

```bash
DEST=xgboost/native-libs/candidates/darwin
otool -L $DEST/libxgbcompat.dylib
otool -L $DEST/libxgboost.dylib
```

Look for any `/nix/store/...` paths in the output.  For each Nix path in
`libxgboost.dylib`:

```bash
# Fix the library's own install name so the loader uses @rpath
install_name_tool -id @rpath/libxgboost.dylib $DEST/libxgboost.dylib

# Rewrite each absolute Nix dependency to a loader-relative path, e.g.:
install_name_tool -change /nix/store/.../lib/libgomp.dylib \
  @rpath/libgomp.dylib $DEST/libxgboost.dylib

# Strip any Nix store RPATHs baked into libxgboost.dylib:
for rp in $(otool -l $DEST/libxgboost.dylib \
            | awk '/^   path /{print $2}' \
            | grep /nix/store); do
  install_name_tool -delete_rpath "$rp" $DEST/libxgboost.dylib
done

# Ensure @loader_path/. is present on libxgboost.dylib too:
install_name_tool -add_rpath @loader_path/. $DEST/libxgboost.dylib
```

## Step 3 — Handle the OpenMP dependency

XGBoost on macOS may depend on `libomp.dylib` (LLVM OpenMP) rather than
`libgomp.dylib` (GCC OpenMP).  Check the `otool -L` output from step 2 and
find the Nix-store path for whichever OpenMP dylib appears.

Copy and patch it:

```bash
# Copy from the Nix closure — use `nix path-info -r .#cpp` to find it
LIBOMP_SRC=$(nix path-info -r .#cpp | xargs -I{} find {} -name "libomp.dylib" 2>/dev/null | head -1)
# or for libgomp: find {} -name "libgomp.dylib"

cp --no-preserve=mode "$LIBOMP_SRC" $DEST/
install_name_tool -id @rpath/libomp.dylib $DEST/libomp.dylib   # adjust name as needed
for rp in $(otool -l $DEST/libomp.dylib \
            | awk '/^   path /{print $2}' \
            | grep /nix/store); do
  install_name_tool -delete_rpath "$rp" $DEST/libomp.dylib
done
install_name_tool -add_rpath @loader_path/. $DEST/libomp.dylib
```

If the OpenMP library has further Nix-store dependencies visible in `otool -L`,
rewrite those with `install_name_tool -change` too.

## Step 4 — Update bundle_darwin in scripts/build-so.sh

`bundle_darwin` currently only patches `libxgbcompat.dylib`.  Update it to
mirror what `bundle_linux` does:

- Find and copy `libxgboost.dylib` (already done) ✓
- Fix `libxgboost.dylib`'s install name and Nix RPATHs
- Find the OpenMP dylib in the closure, copy and patch it
- Add `@loader_path/.` to both `libxgboost.dylib` and the OpenMP dylib

This ensures re-running `./scripts/build-so.sh darwin` always produces a
correct, portable candidate without manual post-processing.

## Step 5 — Update the pre-installer copy pattern if needed

`xgboost/private/install-xgboost-native.rkt` has:

```racket
(define pattern #rx"^lib(xgbcompat|xgboost|gomp)\\.")
```

If the OpenMP dylib is `libomp.dylib` (not `libgomp`), add `omp` to the
alternation so the pre-installer copies it:

```racket
(define pattern #rx"^lib(xgbcompat|xgboost|gomp|omp)\\.")
```

Only do this if `libomp` is actually present — don't add dead alternatives.

## Step 6 — Run the catalog simulation E2E test

This is the same test documented in AGENTS.md, adapted for the darwin path.
Run inside `nix develop`:

```bash
nix develop --command bash -c "
  set -euo pipefail
  raco pkg remove xgboost 2>/dev/null || true
  rm -f xgboost/native-libs/libxgbcompat.* \
        xgboost/native-libs/libxgboost.*   \
        xgboost/native-libs/libomp.*       \
        xgboost/native-libs/libgomp.*
  echo '--- installing from candidates ---'
  raco pkg install --name xgboost ./xgboost
  echo '--- raco test xgboost/ ---'
  raco test xgboost/
  echo '--- example tests ---'
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
"
```

All 129 tests must pass with no Nix paths on `DYLD_LIBRARY_PATH`.

## Step 7 — Commit and push

```bash
git add xgboost/native-libs/candidates/darwin/ \
        scripts/build-so.sh \
        xgboost/private/install-xgboost-native.rkt
git commit -m "Add darwin candidate: bundle libxgboost.dylib and libomp.dylib with @loader_path RPATHs"
git push
```

Then remove this file:

```bash
git rm DARWIN_BOOTSTRAP.md
git commit -m "Remove darwin bootstrap instructions"
git push
```
