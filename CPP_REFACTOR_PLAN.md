# C++ Refactor And Tooling Plan

## Summary

Refactor the `master` C++ bridge from one large implementation file and one
large test file into small, logical modules with mirrored tests. Preserve the
existing Racket-facing C ABI exactly: all `xgb_*` names, argument types, return
codes, and size-then-fill behavior remain unchanged.

This plan is based on inspecting local `master` without checking it out because
the current worktree was on `publish` with untracked files. Implementation
should begin from `master` after protecting any local work.

## Directory Structure

Target production layout:

```text
cpp/
  CMakeLists.txt
  CMakePresets.json
  cmake/
    XgbcompatWarnings.cmake
    XgbcompatTools.cmake
  include/xgbcompat/
    xgbcompat.hpp          # compatibility umbrella
    demo.hpp               # C++ DemoResult/demo API
    c_api.h                # C ABI umbrella
    c_api/
      global.h
      dmatrix.h
      booster.h
  src/xgbcompat/
    detail/
      buffer.hpp/.cpp      # size-then-fill, NUL-separated, prediction copy helpers
      error.hpp/.cpp       # thread-local error state, XGBoost rc checking
      handles.hpp          # handle casts/static_asserts/RAII handle helpers
    global.cpp             # version, last-error, build-info, global config, logging
    demo.cpp               # regression/classification demo implementation + C wrappers
    dmatrix/
      create.cpp
      info.cpp
      query.cpp
      io.cpp
    booster/
      lifecycle.cpp
      training.cpp
      prediction.cpp
      serialization.cpp
      attributes.cpp
      feature_info.cpp
      dump.cpp
      evaluation.cpp
```

Target test layout mirrors `src/xgbcompat`:

```text
cpp/tests/xgbcompat/
  support/
    array_interface.hpp
    fixtures.hpp/.cpp
    handles.hpp
    prediction.hpp/.cpp
  global_test.cpp
  demo_test.cpp
  c_api_compile_test.c
  dmatrix/
    create_test.cpp
    info_test.cpp
    query_test.cpp
    io_test.cpp
  booster/
    lifecycle_test.cpp
    training_test.cpp
    prediction_test.cpp
    serialization_test.cpp
    attributes_test.cpp
    feature_info_test.cpp
    dump_test.cpp
    evaluation_test.cpp
```

All C++ source, header, and test files should be kept under 500 lines, enforced
by a Nix check.

## Key Changes

- Keep `#include <xgbcompat/xgbcompat.hpp>` working, but make it an umbrella
  over focused public headers.
- Add a pure C public header, `xgbcompat/c_api.h`, and a small C compile test to
  prove the C ABI remains consumable outside C++.
- Replace `src/internal.hpp` with focused helpers in `src/xgbcompat/detail`.
- Use internal namespaces:
  - `xgbcompat::detail` for shared ABI/error/buffer/handle utilities.
  - `xgbcompat::dmatrix` and `xgbcompat::booster` for domain helpers.
  - `extern "C"` functions remain global and are implemented in the relevant
    module files.
- Keep the library target name and installed shared library name as
  `xgbcompat`.
- Lower the required C++ standard to C++20 unless baseline build proves a
  current dependency requires C++26.

## Build, Tooling, And CI

- Update CMake to list sources explicitly via `target_sources`, keep
  `BUILD_TESTING`, export compile commands, and build one `xgbcompat_tests`
  executable from the mirrored test tree.
- Add `.clang-format` based on Google style with the existing two-space feel and
  run `clang-format --dry-run --Werror` in checks.
- Add `.clang-tidy` with a practical profile: `bugprone-*`, `performance-*`,
  `readability-*`, selected `modernize-*`, with noisy FFI-hostile checks
  disabled.
- Add CMake targets for local use: `format-check`, `tidy`, and normal `ctest`.
- Add Nix checks:
  - `cpp` build plus CTest.
  - `cpp-format`.
  - `cpp-tidy`.
  - `cpp-line-count`.
  - `nix-format` or basic Nix linting for touched Nix files.
- Update the dev shell to include `clang-tools`, CMake, Ninja, GTest, XGBoost,
  and the formatting/linting tools.
- Add GitHub Actions on PR and `master` push to run
  `nix flake check --print-build-logs`; Linux runs the full quality suite,
  macOS at least builds/tests `.#cpp`.

## Test Plan

- Before refactor: run `nix build .#cpp` or `cmake --build` plus `ctest` on
  `master` to capture baseline.
- After refactor:
  - `nix build .#cpp`
  - `nix flake check --print-build-logs`
  - `ctest --test-dir cpp/build --output-on-failure`
  - `raco test xgboost/` or full `nix build` to verify the Racket FFI still
    loads the same C ABI.
- Preserve existing behavioral coverage by moving tests, not deleting scenarios.
- Add one C-only compile/link smoke test for `xgbcompat/c_api.h`.

## Assumptions

- This first pass is a structural refactor plus tooling pass, not a public API
  redesign.
- The Racket FFI files should not need changes except to react to an accidental
  ABI break, which should be treated as a bug.
- CI can start with Nix-based checks instead of adding a separate non-Nix CMake
  workflow.
