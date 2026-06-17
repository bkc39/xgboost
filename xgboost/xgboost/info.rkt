#lang info

;; raco review lints this as a normal module and flags every `info`
;; definition as unused; #lang info has no value-level uses to detect.
#|review: ignore|#

;; Collection-level info for the `xgboost` code collection. Package-level
;; metadata (version, deps, license) lives in the package-root info.rkt one
;; directory up.

;; Selects and copies the platform's native libraries into native-libs/ at
;; install time. `pre-install-collection` is a collection-level setting, so it
;; must live here (not in the package root); the path is relative to this
;; collection.
(define pre-install-collection "private/install-xgboost-native.rkt")

;; `raco test --drdr` (used by pkg-build.racket-lang.org) defaults to a 90s
;; per-test timeout. Give the native-heavy integration tests generous headroom
;; so a slow/loaded builder cannot transiently time them out. Paths are
;; relative to this collection.
(define test-timeouts
  '(("tests/core-test.rkt" 300)
    ("tests/foreign-booster-test.rkt" 300)
    ("tests/foreign-dmatrix-test.rkt" 300)
    ("private/lifetime-test.rkt" 300)))
