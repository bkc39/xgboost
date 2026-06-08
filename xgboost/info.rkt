#lang info

;; raco review lints this as a normal module and flags every `info`
;; definition as unused; #lang info has no value-level uses to detect.
#|review: ignore|#

(define collection "xgboost")
(define version "0.1")
;; `scribble-lib` is a runtime dep (not build-only): the in-package
;; `examples/*.rkt` are `#lang scribble/lp2` literate programs, so their
;; compiled form requires scribble's lp2 module language at run time.
(define deps '("base" "net-lib" "scribble-lib"))
(define build-deps '("rackunit-lib" "racket-doc"))
(define scribblings '(("scribblings/xgboost.scrbl" (multi-page))))
(define pkg-desc "Racket bindings for XGBoost")
(define pkg-authors '("bkschemer@gmail.com"))
(define license 'Apache-2.0)
(define pkg-tags '("machine-learning" "statistics" "data-science" "regression" "classification"))
(define pre-install-collection "private/install-xgboost-native.rkt")

;; `raco test --drdr` (used by pkg-build.racket-lang.org) defaults to a 90s
;; per-test timeout. Give the native-heavy integration tests generous headroom
;; so a slow/loaded builder cannot transiently time them out.
(define test-timeouts
  '(("tests/core-test.rkt" 300)
    ("tests/foreign-booster-test.rkt" 300)
    ("tests/foreign-dmatrix-test.rkt" 300)
    ("private/lifetime-test.rkt" 300)))
