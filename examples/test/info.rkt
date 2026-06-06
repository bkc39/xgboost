#lang info

;; raco review lints this as a normal module and flags every `info`
;; definition as unused; #lang info has no value-level uses to detect.
#|review: ignore|#

;; Test harnesses for the literate `examples/*.rkt` programs.  Each `NN-name.rkt`
;; here requires its `#lang scribble/lp2` sibling `../NN-name.rkt` (which exports
;; a `run-example` thunk), drives it from `(module+ main ...)`, and checks it from
;; `(module+ test ...)`.  Run the whole suite with `raco test examples/test/`.

;; The native-backed demos can be slow on a loaded/`--drdr` builder; give them the
;; same generous per-test headroom as the package integration tests.
(define test-timeouts
  '(("00-print-dmatrix.rkt" 300)
    ("01-train-regression.rkt" 300)
    ("02-train-classifier.rkt" 300)
    ("04-train-multiclass.rkt" 300)
    ("05-train-with-eval.rkt" 300)
    ("06-iris.rkt" 300)
    ("27-get-started.rkt" 300)
    ("07-robust-regression.rkt" 300)
    ("08-quantile-regression.rkt" 300)
    ("09-poisson-bikes.rkt" 300)
    ("10-aft-survival.rkt" 300)
    ("23-custom-objective.rkt" 300)
    ("03-save-load.rkt" 300)
    ("26-booster-snapshot.rkt" 300)
    ("12-dmatrix-constructors.rkt" 300)
    ("14-dmatrix-metadata.rkt" 300)
    ("15-dmatrix-slicing-binary.rkt" 300)
    ("16-quantile-cuts.rkt" 300)
    ("13-high-level-root-api.rkt" 300)
    ("17-booster-lifecycle-config.rkt" 300)
    ("18-booster-attrs.rkt" 300)
    ("19-booster-dumps-feature-scores.rkt" 300)
    ("20-inplace-predict-dense.rkt" 300)
    ("21-inplace-predict-csr.rkt" 300)
    ("22-inplace-predict-columnar.rkt" 300)
    ("28-param-recipes.rkt" 300)
    ("29-learning-to-rank.rkt" 300)
    ("11-global-apis.rkt" 300)
    ("24-cuda-regression.rkt" 300)
    ("25-cuda-classification.rkt" 300)))
