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
    ("27-get-started.rkt" 300)))
