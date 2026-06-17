#lang info

;; raco review lints this as a normal module and flags every `info`
;; definition as unused; #lang info has no value-level uses to detect.
#|review: ignore|#

;; Documentation collection for the `xgboost` package. The rendered manual is
;; built from xgboost.scrbl; the scribble build-deps (scribble-lib, racket-doc)
;; are declared in the package-root info.rkt.
(define scribblings '(("xgboost.scrbl" ())))
