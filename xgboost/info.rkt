#lang info

;; raco review lints this as a normal module and flags every `info`
;; definition as unused; #lang info has no value-level uses to detect.
#|review: ignore|#

(define collection "xgboost")
(define version "0.1")
(define deps '("base"))
(define build-deps '("rackunit-lib" "racket-doc" "scribble-lib"))
(define scribblings '(("scribblings/guide.scrbl" (multi-page))
                      ("scribblings/xgboost.scrbl" ())))
(define pkg-desc "Racket bindings for XGBoost")
(define pkg-authors '("bkschemer@gmail.com"))
(define license 'Apache-2.0)
(define pkg-tags '("machine-learning" "statistics" "data-science" "regression" "classification"))
(define pre-install-collection "private/install-xgboost-native.rkt")
