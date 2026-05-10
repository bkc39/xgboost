#lang info

(define collection "xgboost")
(define version "0.1.0")
(define deps '("base"))
(define build-deps '("rackunit-lib" "racket-doc" "scribble-lib"))
(define scribblings '(("scribblings/xgboost.scrbl" ())))
(define pkg-desc "Racket bindings for XGBoost")
(define pkg-authors '(bkc))
(define license "Apache-2.0")
(define pkg-tags '("machine-learning" "statistics" "data-science" "regression" "classification"))
(define pre-install-collection "private/install-xgboost-native.rkt")
