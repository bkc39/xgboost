#lang info

(define collection "xgboost")
(define version "0.1.0")
(define deps '("base"))
(define build-deps '("rackunit-lib"))
(define scribblings '())
(define pkg-desc "Racket bindings for XGBoost")
(define pkg-authors '(bkc))
(define pre-install-collection "private/install-xgboost-native.rkt")
