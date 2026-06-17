#lang info

;; raco review lints this as a normal module and flags every `info`
;; definition as unused; #lang info has no value-level uses to detect.
#|review: ignore|#

;; Package root for the multi-collection `xgboost` package. This package ships
;; two collections:
;;   - `xgboost`     (./xgboost)     the bindings and native libraries
;;   - `xgboost-doc` (./xgboost-doc) the Scribble manual
;; `raco pkg install ./xgboost` installs both. Package-level metadata lives
;; here; each collection carries its own collection-level info.rkt.
(define collection 'multi)
(define version "0.1")
(define deps '("base"))
(define build-deps '("rackunit-lib" "racket-doc" "scribble-lib"))
(define pkg-desc "Racket bindings for XGBoost")
(define pkg-authors '("bkschemer@gmail.com"))
(define license 'Apache-2.0)
(define pkg-tags '("machine-learning" "statistics" "data-science" "regression" "classification"))

;; The native-library pre-install hook is a *collection-level* setting; it is
;; declared in the `xgboost` collection's own info.rkt (xgboost/info.rkt).
