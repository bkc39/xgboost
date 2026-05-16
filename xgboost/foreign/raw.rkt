#lang racket/base

;; raco review does surface-level linting without macro expansion; it would
;; report every `all-from-out` re-export below as "provided but not defined".
;; This is a pure re-export facade with nothing else to lint.
#|review: ignore|#

;; Facade for the raw C FFI layer.  `(require xgboost/foreign/raw)` re-exports
;; the direct `define-ffi-definer` bindings; all implementation lives in
;; raw/{global,dmatrix,booster}.rkt.

(require "raw/global.rkt"
         "raw/dmatrix.rkt"
         "raw/booster.rkt")

(provide (all-from-out "raw/global.rkt")
         (all-from-out "raw/dmatrix.rkt")
         (all-from-out "raw/booster.rkt"))
