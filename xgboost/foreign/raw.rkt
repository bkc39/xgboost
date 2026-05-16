#lang racket/base

;; Facade for the raw C FFI layer.  `(require xgboost/foreign/raw)` re-exports
;; the direct `define-ffi-definer` bindings; all implementation lives in
;; raw/{global,dmatrix,booster}.rkt.

(require "raw/global.rkt"
         "raw/dmatrix.rkt"
         "raw/booster.rkt")

(provide (all-from-out "raw/global.rkt")
         (all-from-out "raw/dmatrix.rkt")
         (all-from-out "raw/booster.rkt"))
