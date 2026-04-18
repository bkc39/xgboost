#lang racket/base

(require ffi/unsafe
         ffi/unsafe/define
         racket/runtime-path)

(provide xgb-version/raw
         xgb-last-error/raw
         xgb-run-regression-demo/raw
         xgb-run-classification-demo/raw)

(define-runtime-path native-libs-dir "../native-libs")

(define-ffi-definer define-xgb
  (ffi-lib (build-path native-libs-dir "libxgbcompat")))

(define-xgb xgb-version/raw
  (_fun -> _string/utf-8)
  #:c-id xgb_version)

(define-xgb xgb-last-error/raw
  (_fun -> _string/utf-8)
  #:c-id xgb_last_error)

(define-xgb xgb-run-regression-demo/raw
  (_fun (out : (_ptr o _double)) -> (rc : _int) -> (values rc out))
  #:c-id xgb_run_regression_demo)

(define-xgb xgb-run-classification-demo/raw
  (_fun (out : (_ptr o _double)) -> (rc : _int) -> (values rc out))
  #:c-id xgb_run_classification_demo)
