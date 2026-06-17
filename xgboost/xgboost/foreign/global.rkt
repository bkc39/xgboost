#lang racket/base

;; Safe wrappers for the global XGBoost APIs: version, build info, global
;; config, and log-callback registration.

(require ffi/unsafe
         "error.rkt"
         "raw/global.rkt")

(provide xgboost-version
         xgboost-build-info
         xgboost-get-global-config
         xgboost-set-global-config!
         xgboost-register-log-callback!)

(define (xgboost-version)
  (xgb-version/raw))

(define (xgboost-build-info)
  (copy-string-result 'xgboost-build-info xgb-build-info/raw))

(define (xgboost-get-global-config)
  (copy-string-result 'xgboost-get-global-config xgb-get-global-config/raw))

(define (xgboost-set-global-config! config)
  (check-ok (xgb-set-global-config/raw config)
            'xgboost-set-global-config!))

;; Holds the most recently registered callback (proc + its fpointer) so the
;; closure is not collected while XGBoost still holds the function pointer.
(define current-log-callback #f)

(define (xgboost-register-log-callback! proc)
  (define callback-ptr (cast proc _xgb-log-callback _fpointer))
  (check-ok (xgb-register-log-callback/raw callback-ptr)
            'xgboost-register-log-callback!)
  (set! current-log-callback (cons proc callback-ptr)))
