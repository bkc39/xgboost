#lang racket/base

;; Raw global XGBoost APIs: version, last-error, build info, global config,
;; and log-callback registration.

(require ffi/unsafe
         "library.rkt")

(provide xgb-version/raw
         xgb-last-error/raw
         _xgb-log-callback
         xgb-build-info/raw
         xgb-get-global-config/raw
         xgb-set-global-config/raw
         xgb-register-log-callback/raw)

(define-xgb xgb-version/raw
  (_fun -> _string/utf-8)
  #:c-id xgb_version)

(define-xgb xgb-last-error/raw
  (_fun -> _string/utf-8)
  #:c-id xgb_last_error)

(define _xgb-log-callback
  (_fun _string/utf-8 -> _void))

(define-xgb xgb-build-info/raw
  (_fun (capacity : _uint64)
        (buf : _bytes)
        (out-len : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-len))
  #:c-id xgb_build_info)

(define-xgb xgb-get-global-config/raw
  (_fun (capacity : _uint64)
        (buf : _bytes)
        (out-len : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-len))
  #:c-id xgb_get_global_config)

(define-xgb xgb-set-global-config/raw
  (_fun _string/utf-8 -> _int)
  #:c-id xgb_set_global_config)

(define-xgb xgb-register-log-callback/raw
  (_fun _fpointer -> _int)
  #:c-id xgb_register_log_callback)
