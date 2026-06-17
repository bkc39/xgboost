#lang racket/base

;; High-level re-exports of the global XGBoost APIs, plus `cuda-available?`,
;; a convenience predicate derived from the build-info JSON.

(require json
         (prefix-in foreign: "../foreign.rkt"))

(provide xgboost-version
         xgboost-build-info
         xgboost-get-global-config
         xgboost-set-global-config!
         xgboost-register-log-callback!
         parse-eval-line
         cuda-available?)

(define xgboost-version foreign:xgboost-version)
(define xgboost-build-info foreign:xgboost-build-info)
(define xgboost-get-global-config foreign:xgboost-get-global-config)
(define xgboost-set-global-config! foreign:xgboost-set-global-config!)
(define xgboost-register-log-callback! foreign:xgboost-register-log-callback!)
(define parse-eval-line foreign:parse-eval-line)

(define (cuda-available?)
  (define info (string->jsexpr (foreign:xgboost-build-info)))
  (if (hash-ref info 'USE_CUDA #f) #t #f))
