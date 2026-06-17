#lang racket/base

;; Test-only runtime hardening, required *first* by the test modules (before
;; `xgboost`/`xgboost/foreign`) so it takes effect before the native library's
;; OpenMP runtime is first exercised.
;;
;; The package-build service runs `raco test --drdr`, which imposes a 90s
;; per-test timeout and a ~1 GB container memory cap. XGBoost defaults to one
;; OpenMP thread per core; on a busy many-core builder that multiplies both
;; scheduling overhead and per-thread native memory for no benefit on these
;; tiny fixtures. Pinning OpenMP to a single thread keeps the suite fast and
;; light there, reducing the chance of a transient timeout/OOM kill.
;;
;; Honors an existing OMP_NUM_THREADS so a developer can override it.

(unless (getenv "OMP_NUM_THREADS")
  (void (putenv "OMP_NUM_THREADS" "1")))
