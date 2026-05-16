#lang racket/base

;; High-level model persistence: filesystem save/load, full-state snapshot
;; bytes, and model bytes round-trips.  The load variants return a fresh
;; booster, so callers never construct one by hand.

(require (prefix-in foreign: "../foreign.rkt"))

(provide make-booster
         save-model
         load-model
         save-model-to-bytes
         load-model-from-bytes
         booster->bytes
         bytes->booster)

(define (make-booster)
  (foreign:booster-create))

(define (save-model b path)
  (foreign:booster-save-model! b path))

(define (load-model path)
  (define b (foreign:booster-create))
  (foreign:booster-load-model! b path)
  b)

(define (save-model-to-bytes b #:format [fmt "ubj"])
  (foreign:booster-save-model-to-bytes b #:format fmt))

(define (load-model-from-bytes bs)
  (define b (foreign:booster-create))
  (foreign:booster-load-model-from-bytes! b bs)
  b)

(define (booster->bytes b)
  (foreign:booster-serialize-to-bytes b))

(define (bytes->booster bs)
  (define b (foreign:booster-create))
  (foreign:booster-unserialize-from-bytes! b bs)
  b)
