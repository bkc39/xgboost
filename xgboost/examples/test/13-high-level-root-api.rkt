#lang racket/base

;; Runner + tests for the literate example ../13-high-level-root-api.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require "../13-high-level-root-api.rkt")

(define (finite-real? v)
  (and (real? v) (= v v) (not (eqv? v +inf.0)) (not (eqv? v -inf.0))))

(module+ main
  (define result (run-example))
  (printf "predictions: ~a\n" (length (hash-ref result 'predictions)))
  (printf "train-rmse: ~a\n" (hash-ref (hash-ref result 'metrics) "train-rmse"))
  (printf "save/load predictions equal: ~a\n"
          (equal? (hash-ref result 'predictions) (hash-ref result 'loaded-predictions))))

(module+ test
  (require rackunit)
  (define result (run-example))
  (define preds (hash-ref result 'predictions))
  (define metrics (hash-ref result 'metrics))
  (check-equal? (length preds) 8)
  (check-true (andmap finite-real? preds))
  (check-true (hash-has-key? metrics "train-rmse"))
  (check-true (finite-real? (hash-ref metrics "train-rmse")))
  (check-equal? (hash-ref result 'loaded-predictions) preds))
