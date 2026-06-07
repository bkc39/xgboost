#lang racket/base

;; Runner + tests for the literate example ../29-learning-to-rank.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require racket/format
         "../29-learning-to-rank.rkt")

(module+ main
  (define r (run-example))
  (printf "learning-to-rank: ~a queries x ~a docs, mean test nDCG@~a = ~a\n"
          (hash-ref r 'n-test-queries) (hash-ref r 'docs-per-query)
          (hash-ref r 'docs-per-query)
          (real->decimal-string (hash-ref r 'mean-ndcg) 4)))

(module+ test
  (require rackunit)
  (define r (run-example))
  ;; group_ptr is the cumulative offset array XGBoost derives from group sizes.
  (check-equal? (hash-ref r 'group-ptr) (hash-ref r 'expected-group-ptr))
  ;; A model that learned the signal ranks far better than chance.
  (check-true (> (hash-ref r 'mean-ndcg) 0.9)
              (format "expected strong ranking, got mean nDCG ~a" (hash-ref r 'mean-ndcg))))
