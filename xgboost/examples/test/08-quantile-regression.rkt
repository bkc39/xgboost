#lang racket/base

;; Runner + tests for the literate example ../08-quantile-regression.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require racket/format
         racket/list
         "../08-quantile-regression.rkt")

(module+ main
  (define r (run-example))
  (define n-test (hash-ref r 'n-test))
  (printf "predict output: ~a floats (= nrow * n_quantiles)\n" (hash-ref r 'pred-len))
  (printf "  monotonicity: ~a/~a rows have p10 <= p50 <= p90\n"
          (- n-test (hash-ref r 'crossings)) n-test)
  (printf "  coverage:     ~a/~a (~a%) of true Y in [p10, p90]  (target ~~80%)\n"
          (hash-ref r 'coverage-80) n-test
          (~r (* 100 (/ (hash-ref r 'coverage-80) n-test)) #:precision '(= 1)))
  (printf "  mean (p90 - p10) width: ~a\n" (~r (hash-ref r 'mean-band) #:precision '(= 1)))
  (define (col s) (~a s #:width 9 #:align 'right))
  (define (fmt v) (~r v #:precision '(= 1) #:min-width 9))
  (printf "\nsample of test predictions (first 12 rows):\n")
  (printf "  ~a  ~a  ~a  ~a  ~a\n" (~a "i" #:width 3)
          (col "actual") (col "p10") (col "p50") (col "p90"))
  (for ([qs (in-list (take (hash-ref r 'rows-with-q) (min 12 n-test)))]
        [i (in-naturals)])
    (printf "  ~a  ~a  ~a  ~a  ~a\n" (~a i #:width 3)
            (fmt (car qs)) (fmt (cadr qs)) (fmt (caddr qs)) (fmt (cadddr qs)))))

(module+ test
  (require rackunit)
  (define r (run-example))
  (define n-test (hash-ref r 'n-test))
  ;; One model, three quantile columns per row.
  (check-equal? (hash-ref r 'pred-len) (* n-test (hash-ref r 'n-q)))
  ;; Quantiles should be (almost) monotone p10 <= p50 <= p90 across rows.
  (check-true (>= (- n-test (hash-ref r 'crossings)) (* 0.9 n-test))
              (format "too many quantile crossings: ~a/~a" (hash-ref r 'crossings) n-test))
  ;; ~80% nominal coverage; allow a wide band so the test isn't brittle.
  (define cov (/ (hash-ref r 'coverage-80) n-test))
  (check-true (and (> cov 0.5) (< cov 0.98))
              (format "coverage out of expected band: ~a" cov)))
