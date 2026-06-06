#lang racket/base

;; Runner + tests for the literate example ../06-iris.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require racket/format
         racket/list
         "../06-iris.rkt")

(define class-names (vector "setosa" "versicolor" "virginica"))

;; Log the first few rounds and every 10th thereafter.
(define (log-iter? i) (or (< i 5) (zero? (modulo (add1 i) 10))))

(module+ main
  (define-values (acc confusion history n-test) (run-example))
  (printf "training (~a rounds, watching mlogloss):\n" (length history))
  (printf "  ~a  ~a  ~a\n"
          (~a "iter" #:width 4)
          (~a "train-mlogloss" #:width 16 #:align 'right)
          (~a "test-mlogloss" #:width 16 #:align 'right))
  (for ([m (in-list history)] [iter (in-naturals)] #:when (log-iter? iter))
    (printf "  ~a  ~a  ~a\n"
            (~a iter #:width 4)
            (~r (hash-ref m "train-mlogloss") #:precision '(= 4) #:min-width 16)
            (~r (hash-ref m "test-mlogloss") #:precision '(= 4) #:min-width 16)))
  (printf "\ntest accuracy: ~a/~a (~a%)\n"
          (* acc n-test) n-test (~r (* 100 acc) #:precision '(= 1)))
  (printf "\nconfusion matrix (rows=truth, cols=pred):\n")
  (printf "  ~a  ~a  ~a  ~a\n" (~a "" #:width 12)
          (~a (vector-ref class-names 0) #:width 10 #:align 'right)
          (~a (vector-ref class-names 1) #:width 10 #:align 'right)
          (~a (vector-ref class-names 2) #:width 10 #:align 'right))
  (for ([truth (in-range 3)])
    (printf "  truth=~a  ~a  ~a  ~a\n"
            (~a (vector-ref class-names truth) #:width 10)
            (~a (vector-ref confusion (+ (* truth 3) 0)) #:width 10 #:align 'right)
            (~a (vector-ref confusion (+ (* truth 3) 1)) #:width 10 #:align 'right)
            (~a (vector-ref confusion (+ (* truth 3) 2)) #:width 10 #:align 'right))))

(module+ test
  (require rackunit)
  (define-values (acc confusion history n-test) (run-example))
  (check-equal? n-test 30)
  (check-equal? (length history) 50)
  ;; Confusion-matrix diagonal must equal the number correct.
  (define diag (+ (vector-ref confusion 0) (vector-ref confusion 4) (vector-ref confusion 8)))
  (check-= (* acc n-test) diag 1e-9)
  ;; mlogloss should fall over training, and iris is easily separable.
  (check-true (< (hash-ref (last history) "test-mlogloss")
                 (hash-ref (first history) "test-mlogloss")))
  (check-true (> acc 0.85) (format "expected accuracy > 0.85, got ~a" acc)))
