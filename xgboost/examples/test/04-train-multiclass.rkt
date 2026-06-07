#lang racket/base

;; Runner + tests for the literate example ../04-train-multiclass.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require ffi/vector
         racket/format
         xgboost
         "../04-train-multiclass.rkt")

(define nclass 3)

;; Argmax class of row i in a flat nrow*nclass softprob block.
(define (softprob-argmax probs i)
  (define p0 (f32vector-ref probs (+ (* i nclass) 0)))
  (define p1 (f32vector-ref probs (+ (* i nclass) 1)))
  (define p2 (f32vector-ref probs (+ (* i nclass) 2)))
  (cond [(and (>= p0 p1) (>= p0 p2)) 0]
        [(>= p1 p2) 1]
        [else 2]))

(define (softprob-correct probs labels)
  (define nrow (f32vector-length labels))
  (for/sum ([i (in-range nrow)])
    (if (= (softprob-argmax probs i) (inexact->exact (f32vector-ref labels i)))
        1 0)))

(define (softmax-correct preds labels)
  (for/sum ([i (in-range (f32vector-length labels))])
    (if (= (inexact->exact (round (f32vector-ref preds i)))
           (inexact->exact (f32vector-ref labels i)))
        1 0)))

(module+ main
  (define-values (labels probs preds) (run-example))
  (define nrow (f32vector-length labels))
  (define (fmt v) (~r v #:precision '(= 4) #:min-width 7))
  (define (col s) (~a s #:width 7 #:align 'right))
  (printf "softprob output: ~a floats (= nrow * nclass)\n" (f32vector-length probs))
  (printf "  ~a  ~a  ~a  ~a  ~a\n"
          (~a "i" #:width 3) (col "p(0)") (col "p(1)") (col "p(2)") (col "truth"))
  (for ([i (in-range nrow)])
    (printf "  ~a  ~a  ~a  ~a  ~a\n"
            (~a i #:width 3)
            (fmt (f32vector-ref probs (+ (* i nclass) 0)))
            (fmt (f32vector-ref probs (+ (* i nclass) 1)))
            (fmt (f32vector-ref probs (+ (* i nclass) 2)))
            (col (inexact->exact (f32vector-ref labels i)))))
  (printf "softprob argmax accuracy: ~a/~a\n" (softprob-correct probs labels) nrow)
  (printf "\nsoftmax output: ~a floats (= nrow)\n" (f32vector-length preds))
  (printf "softmax accuracy: ~a/~a\n" (softmax-correct preds labels) nrow))

(module+ test
  (require rackunit)
  (define-values (labels probs preds) (run-example))
  (define nrow (f32vector-length labels))
  (check-equal? (f32vector-length probs) (* nrow nclass))
  (check-equal? (f32vector-length preds) nrow)
  ;; Each softprob row is a probability distribution.
  (for ([i (in-range nrow)])
    (define row-sum (+ (f32vector-ref probs (+ (* i nclass) 0))
                       (f32vector-ref probs (+ (* i nclass) 1))
                       (f32vector-ref probs (+ (* i nclass) 2))))
    (check-= row-sum 1.0 1e-5))
  ;; Both objectives recover the well-separated clusters exactly.
  (check-equal? (softprob-correct probs labels) nrow)
  (check-equal? (softmax-correct preds labels) nrow))
