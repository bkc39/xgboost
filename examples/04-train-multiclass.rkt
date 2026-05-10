#lang racket/base

;; Three-class classification on a hand-laid synthetic dataset (3 well-
;; separated clusters in 2-D feature space).  Demonstrates two multiclass
;; objectives:
;;
;;   multi:softprob — output is `nrow * num_class` probabilities; the
;;                    booster-predict resize-on-rc=2 path handles the
;;                    extra width transparently.
;;   multi:softmax  — output is `nrow` predicted class indices.
;;
;; Run from the repo root:
;;   nix develop --command racket examples/04-train-multiclass.rkt

(require ffi/vector
         racket/format
         xgboost)

;; Feature layout: 18 rows x 2 cols, 6 rows per class.  Centers are
;; (0,0), (6,0), (3,6); each cluster is a tight square around its center.
(define features
  (f32vector
   ;; class 0 — around (0, 0)
   0.0  0.0    1.0  0.0    0.0  1.0    -1.0  0.0    0.0 -1.0    1.0  1.0
   ;; class 1 — around (6, 0)
   6.0  0.0    7.0  0.0    6.0  1.0     5.0  0.0    6.0 -1.0    7.0  1.0
   ;; class 2 — around (3, 6)
   3.0  6.0    4.0  6.0    3.0  7.0     2.0  6.0    3.0  5.0    4.0  7.0))

(define labels (f32vector 0.0 0.0 0.0 0.0 0.0 0.0
                          1.0 1.0 1.0 1.0 1.0 1.0
                          2.0 2.0 2.0 2.0 2.0 2.0))

(define nrow 18)
(define ncol 2)
(define nclass 3)

(define dtrain
  (make-dmatrix features #:nrow nrow #:ncol ncol #:labels labels))

(printf "training data:\n")
(dmatrix-show dtrain)
(newline)

(define (train-booster objective)
  (train dtrain
         #:objective objective
         #:num-class nclass
         #:max-depth 3
         #:eta 0.3
         #:verbosity 0
         #:rounds 30))

;; ----- softprob: nrow * nclass probabilities ----------------------------

(define softprob-booster (train-booster "multi:softprob"))
(define probs (predict softprob-booster dtrain #:as 'f32vector))

(printf "softprob output: ~a floats (expected ~a = nrow * nclass)\n"
        (f32vector-length probs) (* nrow nclass))

(define (fmt v) (~r v #:precision '(= 4) #:min-width 7))
(define (col s) (~a s #:width 7 #:align 'right))

(printf "\n  ~a  ~a  ~a  ~a  ~a  ~a  ~a\n"
        (~a "i"     #:width 3)
        (col "x0")  (col "x1")
        (col "p(0)") (col "p(1)") (col "p(2)")
        (col "truth"))

(define n-correct-softprob 0)
(define max-row-sum-err 0.0)

(for ([i (in-range nrow)])
  (define p0 (f32vector-ref probs (+ (* i nclass) 0)))
  (define p1 (f32vector-ref probs (+ (* i nclass) 1)))
  (define p2 (f32vector-ref probs (+ (* i nclass) 2)))
  (define row-sum (+ p0 p1 p2))
  (set! max-row-sum-err (max max-row-sum-err (abs (- row-sum 1.0))))
  (define argmax
    (cond [(and (>= p0 p1) (>= p0 p2)) 0]
          [(>= p1 p2) 1]
          [else 2]))
  (define truth (inexact->exact (f32vector-ref labels i)))
  (when (= argmax truth) (set! n-correct-softprob (+ n-correct-softprob 1)))
  (printf "  ~a  ~a  ~a  ~a  ~a  ~a  ~a\n"
          (~a i #:width 3)
          (fmt (f32vector-ref features (+ (* i ncol) 0)))
          (fmt (f32vector-ref features (+ (* i ncol) 1)))
          (fmt p0) (fmt p1) (fmt p2)
          (col truth)))

(printf "\nsoftprob: max |row-sum - 1| = ~a   (probabilities normalize)\n"
        (~r max-row-sum-err #:precision '(= 6)))
(printf "softprob argmax accuracy: ~a/~a\n" n-correct-softprob nrow)

;; ----- softmax: nrow predicted class indices ----------------------------

(define softmax-booster (train-booster "multi:softmax"))
(define preds (predict softmax-booster dtrain #:as 'f32vector))

(printf "\nsoftmax output: ~a floats (expected ~a = nrow)\n"
        (f32vector-length preds) nrow)

(define n-correct-softmax
  (for/sum ([i (in-range nrow)])
    (define pred  (inexact->exact (round (f32vector-ref preds i))))
    (define truth (inexact->exact (f32vector-ref labels i)))
    (if (= pred truth) 1 0)))
(printf "softmax accuracy: ~a/~a\n" n-correct-softmax nrow)
