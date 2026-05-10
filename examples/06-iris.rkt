#lang racket/base

;; "Batteries-included" Iris classification example.
;;
;; The full UCI Iris dataset (150 rows, 4 features, 3 classes) is embedded
;; below as a CSV string literal — no network or filesystem needed to run.
;; The flow shows the full real-data path:
;;
;;   1. parse the CSV → list of (features-list, label-int) rows
;;   2. deterministic 80/20 train/test split (every 5th row → test)
;;   3. flatten into row-major f32vector → DMatrix + label info
;;   4. train multi:softprob, log mlogloss on both splits per round
;;   5. predict on the held-out test set; report accuracy + confusion matrix
;;
;; Run from the repo root:
;;   nix develop --command racket examples/06-iris.rkt
;;
;; Source: UCI Machine Learning Repository.
;;   https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

(require ffi/vector
         racket/format
         racket/string
         xgboost)

(define iris-csv #<<IRIS-CSV
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5.0,3.6,1.4,0.2,Iris-setosa
5.4,3.9,1.7,0.4,Iris-setosa
4.6,3.4,1.4,0.3,Iris-setosa
5.0,3.4,1.5,0.2,Iris-setosa
4.4,2.9,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
5.4,3.7,1.5,0.2,Iris-setosa
4.8,3.4,1.6,0.2,Iris-setosa
4.8,3.0,1.4,0.1,Iris-setosa
4.3,3.0,1.1,0.1,Iris-setosa
5.8,4.0,1.2,0.2,Iris-setosa
5.7,4.4,1.5,0.4,Iris-setosa
5.4,3.9,1.3,0.4,Iris-setosa
5.1,3.5,1.4,0.3,Iris-setosa
5.7,3.8,1.7,0.3,Iris-setosa
5.1,3.8,1.5,0.3,Iris-setosa
5.4,3.4,1.7,0.2,Iris-setosa
5.1,3.7,1.5,0.4,Iris-setosa
4.6,3.6,1.0,0.2,Iris-setosa
5.1,3.3,1.7,0.5,Iris-setosa
4.8,3.4,1.9,0.2,Iris-setosa
5.0,3.0,1.6,0.2,Iris-setosa
5.0,3.4,1.6,0.4,Iris-setosa
5.2,3.5,1.5,0.2,Iris-setosa
5.2,3.4,1.4,0.2,Iris-setosa
4.7,3.2,1.6,0.2,Iris-setosa
4.8,3.1,1.6,0.2,Iris-setosa
5.4,3.4,1.5,0.4,Iris-setosa
5.2,4.1,1.5,0.1,Iris-setosa
5.5,4.2,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
5.0,3.2,1.2,0.2,Iris-setosa
5.5,3.5,1.3,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
4.4,3.0,1.3,0.2,Iris-setosa
5.1,3.4,1.5,0.2,Iris-setosa
5.0,3.5,1.3,0.3,Iris-setosa
4.5,2.3,1.3,0.3,Iris-setosa
4.4,3.2,1.3,0.2,Iris-setosa
5.0,3.5,1.6,0.6,Iris-setosa
5.1,3.8,1.9,0.4,Iris-setosa
4.8,3.0,1.4,0.3,Iris-setosa
5.1,3.8,1.6,0.2,Iris-setosa
4.6,3.2,1.4,0.2,Iris-setosa
5.3,3.7,1.5,0.2,Iris-setosa
5.0,3.3,1.4,0.2,Iris-setosa
7.0,3.2,4.7,1.4,Iris-versicolor
6.4,3.2,4.5,1.5,Iris-versicolor
6.9,3.1,4.9,1.5,Iris-versicolor
5.5,2.3,4.0,1.3,Iris-versicolor
6.5,2.8,4.6,1.5,Iris-versicolor
5.7,2.8,4.5,1.3,Iris-versicolor
6.3,3.3,4.7,1.6,Iris-versicolor
4.9,2.4,3.3,1.0,Iris-versicolor
6.6,2.9,4.6,1.3,Iris-versicolor
5.2,2.7,3.9,1.4,Iris-versicolor
5.0,2.0,3.5,1.0,Iris-versicolor
5.9,3.0,4.2,1.5,Iris-versicolor
6.0,2.2,4.0,1.0,Iris-versicolor
6.1,2.9,4.7,1.4,Iris-versicolor
5.6,2.9,3.6,1.3,Iris-versicolor
6.7,3.1,4.4,1.4,Iris-versicolor
5.6,3.0,4.5,1.5,Iris-versicolor
5.8,2.7,4.1,1.0,Iris-versicolor
6.2,2.2,4.5,1.5,Iris-versicolor
5.6,2.5,3.9,1.1,Iris-versicolor
5.9,3.2,4.8,1.8,Iris-versicolor
6.1,2.8,4.0,1.3,Iris-versicolor
6.3,2.5,4.9,1.5,Iris-versicolor
6.1,2.8,4.7,1.2,Iris-versicolor
6.4,2.9,4.3,1.3,Iris-versicolor
6.6,3.0,4.4,1.4,Iris-versicolor
6.8,2.8,4.8,1.4,Iris-versicolor
6.7,3.0,5.0,1.7,Iris-versicolor
6.0,2.9,4.5,1.5,Iris-versicolor
5.7,2.6,3.5,1.0,Iris-versicolor
5.5,2.4,3.8,1.1,Iris-versicolor
5.5,2.4,3.7,1.0,Iris-versicolor
5.8,2.7,3.9,1.2,Iris-versicolor
6.0,2.7,5.1,1.6,Iris-versicolor
5.4,3.0,4.5,1.5,Iris-versicolor
6.0,3.4,4.5,1.6,Iris-versicolor
6.7,3.1,4.7,1.5,Iris-versicolor
6.3,2.3,4.4,1.3,Iris-versicolor
5.6,3.0,4.1,1.3,Iris-versicolor
5.5,2.5,4.0,1.3,Iris-versicolor
5.5,2.6,4.4,1.2,Iris-versicolor
6.1,3.0,4.6,1.4,Iris-versicolor
5.8,2.6,4.0,1.2,Iris-versicolor
5.0,2.3,3.3,1.0,Iris-versicolor
5.6,2.7,4.2,1.3,Iris-versicolor
5.7,3.0,4.2,1.2,Iris-versicolor
5.7,2.9,4.2,1.3,Iris-versicolor
6.2,2.9,4.3,1.3,Iris-versicolor
5.1,2.5,3.0,1.1,Iris-versicolor
5.7,2.8,4.1,1.3,Iris-versicolor
6.3,3.3,6.0,2.5,Iris-virginica
5.8,2.7,5.1,1.9,Iris-virginica
7.1,3.0,5.9,2.1,Iris-virginica
6.3,2.9,5.6,1.8,Iris-virginica
6.5,3.0,5.8,2.2,Iris-virginica
7.6,3.0,6.6,2.1,Iris-virginica
4.9,2.5,4.5,1.7,Iris-virginica
7.3,2.9,6.3,1.8,Iris-virginica
6.7,2.5,5.8,1.8,Iris-virginica
7.2,3.6,6.1,2.5,Iris-virginica
6.5,3.2,5.1,2.0,Iris-virginica
6.4,2.7,5.3,1.9,Iris-virginica
6.8,3.0,5.5,2.1,Iris-virginica
5.7,2.5,5.0,2.0,Iris-virginica
5.8,2.8,5.1,2.4,Iris-virginica
6.4,3.2,5.3,2.3,Iris-virginica
6.5,3.0,5.5,1.8,Iris-virginica
7.7,3.8,6.7,2.2,Iris-virginica
7.7,2.6,6.9,2.3,Iris-virginica
6.0,2.2,5.0,1.5,Iris-virginica
6.9,3.2,5.7,2.3,Iris-virginica
5.6,2.8,4.9,2.0,Iris-virginica
7.7,2.8,6.7,2.0,Iris-virginica
6.3,2.7,4.9,1.8,Iris-virginica
6.7,3.3,5.7,2.1,Iris-virginica
7.2,3.2,6.0,1.8,Iris-virginica
6.2,2.8,4.8,1.8,Iris-virginica
6.1,3.0,4.9,1.8,Iris-virginica
6.4,2.8,5.6,2.1,Iris-virginica
7.2,3.0,5.8,1.6,Iris-virginica
7.4,2.8,6.1,1.9,Iris-virginica
7.9,3.8,6.4,2.0,Iris-virginica
6.4,2.8,5.6,2.2,Iris-virginica
6.3,2.8,5.1,1.5,Iris-virginica
6.1,2.6,5.6,1.4,Iris-virginica
7.7,3.0,6.1,2.3,Iris-virginica
6.3,3.4,5.6,2.4,Iris-virginica
6.4,3.1,5.5,1.8,Iris-virginica
6.0,3.0,4.8,1.8,Iris-virginica
6.9,3.1,5.4,2.1,Iris-virginica
6.7,3.1,5.6,2.4,Iris-virginica
6.9,3.1,5.1,2.3,Iris-virginica
5.8,2.7,5.1,1.9,Iris-virginica
6.8,3.2,5.9,2.3,Iris-virginica
6.7,3.3,5.7,2.5,Iris-virginica
6.7,3.0,5.2,2.3,Iris-virginica
6.3,2.5,5.0,1.9,Iris-virginica
6.5,3.0,5.2,2.0,Iris-virginica
6.2,3.4,5.4,2.3,Iris-virginica
5.9,3.0,5.1,1.8,Iris-virginica
IRIS-CSV
)

(define class-names (vector "setosa" "versicolor" "virginica"))
(define class-index
  (hash "Iris-setosa"     0
        "Iris-versicolor" 1
        "Iris-virginica"  2))

;; ----- Parse the CSV ------------------------------------------------------

(define rows
  (for/list ([line (in-list (string-split iris-csv "\n"))]
             #:when (positive? (string-length (string-trim line))))
    (define parts (string-split line ","))
    (cons (for/list ([p (in-list (list (list-ref parts 0) (list-ref parts 1)
                                       (list-ref parts 2) (list-ref parts 3)))])
            (string->number p))
          (hash-ref class-index (list-ref parts 4)))))

(printf "loaded ~a rows from embedded CSV\n" (length rows))

;; ----- Train/test split ---------------------------------------------------

;; Every 5th row (in CSV order, which interleaves classes by stride 50) goes
;; to test; the remaining 4/5 form the training set.  Deterministic and
;; class-balanced: 30 test rows / 120 train rows, 10 per class in test.
(define-values (train-rows test-rows)
  (for/fold ([train '()] [test '()])
            ([row (in-list rows)] [i (in-naturals)])
    (if (zero? (modulo i 5))
        (values train (cons row test))
        (values (cons row train) test))))

(printf "  train: ~a rows, test: ~a rows\n"
        (length train-rows) (length test-rows))

;; ----- Build DMatrices ---------------------------------------------------

(define ncol 4)

(define (rows->dmatrix rs)
  (define n (length rs))
  (define features (make-f32vector (* n ncol)))
  (define labels (make-f32vector n))
  (for ([row (in-list rs)] [i (in-naturals)])
    (for ([v (in-list (car row))] [j (in-naturals)])
      (f32vector-set! features (+ (* i ncol) j) (exact->inexact v)))
    (f32vector-set! labels i (exact->inexact (cdr row))))
  (values (make-dmatrix features #:nrow n #:ncol ncol #:labels labels)
          labels))

(define-values (dtrain train-labels) (rows->dmatrix train-rows))
(define-values (dtest  test-labels)  (rows->dmatrix test-rows))

;; ----- Train ------------------------------------------------------------

(define booster
  (train dtrain
         #:evals (list (cons "test" dtest))
         #:objective "multi:softprob"
         #:num-class 3
         #:eval-metric "mlogloss"
         #:max-depth 4
         #:eta 0.3
         #:verbosity 0
         #:rounds 0))

(define rounds 50)
(define eval-set (list (cons "train" dtrain) (cons "test" dtest)))

(printf "\ntraining (~a rounds, watching mlogloss):\n" rounds)
(printf "  ~a  ~a  ~a\n"
        (~a "iter" #:width 4)
        (~a "train-mlogloss" #:width 16 #:align 'right)
        (~a "test-mlogloss"  #:width 16 #:align 'right))

(define (log-iter? i)
  (or (< i 5) (zero? (modulo (+ i 1) 10))))

(for ([iter (in-range rounds)])
  (booster-update-one-iter! booster iter dtrain)
  (when (log-iter? iter)
    (define m (parse-eval-line (eval-one-iter booster iter eval-set)))
    (printf "  ~a  ~a  ~a\n"
            (~a iter #:width 4)
            (~r (hash-ref m "train-mlogloss") #:precision '(= 4) #:min-width 16)
            (~r (hash-ref m "test-mlogloss")  #:precision '(= 4) #:min-width 16))))

;; ----- Evaluate on the held-out test set --------------------------------

(define n-test (length test-rows))
(define probs (predict booster dtest #:as 'f32vector))

(define preds
  (for/list ([i (in-range n-test)])
    (define p0 (f32vector-ref probs (+ (* i 3) 0)))
    (define p1 (f32vector-ref probs (+ (* i 3) 1)))
    (define p2 (f32vector-ref probs (+ (* i 3) 2)))
    (cond [(and (>= p0 p1) (>= p0 p2)) 0]
          [(>= p1 p2) 1]
          [else 2])))

(define correct
  (for/sum ([pred (in-list preds)] [i (in-range n-test)])
    (if (= pred (inexact->exact (f32vector-ref test-labels i))) 1 0)))

;; 3x3 confusion matrix indexed as truth*3 + pred.
(define confusion (make-vector 9 0))
(for ([pred (in-list preds)] [i (in-range n-test)])
  (define truth (inexact->exact (f32vector-ref test-labels i)))
  (define cell (+ (* truth 3) pred))
  (vector-set! confusion cell (+ 1 (vector-ref confusion cell))))

(printf "\ntest accuracy: ~a/~a (~a%)\n"
        correct n-test
        (~r (* 100 (/ correct n-test)) #:precision '(= 1)))

(printf "\nconfusion matrix (rows=truth, cols=pred):\n")
(printf "  ~a  ~a  ~a  ~a\n"
        (~a "" #:width 14)
        (~a (vector-ref class-names 0) #:width 10 #:align 'right)
        (~a (vector-ref class-names 1) #:width 10 #:align 'right)
        (~a (vector-ref class-names 2) #:width 10 #:align 'right))
(for ([truth (in-range 3)])
  (printf "  truth=~a  ~a  ~a  ~a\n"
          (~a (vector-ref class-names truth) #:width 10)
          (~a (vector-ref confusion (+ (* truth 3) 0)) #:width 10 #:align 'right)
          (~a (vector-ref confusion (+ (* truth 3) 1)) #:width 10 #:align 'right)
          (~a (vector-ref confusion (+ (* truth 3) 2)) #:width 10 #:align 'right)))
