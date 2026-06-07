#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     racket/list
                     xgboost
                     xgboost/private/demo-utils))

@section[#:tag "ex-robust-regression"]{Robust regression}

When training labels contain outliers, squared error over-fits to them and
generalizes worse on clean data. Bounded-influence losses --- Huber
(@racket["reg:pseudohubererror"]) and L1 (@racket["reg:absoluteerror"]) ---
ignore the outlier tails. This example makes the point concretely: take the
Stanford LARS diabetes data, @emph{corrupt 10% of the training labels} by
multiplying them by 5, train three losses on the corrupted data, and score all
three on the @emph{clean} held-out test set.

@chunk[<r07-require>
(require ffi/vector
         racket/list
         xgboost
         xgboost/private/demo-utils)]

@chunk[<r07-provide>
(provide run-example)]

@bold{Helpers.} A deterministic every-5th-row test split, an outlier injector
(every 10th training row, target ×5), and a row→DMatrix packer. Each diabetes
row is 10 features followed by the target @tt{Y}:

@chunk[<r07-helpers>
  (define ncol 10)
  (define (split-rows rows)
    (for/fold ([tr '()] [te '()] #:result (values (reverse tr) (reverse te)))
              ([row (in-list rows)] [i (in-naturals)])
      (if (zero? (modulo i 5)) (values tr (cons row te)) (values (cons row tr) te))))
  (define (corrupt-labels rows)
    (for/list ([row (in-list rows)] [i (in-naturals)])
      (if (zero? (modulo i 10))
          (append (take row ncol) (list (* 5 (last row))))
          row)))
  (define (rows->dmatrix rs)
    (define n (length rs))
    (define features (make-f32vector (* n ncol)))
    (define labels (make-f32vector n))
    (for ([row (in-list rs)] [i (in-naturals)])
      (for ([v (in-list (take row ncol))] [j (in-naturals)])
        (f32vector-set! features (+ (* i ncol) j) (exact->inexact v)))
      (f32vector-set! labels i (exact->inexact (last row))))
    (make-dmatrix features #:nrow n #:ncol ncol #:labels labels))]

@bold{The run.} Train each loss on the corrupted training matrix and score MSE
and MAE against the clean test labels; sweep @racket["huber_slope"] to find a
sensible δ for the data's scale. @racket[run-example] returns the scores:

@chunk[<r07-run>
(define (run-example)
  (define-values (train-rows test-rows) (split-rows (load-diabetes)))
  (define dtrain-dirty (rows->dmatrix (corrupt-labels train-rows)))
  (define dtest (rows->dmatrix test-rows))
  (define test-clean (map last test-rows))
  (define (train-and-score objective extra-params)
    (define b (train dtrain-dirty
                     #:objective objective
                     #:max-depth 4 #:eta 0.1 #:verbosity 0
                     #:params extra-params #:rounds 100))
    (define preds (predict b dtest #:as 'f32vector))
    (for/fold ([sse 0.0] [sae 0.0]
               #:result (values (/ sse (length test-clean)) (/ sae (length test-clean))))
              ([y (in-list test-clean)] [i (in-naturals)])
      (define d (- (f32vector-ref preds i) y))
      (values (+ sse (* d d)) (+ sae (abs d)))))
  (define-values (sq-mse sq-mae) (train-and-score "reg:squarederror" '()))
  (define-values (l1-mse l1-mae) (train-and-score "reg:absoluteerror" '()))
  (define hub-table
    (for/list ([slope (in-list '("1" "5" "25" "100" "500"))])
      (define-values (mse mae)
        (train-and-score "reg:pseudohubererror" (list (cons "huber_slope" slope))))
      (list slope mse mae)))
  (define best (argmin cadr hub-table))
  (hash 'n-train (length train-rows) 'n-test (length test-rows)
        'sq-mse sq-mse 'sq-mae sq-mae
        'l1-mse l1-mse 'l1-mae l1-mae
        'hub-table hub-table
        'hub-slope (car best) 'hub-mse (cadr best) 'hub-mae (caddr best)))]

The harness @filepath{test/07-robust-regression.rkt} prints the huber sweep and
the three-way comparison, and asserts that under outlier-corrupted training the
robust losses beat squared error on the clean test set.

@chunk[<*>
  <r07-require>
  <r07-provide>
  <r07-helpers>
  <r07-run>]
