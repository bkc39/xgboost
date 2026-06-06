#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     racket/list
                     xgboost
                     xgboost/private/demo-utils))

@section[#:tag "ex-poisson-bikes"]{Poisson count regression}

Daily bike-rental counts are nonnegative integers --- exactly the use case for
@racket["count:poisson"]. Its link function makes predictions structurally
nonnegative (@tt{predict = exp(margin)}) and its loss is the Poisson negative
log-likelihood. For contrast we also fit @racket["reg:squarederror"], which will
happily predict @emph{negative} ride counts on low-volume days. The split is
chronological (the data is a time series, so a random split would leak the
future).

@chunk[<r09-require>
(require ffi/vector
         racket/list
         xgboost
         xgboost/private/demo-utils)]

@chunk[<r09-provide>
(provide run-example)]

@bold{Helpers.} Each bike row is 11 features then the count @tt{cnt}:

@chunk[<r09-helpers>
  (define ncol 11)
  (define (rows->dmatrix rs)
    (define n (length rs))
    (define features (make-f32vector (* n ncol)))
    (define labels (make-f32vector n))
    (for ([row (in-list rs)] [i (in-naturals)])
      (for ([v (in-list (take row ncol))] [j (in-naturals)])
        (f32vector-set! features (+ (* i ncol) j) (exact->inexact v)))
      (f32vector-set! labels i (exact->inexact (last row))))
    (make-dmatrix features #:nrow n #:ncol ncol #:labels labels))]

@bold{The run.} Hold out the chronologically last 20% of days, train both
objectives, then compare error and --- the headline --- how many predictions go
negative. @racket[run-example] returns the comparison:

@chunk[<r09-run>
(define (run-example)
  (define data-rows (load-bikes))
  (define-values (train-rows test-rows)
    (split-at data-rows (inexact->exact (round (* 0.8 (length data-rows))))))
  (define dtrain (rows->dmatrix train-rows))
  (define dtest (rows->dmatrix test-rows))
  (define (train-with objective extra)
    (train dtrain #:evals (list (cons "test" dtest))
           #:objective objective #:max-depth 5 #:eta 0.1 #:verbosity 0
           #:params extra #:rounds 200))
  (define poisson (train-with "count:poisson" '(("max_delta_step" . "0.7"))))
  (define gaussian (train-with "reg:squarederror" '()))
  (define actuals (map last test-rows))
  (define (scores preds)
    (define n (length actuals))
    (define-values (sse sae mn neg)
      (for/fold ([sse 0.0] [sae 0.0] [mn +inf.0] [neg 0])
                ([y (in-list actuals)] [i (in-naturals)])
        (define p (f32vector-ref preds i))
        (values (+ sse (expt (- p y) 2)) (+ sae (abs (- p y)))
                (min mn p) (+ neg (if (negative? p) 1 0)))))
    (hash 'rmse (sqrt (/ sse n)) 'mae (/ sae n) 'min mn 'neg neg))
  (define poisson-preds (predict poisson dtest #:as 'f32vector))
  (define gaussian-preds (predict gaussian dtest #:as 'f32vector))
  (hash 'n-test (length test-rows)
        'final-line (eval-one-iter poisson 199
                                   (list (cons "train" dtrain) (cons "test" dtest)))
        'poisson (scores poisson-preds)
        'gaussian (scores gaussian-preds)
        'sample (for/list ([i (in-range (min 12 (length actuals)))] [y (in-list actuals)])
                  (list y (f32vector-ref poisson-preds i) (f32vector-ref gaussian-preds i)))))]

The harness @filepath{test/09-poisson-bikes.rkt} prints the metric comparison
and a sample of held-out days, and asserts that the Poisson model never predicts
a negative count.

@chunk[<*>
  <r09-require>
  <r09-provide>
  <r09-helpers>
  <r09-run>]
