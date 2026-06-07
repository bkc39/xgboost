#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     xgboost))

@section[#:tag "ex-custom-objective"]{Custom objective}

Instead of naming a built-in @racket[#:objective], you can supply your own loss
as a Racket procedure via @racket[#:objective-fn]. Each boosting round XGBoost
calls it with the current predictions and the training matrix, and expects back
two @racket[f32vector]s: the per-row gradient and Hessian of your loss. This
example reimplements ordinary squared error that way --- the gradient of
@tt{½(pred − label)²} is @tt{pred − label} and the Hessian is @tt{1} --- and
checks the model improves over its untrained baseline.

@chunk[<r23-require>
(require ffi/vector
         xgboost)]

@chunk[<r23-provide>
(provide run-example)]

@bold{The data.} The same eight-row synthetic regression set as
@secref["ex-train-regression"], plus a mean-squared-error helper:

@chunk[<r23-data>
  (define features
    (f32vector 1.0 2.0 0.5   2.0 1.0 1.5   3.0 0.5 0.0   0.5 3.0 2.0
               4.0 2.0 1.0   1.5 1.5 0.5   2.5 3.5 1.5   0.0 1.0 0.0))
  (define labels (f32vector 3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0))
  (define (mse preds)
    (/ (for/sum ([i (in-range (f32vector-length labels))])
         (define d (- (f32vector-ref preds i) (f32vector-ref labels i)))
         (* d d))
       (f32vector-length labels)))]

@bold{The objective.} A custom objective returns @racket[(values grad hess)]:

@chunk[<r23-objective>
  (define (squared-error preds _dtrain)
    (define n (f32vector-length preds))
    (define grad (make-f32vector n))
    (define hess (make-f32vector n 1.0))
    (for ([i (in-range n)])
      (f32vector-set! grad i (- (f32vector-ref preds i)
                                (f32vector-ref labels i))))
    (values grad hess))]

@bold{The run.} Measure the baseline margin error from a zero-round booster,
then train 20 rounds under the custom objective and compare. @racket[run-example]
returns a hash of the prediction count and the before/after MSE:

@chunk[<r23-run>
(define (run-example)
  (define dtrain
    (make-dmatrix features #:nrow 8 #:ncol 3 #:missing -1.0 #:labels labels))
  (define baseline (train dtrain #:max-depth 3 #:eta 0.2 #:verbosity 0 #:rounds 0))
  (define initial-mse
    (mse (predict baseline dtrain #:output 'margin #:as 'f32vector)))
  (define booster
    (train dtrain
           #:objective-fn squared-error
           #:max-depth 3
           #:eta 0.2
           #:verbosity 0
           #:rounds 20))
  (define final-preds (predict booster dtrain #:as 'f32vector))
  (define final-mse (mse final-preds))
  (hash 'prediction-count (f32vector-length final-preds)
        'initial-mse initial-mse
        'final-mse final-mse
        'improved? (< final-mse initial-mse)))]

The harness @filepath{test/23-custom-objective.rkt} prints the before/after MSE
and asserts the custom-objective model fits.

@chunk[<*>
  <r23-require>
  <r23-provide>
  <r23-data>
  <r23-objective>
  <r23-run>]
