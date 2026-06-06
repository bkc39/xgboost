#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     xgboost))

@section[#:tag "ex-inplace-dense"]{In-place prediction (dense)}

For serving, building a @tech{DMatrix} per request is wasteful.
@racket[predict-from-dense] runs prediction directly on a row-major
@racket[f32vector], skipping the DMatrix entirely. This example trains a
regressor and checks the in-place result matches the DMatrix path exactly.

@chunk[<r20-require>
(require ffi/vector
         xgboost)]

@chunk[<r20-provide>
(provide run-example)]

@bold{Helpers.} A trained booster and an approximate @racket[f32vector] equality:

@chunk[<r20-helpers>
  (define features
    (f32vector 1.0 2.0 0.5   2.0 1.0 1.5   3.0 0.5 0.0   0.5 3.0 2.0
               4.0 2.0 1.0   1.5 1.5 0.5   2.5 3.5 1.5   0.0 1.0 0.0))
  (define labels (f32vector 3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0))
  (define (f32vector~= a b)
    (and (= (f32vector-length a) (f32vector-length b))
         (for/and ([i (in-range (f32vector-length a))])
           (< (abs (- (f32vector-ref a i) (f32vector-ref b i))) 1e-6))))
  (define (make-trained)
    (define dtrain (make-dmatrix features #:nrow 8 #:ncol 3 #:missing -1.0 #:labels labels))
    (values (train dtrain #:objective "reg:squarederror"
                   #:max-depth 3 #:eta 0.1 #:verbosity 0 #:rounds 20)
            dtrain))]

@chunk[<r20-run>
(define (run-example)
  (define-values (booster dtrain) (make-trained))
  (define dmatrix-preds (predict booster dtrain #:as 'f32vector))
  (define inplace-preds
    (predict-from-dense booster features #:nrow 8 #:ncol 3 #:missing -1.0 #:as 'f32vector))
  (hash 'prediction-count (f32vector-length inplace-preds)
        'inplace-preds inplace-preds
        'dmatrix-preds dmatrix-preds
        'matches-dmatrix? (f32vector~= inplace-preds dmatrix-preds)))]

The harness @filepath{test/20-inplace-predict-dense.rkt} prints the prediction
count and asserts the in-place predictions equal the DMatrix predictions.

@chunk[<*>
  <r20-require>
  <r20-provide>
  <r20-helpers>
  <r20-run>]
