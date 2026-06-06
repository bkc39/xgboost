#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     xgboost))

@section[#:tag "ex-inplace-columnar"]{In-place prediction (columnar)}

@racket[predict-from-columnar] serves predictions from a struct-of-arrays layout
--- one @racket[f32vector] per feature column --- which is how dataframe-style
columnar data is laid out in memory. The same rows expressed densely and
columnar-ly must produce identical predictions.

@chunk[<r22-require>
(require ffi/vector
         xgboost)]

@chunk[<r22-provide>
(provide run-example)]

@bold{Helpers.} The training rows densely, the same rows as three feature
columns, a trained booster, and an approximate equality:

@chunk[<r22-helpers>
  (define dense-features
    (f32vector 1.0 2.0 0.5   2.0 1.0 1.5   3.0 0.5 0.0   0.5 3.0 2.0
               4.0 2.0 1.0   1.5 1.5 0.5   2.5 3.5 1.5   0.0 1.0 0.0))
  (define columnar-features
    (list (f32vector 1.0 2.0 3.0 0.5 4.0 1.5 2.5 0.0)
          (f32vector 2.0 1.0 0.5 3.0 2.0 1.5 3.5 1.0)
          (f32vector 0.5 1.5 0.0 2.0 1.0 0.5 1.5 0.0)))
  (define labels (f32vector 3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0))
  (define (f32vector~= a b)
    (and (= (f32vector-length a) (f32vector-length b))
         (for/and ([i (in-range (f32vector-length a))])
           (< (abs (- (f32vector-ref a i) (f32vector-ref b i))) 1e-4))))
  (define (make-trained)
    (define dtrain (make-dmatrix dense-features #:nrow 8 #:ncol 3 #:missing -1.0 #:labels labels))
    (values (train dtrain #:objective "reg:squarederror"
                   #:max-depth 3 #:eta 0.1 #:verbosity 0 #:rounds 20)
            dtrain))]

@chunk[<r22-run>
(define (run-example)
  (define-values (booster dtrain) (make-trained))
  (define dmatrix-preds (predict booster dtrain #:as 'f32vector))
  (define inplace-preds
    (predict-from-columnar booster columnar-features #:missing -1.0 #:as 'f32vector))
  (hash 'prediction-count (f32vector-length inplace-preds)
        'matches-dmatrix? (f32vector~= inplace-preds dmatrix-preds)))]

The harness @filepath{test/22-inplace-predict-columnar.rkt} prints the count and
asserts the columnar in-place predictions match the DMatrix predictions.

@chunk[<*>
  <r22-require>
  <r22-provide>
  <r22-helpers>
  <r22-run>]
