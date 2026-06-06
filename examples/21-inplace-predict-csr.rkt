#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     xgboost))

@section[#:tag "ex-inplace-csr"]{In-place prediction (CSR)}

@racket[predict-from-csr] serves predictions directly from a CSR sparse layout
--- an @racket[indptr] / @racket[indices] / @racket[values] triple --- without
building a @tech{DMatrix}. Here the rows happen to be dense, so the result must
match the DMatrix path exactly.

@chunk[<r21-require>
(require ffi/vector
         xgboost)]

@chunk[<r21-provide>
(provide run-example)]

@bold{Helpers.} The dense values plus the CSR structure that addresses them, a
trained booster, and an approximate equality:

@chunk[<r21-helpers>
  (define dense-features
    (f32vector 1.0 2.0 0.5   2.0 1.0 1.5   3.0 0.5 0.0   0.5 3.0 2.0
               4.0 2.0 1.0   1.5 1.5 0.5   2.5 3.5 1.5   0.0 1.0 0.0))
  (define labels (f32vector 3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0))
  (define csr-indptr (u64vector 0 3 6 9 12 15 18 21 24))
  (define csr-indices
    (u32vector 0 1 2  0 1 2  0 1 2  0 1 2  0 1 2  0 1 2  0 1 2  0 1 2))
  (define (f32vector~= a b)
    (and (= (f32vector-length a) (f32vector-length b))
         (for/and ([i (in-range (f32vector-length a))])
           (< (abs (- (f32vector-ref a i) (f32vector-ref b i))) 1e-4))))
  (define (make-trained)
    (define dtrain (make-dmatrix dense-features #:nrow 8 #:ncol 3 #:missing -1.0 #:labels labels))
    (values (train dtrain #:objective "reg:squarederror"
                   #:max-depth 3 #:eta 0.1 #:verbosity 0 #:rounds 20)
            dtrain))]

@chunk[<r21-run>
(define (run-example)
  (define-values (booster dtrain) (make-trained))
  (define dmatrix-preds (predict booster dtrain #:as 'f32vector))
  (define inplace-preds
    (predict-from-csr booster csr-indptr csr-indices dense-features 3
                      #:missing -1.0 #:as 'f32vector))
  (hash 'prediction-count (f32vector-length inplace-preds)
        'matches-dmatrix? (f32vector~= inplace-preds dmatrix-preds)))]

The harness @filepath{test/21-inplace-predict-csr.rkt} prints the count and
asserts the CSR in-place predictions match the DMatrix predictions.

@chunk[<*>
  <r21-require>
  <r21-provide>
  <r21-helpers>
  <r21-run>]
