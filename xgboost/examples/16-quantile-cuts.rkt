#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     xgboost))

@section[#:tag "ex-quantile-cuts"]{Quantile cuts}

The @racket["hist"] tree method buckets each feature into quantile bins before
training. @racket[dmatrix-quantile-cut] exposes those cut points after a model
has been built: it returns @racket[(values indptr data)], a CSR-style pair where
@racket[indptr] gives each feature's slice into the flat @racket[data] vector of
cut values. This example trains a one-round @racket["hist"] model and inspects
its cuts.

@chunk[<r16-require>
(require ffi/vector
         xgboost)]

@chunk[<r16-provide>
(provide run-example)]

@chunk[<r16-run>
(define (run-example)
  (define dtrain
    (make-dmatrix (f32vector 1.0 2.0  3.0 4.0  5.0 6.0  7.0 8.0)
                  #:nrow 4 #:ncol 2 #:missing -1.0
                  #:labels (f32vector 1.0 3.0 5.0 7.0)))
  (define booster
    (train dtrain #:objective "reg:squarederror"
           #:params '(("tree_method" . "hist"))
           #:max-depth 2 #:verbosity 0 #:rounds 1))
  (define-values (indptr data) (dmatrix-quantile-cut dtrain))
  (hash 'indptr indptr 'data data
        'indptr-length (length indptr)
        'data-length (f32vector-length data)
        'prediction-count (f32vector-length (predict booster dtrain #:as 'f32vector))))]

The harness @filepath{test/16-quantile-cuts.rkt} prints the cut-vector lengths
and asserts the CSR invariants (@racket[indptr] starts at 0 and its last entry
equals the @racket[data] length).

@chunk[<*>
  <r16-require>
  <r16-provide>
  <r16-run>]
