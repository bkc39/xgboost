#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     xgboost))

@section[#:tag "ex-cuda-classification"]{CUDA classification}

The GPU counterpart of @secref["ex-train-classifier"]: the same two
well-separated clusters and the same high-level @racket[train] /
@racket[predict], with @racket["device=cuda"] passed through @racket[#:params].
As with @secref["ex-cuda-regression"], it needs a CUDA-enabled native library and
a physical GPU, and @racket[cuda-available?] gates the work so it skips on
CPU-only builds.

@chunk[<r25-require>
(require ffi/vector
         xgboost)]

@chunk[<r25-provide>
(provide run-example cuda-available?)]

@chunk[<r25-run>
(define features
  (f32vector 0.1 0.2 0.1 0.0   5.0 4.0 5.5 6.0   0.3 0.5 0.1 0.2   4.5 5.0 4.0 5.5
             0.0 0.1 0.2 0.0   6.0 5.5 6.5 5.0   0.4 0.3 0.2 0.5   5.5 6.0 4.5 5.0
             0.2 0.1 0.3 0.1   4.0 4.5 5.0 4.0))
(define labels (f32vector 0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0))

(define (run-example)
  (define dtrain (make-dmatrix features #:nrow 10 #:ncol 4 #:labels labels))
  (define booster
    (train dtrain
           #:objective "binary:logistic"
           #:params '((device . "cuda") (tree_method . "hist"))
           #:max-depth 3 #:eta 0.3 #:verbosity 0 #:rounds 30))
  (define probs (predict booster dtrain #:as 'f32vector))
  (define n (f32vector-length labels))
  (define correct
    (for/sum ([i (in-range n)])
      (if (= (if (> (f32vector-ref probs i) 0.5) 1 0)
             (inexact->exact (round (f32vector-ref labels i)))) 1 0)))
  (hash 'prediction-count n
        'all-valid-probs? (for/and ([i (in-range n)]) (<= 0.0 (f32vector-ref probs i) 1.0))
        'correct correct 'accuracy (/ correct n) 'perfect? (= correct n)))]

The harness @filepath{test/25-cuda-classification.rkt} runs the example only when
@racket[cuda-available?] is true, printing accuracy and asserting the clusters
separate perfectly; on CPU-only builds it prints a skip notice.

@chunk[<*>
  <r25-require>
  <r25-provide>
  <r25-run>]
