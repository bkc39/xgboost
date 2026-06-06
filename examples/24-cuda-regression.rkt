#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     xgboost))

@section[#:tag "ex-cuda-regression"]{CUDA regression}

This is @secref["ex-train-regression"] moved onto the GPU: the same synthetic
data and the same high-level @racket[train] / @racket[predict], but with
@racket["device=cuda"] and @racket["tree_method=hist"] passed through
@racket[#:params] so tree construction runs on an NVIDIA GPU. Running it needs a
CUDA-enabled native library (@exec{./scripts/build-so.sh linux-cuda} or
@exec{nix build .#cpp-cuda}) and a physical GPU; @racket[cuda-available?] gates
the work so the example skips gracefully on CPU-only builds.

@chunk[<r24-require>
(require ffi/vector
         xgboost)]

@chunk[<r24-provide>
(provide run-example cuda-available?)]

@bold{The run.} The only difference from the CPU regressor is the
@racket[#:params] device/tree-method pair. @racket[run-example] returns the
prediction count and training MSE:

@chunk[<r24-run>
(define features
  (f32vector 1.0 2.0 0.5   2.0 1.0 1.5   3.0 0.5 0.0   0.5 3.0 2.0
             4.0 2.0 1.0   1.5 1.5 0.5   2.5 3.5 1.5   0.0 1.0 0.0))
(define labels (f32vector 3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0))

(define (run-example)
  (define dtrain (make-dmatrix features #:nrow 8 #:ncol 3 #:labels labels))
  (define booster
    (train dtrain
           #:objective "reg:squarederror"
           #:params '((device . "cuda") (tree_method . "hist"))
           #:max-depth 3 #:eta 0.1 #:verbosity 0 #:rounds 50))
  (define preds (predict booster dtrain #:as 'f32vector))
  (define n (f32vector-length labels))
  (define mse
    (/ (for/sum ([i (in-range n)])
         (expt (- (f32vector-ref preds i) (f32vector-ref labels i)) 2)) n))
  (hash 'prediction-count n 'mse mse 'improved? (< mse 1.0)))]

The harness @filepath{test/24-cuda-regression.rkt} runs the example only when
@racket[cuda-available?] is true, printing the MSE and asserting it drops below
@racket[1.0]; on CPU-only builds it prints a skip notice.

@chunk[<*>
  <r24-require>
  <r24-provide>
  <r24-run>]
