#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     xgboost))

@section[#:tag "ex-train-with-eval"]{Watching an evaluation set}

Rather than letting @racket[train] run the whole boosting loop, you can drive it
one round at a time and inspect a held-out metric after each --- the basis for
logging, early stopping, or custom schedules. This example trains a regressor
while watching RMSE on both the training and an evaluation split.

The two pieces are @racket[booster-update-one-iter!] (do one boosting round) and
@racket[eval-one-iter] (return XGBoost's metric line, which
@racket[parse-eval-line] turns into a hash).

@chunk[<r05-require>
(require ffi/vector
         xgboost)]

@chunk[<r05-provide>
(provide run-example)]

@bold{The data.} Two non-overlapping splits of a @tt{y ≈ 2·x₀ + x₁ − x₂}
dataset --- eight training rows, four evaluation rows:

@chunk[<r05-data>
  (define dtrain
    (make-dmatrix (f32vector 1.0 2.0 0.5   2.0 1.0 1.5   3.0 0.5 0.0
                             0.5 3.0 2.0   4.0 2.0 1.0   1.5 1.5 0.5
                             2.5 3.5 1.5   0.0 1.0 0.0)
                  #:nrow 8 #:ncol 3
                  #:labels (f32vector 3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0)))
  (define deval
    (make-dmatrix (f32vector 2.0 0.5 0.5   1.0 1.0 1.0
                             3.5 1.0 0.5   0.5 0.5 0.5)
                  #:nrow 4 #:ncol 3
                  #:labels (f32vector 4.0 2.0 7.5 1.0)))]

@bold{Set up the booster.} Training with @racket[#:rounds 0] and an
@racket[#:evals] list builds the booster and binds both matrices into its cache
(so the GC keeps them alive) without doing any boosting yet:

@chunk[<r05-setup>
  (define booster
    (train dtrain
           #:evals (list (cons "eval" deval))
           #:objective "reg:squarederror"
           #:max-depth 3
           #:eta 0.1
           #:verbosity 0
           #:rounds 0))]

@bold{The loop.} Each round, advance the booster and record the parsed metrics
for both watched matrices. @racket[run-example] returns the booster and the
per-round history:

@chunk[<r05-loop>
  (define eval-set (list (cons "train" dtrain) (cons "eval" deval)))
  (define history
    (for/list ([iter (in-range 30)])
      (booster-update-one-iter! booster iter dtrain)
      (parse-eval-line (eval-one-iter booster iter eval-set))))]

The harness @filepath{test/05-train-with-eval.rkt} prints the per-round table
and the final metrics, and asserts the evaluation RMSE falls over training:

@racketblock[
(code:comment "iter  train-rmse   eval-rmse")
(code:comment "   0      3.8019      3.6960")
(code:comment "  29      0.0530      0.3327")
]

@chunk[<r05-run>
(define (run-example)
  <r05-data>
  <r05-setup>
  <r05-loop>
  (values booster history))]

@chunk[<*>
  <r05-require>
  <r05-provide>
  <r05-run>]
