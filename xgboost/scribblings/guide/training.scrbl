#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     "../../main.rkt"))

@title[#:tag "training"]{Training}

With a DMatrix and parameters in hand, @racket[train] fits a booster. The
@racket[#:rounds] keyword sets the number of boosting iterations; the result is
an opaque booster value:

@racketblock[
(require xgboost)

(define dtrain
  (make-dmatrix features #:nrow 8 #:ncol 3 #:labels labels))

(define booster
  (train dtrain
         #:objective "reg:squarederror"
         #:max-depth 3
         #:eta 0.1
         #:verbosity 0
         #:rounds 50))
]

The returned booster keeps the training DMatrix (and any @racket[#:evals]
DMatrices) reachable, so their native buffers stay alive for as long as the
booster does.

@section{Saving and loading models}

@racket[save-model] writes the trained tree ensemble to a file; XGBoost picks
the format from the extension (use @filepath{.json} or @filepath{.ubj}).
@racket[load-model] reads it back into a fresh booster:

@racketblock[
(save-model booster "model.json")
(define reloaded (load-model "model.json"))
]

To keep the model in memory instead of on disk, serialize to bytes.
@racket[save-model-to-bytes] defaults to compact UBJSON; pass
@racket[#:format "json"] for the textual form. @racket[load-model-from-bytes]
reverses either:

@racketblock[
(define blob (save-model-to-bytes booster))           (code:comment "UBJSON")
(define json (save-model-to-bytes booster #:format "json"))
(define restored (load-model-from-bytes blob))
]

A model loaded by any of these routes produces the same predictions as the
original booster.

@margin-note{@racket[save-model] persists only the trained trees. To checkpoint
mid-training and resume per-iteration updates in lockstep, use the full-state
snapshots described in @secref["config"].}

@section[#:tag "early-stopping"]{Early stopping}

XGBoost's Python package offers a built-in early-stopping callback. The Racket
API has no callback mechanism; instead you drive the boosting loop yourself and
decide when to stop. Build an untrained booster with @racket[#:rounds 0] (which
still binds @racket[dtrain] and the @racket[#:evals] DMatrices into its cache),
then step it one round at a time with @racket[booster-update-one-iter!]. After
each round, @racket[eval-one-iter] returns XGBoost's metric line, which
@racket[parse-eval-line] turns into a hash:

@racketblock[
(define booster
  (train dtrain
         #:evals (list (cons "eval" deval))
         #:objective "reg:squarederror"
         #:max-depth 3
         #:eta 0.1
         #:verbosity 0
         #:rounds 0))

(define eval-set (list (cons "train" dtrain) (cons "eval" deval)))

(for ([iter (in-range 30)])
  (booster-update-one-iter! booster iter dtrain)
  (define metrics (parse-eval-line (eval-one-iter booster iter eval-set)))
  (printf "round ~a  eval-rmse ~a\n"
          iter (hash-ref metrics "eval-rmse")))
]

Because you own the loop, early stopping is just ordinary Racket control flow:
track the best metric so far and break out (or stop updating) once it fails to
improve for a chosen number of rounds. The booster after @racket[N] calls to
@racket[booster-update-one-iter!] is exactly an @racket[N]-round model, and
@racket[booster-slice] can later extract the best-iteration prefix if you
trained past it.
