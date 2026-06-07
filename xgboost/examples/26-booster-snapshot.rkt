#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     xgboost))

@section[#:tag "ex-booster-snapshot"]{Booster snapshots}

@racket[booster->bytes] / @racket[bytes->booster] capture a booster's
@emph{full state} --- including XGBoost's internal training caches, not just the
fitted trees. That means a restored handle can resume per-iteration updates in
lockstep with the original, which @racket[save-model] (model-only) cannot
guarantee. This example trains five rounds, snapshots, restores into a fresh
handle, then trains five more on both and checks they stay identical.

@chunk[<r26-require>
(require ffi/vector
         xgboost)]

@chunk[<r26-provide>
(provide run-example)]

@bold{Helper.} Advance a booster a few rounds by hand:

@chunk[<r26-helper>
  (define features
    (f32vector 1.0 2.0 0.5   2.0 1.0 1.5   3.0 0.5 0.0   0.5 3.0 2.0
               4.0 2.0 1.0   1.5 1.5 0.5   2.5 3.5 1.5   0.0 1.0 0.0))
  (define labels (f32vector 3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0))
  (define (train-rounds! b dtrain start rounds)
    (for ([iter (in-range start (+ start rounds))])
      (booster-update-one-iter! b iter dtrain)))]

@bold{The run.} Snapshot after five rounds, restore, and confirm predictions
match immediately and again after both continue training. @racket[run-example]
returns the snapshot size and the two match flags:

@chunk[<r26-run>
(define (run-example)
  (define dtrain
    (make-dmatrix features #:nrow 8 #:ncol 3 #:missing -1.0 #:labels labels))
  (define b
    (train dtrain #:objective "reg:squarederror"
           #:max-depth 3 #:eta 0.1 #:verbosity 0 #:rounds 0))
  (train-rounds! b dtrain 0 5)
  (define snapshot (booster->bytes b))
  (define snapshot-preds (predict b dtrain #:as 'f32vector))
  (define restored (bytes->booster snapshot))
  (define restored-preds (predict restored dtrain #:as 'f32vector))
  ;; Continue both from round 5; they must stay in lockstep.
  (train-rounds! b dtrain 5 5)
  (train-rounds! restored dtrain 5 5)
  (hash 'snapshot-bytes (bytes-length snapshot)
        'matches-immediately?
        (equal? (f32vector->list snapshot-preds) (f32vector->list restored-preds))
        'matches-after-resume?
        (equal? (f32vector->list (predict b dtrain #:as 'f32vector))
                (f32vector->list (predict restored dtrain #:as 'f32vector)))))]

The harness @filepath{test/26-booster-snapshot.rkt} prints the snapshot size and
match flags, and asserts the restored booster tracks the original exactly.

@chunk[<*>
  <r26-require>
  <r26-provide>
  <r26-helper>
  <r26-run>]
