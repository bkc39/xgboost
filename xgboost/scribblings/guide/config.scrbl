#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     "../../main.rkt"))

@title[#:tag "config"]{Global Configuration, Snapshots, and GPU}

@section{Version and build information}

@racket[xgboost-version] returns the linked XGBoost version string, and
@racket[xgboost-build-info] returns the build configuration as JSON (compiler
flags, CUDA support, and so on):

@racketblock[
(require xgboost)

(xgboost-version)      (code:comment "e.g. \"2.1.0\"")
(xgboost-build-info)   (code:comment "JSON string")
]

@section{Process-global configuration}

XGBoost keeps some settings at process scope. Read and write them as JSON with
@racket[xgboost-get-global-config] and @racket[xgboost-set-global-config!]. A
common pattern is to save, change, then restore around a region of code:

@racketblock[
(define saved (xgboost-get-global-config))
(dynamic-wind
  void
  (lambda ()
    (xgboost-set-global-config! "{\"verbosity\":0}")
    (code:comment "... quiet work ...")
    (void))
  (lambda ()
    (xgboost-set-global-config! saved)))
]

@racket[xgboost-register-log-callback!] installs a process-global callback that
receives XGBoost's log messages as strings. Because it is process-global, treat
it as shared mutable state.

@section{Full-state snapshots}

@racket[save-model] persists only the trained trees. A @deftech{snapshot}
additionally captures XGBoost's internal training caches, so a restored booster
can resume per-iteration updates in lockstep with the original.
@racket[booster->bytes] serializes that full state and @racket[bytes->booster]
reconstructs it:

@racketblock[
(train-rounds! booster dtrain 0 5)            (code:comment "train 5 rounds")

(define snapshot (booster->bytes booster))
(define restored (bytes->booster snapshot))

(code:comment "both continue identically from round 5")
(train-rounds! booster  dtrain 5 5)
(train-rounds! restored dtrain 5 5)
]

A booster restored from a snapshot has an empty DMatrix cache, so pass the
training data explicitly when you resume with @racket[booster-update-one-iter!].

@section{GPU training}

When the native library is built with CUDA, training runs on the GPU by setting
@racket["device"] to @racket["cuda"] (with @racket["tree_method"] @racket["hist"]).
Check availability first with @racket[cuda-available?] so code degrades
gracefully on CPU-only builds:

@racketblock[
(when (cuda-available?)
  (train dtrain
         #:params '((device      . "cuda")
                    (tree-method . "hist"))
         #:objective "reg:squarederror"
         #:rounds 50))
]

Whether @racket[cuda-available?] is @racket[#t] depends on the native build:
the package prefers a CUDA-enabled library on Linux when one is present and
falls back to the CPU build otherwise.
