#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     "../../main.rkt"))

@title[#:tag "recipes"]{Parameter Recipes}

Several XGBoost features that have their own tutorials upstream are, from the
API's point of view, just parameter settings on an otherwise ordinary
@racket[train] call. This chapter collects them as short recipes. Each passes
its settings through @racket[#:params] (see @secref["parameters"]); the
@secref["ex-param-recipes"] example walks all four as a runnable,
assertion-backed program.

@section{DART booster}

@hyperlink["https://xgboost.readthedocs.io/en/stable/tutorials/dart.html"]{DART}
drops a random subset of existing trees on each boosting round to regularize
the ensemble. Select it with @racket["booster"] @racket["dart"] and tune the
drop rates:

@racketblock[
(train dtrain
       #:params '((booster   . "dart")
                  (objective . "reg:squarederror")
                  (rate_drop . 0.1)
                  (skip_drop . 0.5))
       #:max-depth 3 #:eta 0.1 #:rounds 30)
]

Prediction works exactly as for the default @racket["gbtree"] booster.

@section{Monotonic constraints}

@hyperlink["https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html"]{Monotonic
constraints} force the model's response to be non-decreasing (@racket[1]) or
non-increasing (@racket[-1]) in chosen features, with @racket[0] for
unconstrained ones. The constraint vector has one entry per feature:

@racketblock[
(code:comment "non-decreasing in feature 0, unconstrained in features 1 and 2")
(train dtrain
       #:params '((objective            . "reg:squarederror")
                  (monotone_constraints . "(1,0,0)"))
       #:max-depth 3 #:eta 0.1 #:rounds 30)
]

The constraint is a hard guarantee: holding the other features fixed and
sweeping feature 0 upward, the prediction never decreases. The example's test
verifies exactly this property.

@section{Feature interaction constraints}

@hyperlink["https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html"]{Interaction
constraints} restrict which features may appear together on a single root-to-leaf
path. Pass groups as a JSON list of lists:

@racketblock[
(code:comment "features 0 and 1 may interact; feature 2 stays on its own")
(train dtrain
       #:params '((objective                . "reg:squarederror")
                  (interaction_constraints . "[[0,1],[2]]"))
       #:max-depth 3 #:eta 0.1 #:rounds 30)
]

@section{Random forests}

A @hyperlink["https://xgboost.readthedocs.io/en/stable/tutorials/rf.html"]{random
forest} is the degenerate case of boosting: a single round
(@racket[#:rounds 1]) that grows many parallel, subsampled trees. Set
@racket["num_parallel_tree"] and the subsampling ratios, and use a full-size
learning rate:

@racketblock[
(train dtrain
       #:params '((objective         . "reg:squarederror")
                  (num_parallel_tree . 20)
                  (subsample         . 0.8)
                  (colsample_bynode  . 0.8))
       #:max-depth 4 #:eta 1.0 #:rounds 1)
]

You can also combine a forest of @racket["num_parallel_tree"] trees with several
boosting rounds to get a boosted forest.
