#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     "../../main.rkt"))

@title[#:tag "inspection"]{Inspecting a Model}

The Python package's @italic{Plotting} section renders feature importance and
tree diagrams with matplotlib and graphviz. The Racket bindings don't ship a
plotting layer; instead they expose the underlying data — model dumps and
importance scores — that you can render with whatever tooling you prefer.

@margin-note{@bold{Future work:} a native rendering built on the Racket
@hyperlink["https://docs.racket-lang.org/plot/"]{@tt{plot}} library
(feature-importance bar charts, and possibly tree diagrams) would be a natural
companion to the @racket[booster-dump] / @racket[booster-feature-score] data
shown here.}

@section{Dumping trees}

@racket[booster-dump] returns one string per tree. @racket[#:format] selects
@racket["text"] (the default), @racket["json"], or @racket["dot"]. The
@racket["dot"] form is graphviz source you can pipe straight to @tt{dot} to
draw the tree:

@racketblock[
(require xgboost)

(booster-dump booster)                    (code:comment "list of text trees")
(booster-dump booster #:format "json")
(define dots (booster-dump booster #:format "dot"))
]

Passing @racket[#:feature-names] (and @racket[#:feature-types]) substitutes
readable names into the dump in place of @tt{f0}, @tt{f1}, … :

@racketblock[
(booster-dump booster
              #:format "text"
              #:feature-names '("x0" "x1" "x2")
              #:feature-types '("q" "q" "q"))
]

@section{Feature importance}

@racket[booster-feature-score] computes per-feature importance. Choose the mode
with @racket[#:importance-type] — @racket["weight"] (the default, split counts),
@racket["gain"], @racket["cover"], or their totals. The result is a hash with
@racket['features], @racket['scores] (an @racket[f32vector?]), and
@racket['shape]:

@racketblock[
(define scores
  (booster-feature-score booster
                         #:importance-type "weight"
                         #:feature-names '("x0" "x1" "x2")))

(hash-ref scores 'features)   (code:comment "feature names in score order")
(hash-ref scores 'scores)     (code:comment "f32vector of importances")
]

To set readable feature names once so every dump and score uses them, call
@racket[booster-set-feature-names!] (and @racket[booster-set-feature-types!]) on
the booster after training.

@section{Other inspection}

@racket[booster-num-feature] and @racket[booster-boosted-rounds] report the
model's width and how many rounds it has been trained for.
@racket[booster-config] returns XGBoost's full configuration as a JSON string
(treat it as opaque; round-trip it with @racket[booster-set-config!]), and
@racket[booster-attr] / @racket[booster-set-attr!] read and write arbitrary
string attributes you want to travel with the model.
