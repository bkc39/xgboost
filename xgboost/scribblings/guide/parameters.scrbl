#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     "../../main.rkt"))

@title[#:tag "parameters"]{Setting Parameters}

XGBoost is configured by a set of string-keyed parameters — the objective, tree
depth, learning rate, and so on. The Racket API lets you pass them in two
complementary ways.

@section{The @racket[#:params] bundle}

@racket[#:params] takes a hash or association list of parameters. Keys may be
strings, symbols, or keywords; symbol and keyword keys are normalized to
XGBoost's underscore style (hyphens become underscores), and values are
converted to strings before reaching the native layer:

@racketblock[
(train dtrain
       #:params '((objective  . "reg:squarederror")
                  (max-depth  . 3)
                  (eta        . 0.1)
                  (tree-method . "hist")))
]

Here @racket['max-depth] reaches XGBoost as @racket["max_depth"], and the
numeric @racket[3] becomes @racket["3"].

@section{Keyword conveniences}

The most common parameters also have dedicated keywords on @racket[train]:
@racket[#:objective], @racket[#:eta], @racket[#:max-depth], @racket[#:num-class],
@racket[#:eval-metric], and @racket[#:verbosity]. They are applied @emph{after}
@racket[#:params], so a keyword overrides any same-named entry in the bundle:

@racketblock[
(train dtrain
       #:objective "binary:logistic"
       #:max-depth 3
       #:eta 0.3
       #:verbosity 0
       #:rounds 30)
]

@section{Evaluation sets}

To watch performance on held-out data during training, pass @racket[#:evals] —
a list of @racket[(cons name dmatrix)] pairs. XGBoost reports each named metric
per round, and the names you choose appear in the metric lines:

@racketblock[
(train dtrain
       #:evals (list (cons "train" dtrain)
                     (cons "eval"  deval))
       #:objective "reg:squarederror"
       #:eval-metric "rmse"
       #:rounds 30)
]

Driving the evaluation loop yourself — to log metrics or stop early — is covered
in @secref["training"].

@section{Updating a booster's parameters}

A parameter can also be set on an existing booster with
@racket[booster-set-param!], which uses the same key/value coercion as
@racket[#:params]:

@racketblock[
(booster-set-param! booster "tree_method" "hist")
]
