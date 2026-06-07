#lang scribble/manual

@(require scribble/lp-include
          (for-label racket/base
                     xgboost))

@title[#:tag "examples" #:style 'toc]{Examples}

Each example below is a @deftech{literate program}: the prose and the code you
see are the @emph{same source} that lives in the package's
@filepath{xgboost/examples/} directory and is exercised by the test suite, so the
walkthroughs never drift from working code. Every example provides a
@racket[run-example] thunk; its companion harness in @filepath{xgboost/examples/test/}
drives it and checks the result (run with @exec{raco test}).

The examples build up in arcs. The @bold{core tasks} start from the
@tech{DMatrix} container and the high-level @racket[train]/@racket[predict]
loop. Later arcs --- built-in and custom objectives, model IO, DMatrix
mechanics, booster inspection, serving-style in-place prediction, parameter
recipes and ranking, the process-global APIs, and GPU training --- layer on from
there.

@local-table-of-contents[]

@; core tasks
@lp-include["../examples/00-print-dmatrix.rkt"]
@lp-include["../examples/01-train-regression.rkt"]
@lp-include["../examples/02-train-classifier.rkt"]
@lp-include["../examples/04-train-multiclass.rkt"]
@lp-include["../examples/05-train-with-eval.rkt"]
@lp-include["../examples/06-iris.rkt"]
@lp-include["../examples/27-get-started.rkt"]

@; built-in and custom objectives
@lp-include["../examples/07-robust-regression.rkt"]
@lp-include["../examples/08-quantile-regression.rkt"]
@lp-include["../examples/09-poisson-bikes.rkt"]
@lp-include["../examples/10-aft-survival.rkt"]
@lp-include["../examples/23-custom-objective.rkt"]

@; model IO and persistence
@lp-include["../examples/03-save-load.rkt"]
@lp-include["../examples/26-booster-snapshot.rkt"]

@; DMatrix construction and metadata
@lp-include["../examples/12-dmatrix-constructors.rkt"]
@lp-include["../examples/14-dmatrix-metadata.rkt"]
@lp-include["../examples/15-dmatrix-slicing-binary.rkt"]
@lp-include["../examples/16-quantile-cuts.rkt"]

@; booster inspection and lifecycle
@lp-include["../examples/13-high-level-root-api.rkt"]
@lp-include["../examples/17-booster-lifecycle-config.rkt"]
@lp-include["../examples/18-booster-attrs.rkt"]
@lp-include["../examples/19-booster-dumps-feature-scores.rkt"]

@; serving: in-place prediction
@lp-include["../examples/20-inplace-predict-dense.rkt"]
@lp-include["../examples/21-inplace-predict-csr.rkt"]
@lp-include["../examples/22-inplace-predict-columnar.rkt"]

@; parameter recipes and ranking
@lp-include["../examples/28-param-recipes.rkt"]
@lp-include["../examples/29-learning-to-rank.rkt"]

@; global / process APIs
@lp-include["../examples/11-global-apis.rkt"]

@; GPU (requires an NVIDIA GPU and a CUDA-enabled native build)
@lp-include["../examples/24-cuda-regression.rkt"]
@lp-include["../examples/25-cuda-classification.rkt"]
