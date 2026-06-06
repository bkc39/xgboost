#lang scribble/manual

@(require scribble/lp-include
          (for-label racket/base
                     xgboost))

@title[#:tag "examples" #:style 'toc]{Examples}

Each example below is a @deftech{literate program}: the prose and the code you
see are the @emph{same source} that lives in the package's
@filepath{examples/} directory and is exercised by the test suite, so the
walkthroughs never drift from working code. Every example provides a
@racket[run-example] thunk; its companion harness in @filepath{examples/test/}
drives it and checks the result (run with @exec{raco test}).

The examples build up in arcs. The @bold{core tasks} start from the
@tech{DMatrix} container and the high-level @racket[train]/@racket[predict]
loop. Later arcs --- custom objectives, DMatrix mechanics, booster inspection,
serving-style in-place prediction, and parameter recipes --- layer on from
there. (More arcs are converted to this literate form incrementally; until then
see @filepath{examples/README.md} for the full index.)

@local-table-of-contents[]

@; core tasks
@lp-include["../../examples/00-print-dmatrix.rkt"]
@lp-include["../../examples/01-train-regression.rkt"]
@lp-include["../../examples/02-train-classifier.rkt"]
@lp-include["../../examples/04-train-multiclass.rkt"]
@lp-include["../../examples/05-train-with-eval.rkt"]
@lp-include["../../examples/06-iris.rkt"]
@lp-include["../../examples/27-get-started.rkt"]
