#lang scribble/manual

@(require (for-label ffi/vector
                     racket/base
                     racket/contract
                     "../../main.rkt"))

@title[#:tag "objectives"]{Custom Objectives}

Beyond the built-in objectives named by strings (such as
@racket["reg:squarederror"] or @racket["binary:logistic"]), you can supply your
own loss by writing a Racket function that returns its gradient and Hessian.

@section{The @racket[#:objective-fn] keyword}

Pass @racket[#:objective-fn] to @racket[train]. Each round, @racket[train]
computes the current margin predictions and calls your function with those
predictions and the training DMatrix; it must return @racket[(values grad hess)]
as sequences (lists, vectors, or @racket[f32vector?] values), one entry per row:

@racketblock[
(require xgboost ffi/vector)

(code:comment "Squared error: grad = pred - label, hess = 1.")
(define (squared-error preds dtrain)
  (define ys (dmatrix-label dtrain))
  (define n  (f32vector-length preds))
  (define grad (make-f32vector n))
  (define hess (make-f32vector n 1.0))
  (for ([i (in-range n)]
        [y (in-list ys)])
    (f32vector-set! grad i (- (f32vector-ref preds i) y)))
  (values grad hess))

(define booster
  (train dtrain
         #:objective-fn squared-error
         #:max-depth 3
         #:eta 0.2
         #:verbosity 0
         #:rounds 20))
]

The predictions passed to your function are raw margins, so the gradient and
Hessian should be expressed with respect to the margin — exactly as a built-in
objective would compute them internally.

@section{Stepping a custom objective by hand}

For full control over the loop, use @racket[booster-train-one-iter!], which runs
a single round from gradient and Hessian vectors you supply directly:

@racketblock[
(for ([iter (in-range 20)])
  (define preds (predict booster iter-dtrain #:output 'margin #:as 'f32vector))
  (define-values (grad hess) (squared-error preds iter-dtrain))
  (booster-train-one-iter! booster iter iter-dtrain grad hess))
]

This is the building block @racket[#:objective-fn] uses under the hood; reach
for it when you need to interleave custom bookkeeping between rounds. The
@racketmodname[xgboost] examples include several worked custom objectives
(robust, quantile, Poisson, and AFT survival regression).
