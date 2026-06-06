#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     racket/list
                     xgboost
                     xgboost/private/demo-utils))

@section[#:tag "ex-quantile-regression"]{Quantile regression}

@racket["reg:quantileerror"] with a comma-separated @racket["quantile_alpha"]
fits several quantiles of the target distribution from one model --- here the
10th, 50th, and 90th percentiles. The @tt{(p10, p90)} interval is a
non-parametric uncertainty band: well-calibrated quantiles place about 80% of
true values inside it, and the band widens where the data is noisier. The
prediction output is @tt{nrow × n_quantiles} in row-major order.

@chunk[<r08-require>
(require ffi/vector
         racket/list
         xgboost
         xgboost/private/demo-utils)]

@chunk[<r08-provide>
(provide run-example)]

@bold{Helpers.} The same every-5th-row split and row→DMatrix packer as
@secref["ex-robust-regression"] (diabetes rows are 10 features then @tt{Y}):

@chunk[<r08-helpers>
  (define ncol 10)
  (define (split-rows rows)
    (for/fold ([tr '()] [te '()] #:result (values (reverse tr) (reverse te)))
              ([row (in-list rows)] [i (in-naturals)])
      (if (zero? (modulo i 5)) (values tr (cons row te)) (values (cons row tr) te))))
  (define (rows->dmatrix rs)
    (define n (length rs))
    (define features (make-f32vector (* n ncol)))
    (define labels (make-f32vector n))
    (for ([row (in-list rs)] [i (in-naturals)])
      (for ([v (in-list (take row ncol))] [j (in-naturals)])
        (f32vector-set! features (+ (* i ncol) j) (exact->inexact v)))
      (f32vector-set! labels i (exact->inexact (last row))))
    (make-dmatrix features #:nrow n #:ncol ncol #:labels labels))]

@bold{The run.} Train one model emitting three quantiles. The vector parameter
must be parens-wrapped --- a bare @racket["0.1,0.5,0.9"] is read as a scalar and
silently yields single-quantile output --- and quantile loss requires
@racket["tree_method=hist"]. Then pull row-wise @tt{(p10, p50, p90)} triples and
measure monotonicity and coverage. @racket[run-example] returns the diagnostics:

@chunk[<r08-run>
(define (run-example)
  (define-values (train-rows test-rows) (split-rows (load-diabetes)))
  (define n-q 3)
  (define b
    (train (rows->dmatrix train-rows)
           #:objective "reg:quantileerror"
           #:params (list (cons "quantile_alpha" "(0.1,0.5,0.9)")
                          (cons "tree_method" "hist"))
           #:max-depth 4 #:eta 0.1 #:verbosity 0 #:rounds 100))
  (define preds (predict b (rows->dmatrix test-rows) #:as 'f32vector))
  (define n-test (length test-rows))
  (define rows-with-q
    (for/list ([i (in-range n-test)] [row (in-list test-rows)])
      (define base (* i n-q))
      (list (last row)
            (f32vector-ref preds (+ base 0))
            (f32vector-ref preds (+ base 1))
            (f32vector-ref preds (+ base 2)))))
  (define crossings
    (for/sum ([qs (in-list rows-with-q)])
      (if (or (> (cadr qs) (caddr qs)) (> (caddr qs) (cadddr qs))) 1 0)))
  (define coverage-80
    (for/sum ([qs (in-list rows-with-q)])
      (if (and (<= (cadr qs) (car qs)) (<= (car qs) (cadddr qs))) 1 0)))
  (define mean-band
    (/ (for/sum ([qs (in-list rows-with-q)]) (- (cadddr qs) (cadr qs))) n-test))
  (hash 'n-test n-test 'n-q n-q 'pred-len (f32vector-length preds)
        'crossings crossings 'coverage-80 coverage-80
        'mean-band mean-band 'rows-with-q rows-with-q))]

The harness @filepath{test/08-quantile-regression.rkt} prints the monotonicity
and coverage diagnostics and a sample of predicted bands, and asserts the
multi-quantile output shape and that the quantiles are well-ordered and
reasonably calibrated.

@chunk[<*>
  <r08-require>
  <r08-provide>
  <r08-helpers>
  <r08-run>]
