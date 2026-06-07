#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     xgboost
                     xgboost/private/demo-utils))

@section[#:tag "ex-aft-survival"]{Survival analysis (AFT)}

Accelerated Failure Time regression (@racket["survival:aft"]) handles
@deftech{right-censored} survival times --- observations where we know a subject
was still alive at time @tt{t} but follow-up ended before death. The label is an
@emph{interval}, not a number: @tt{[t, t]} for an observed death and
@tt{[t, +inf]} for a censored row. XGBoost reads these from the
@racket[dmatrix-set-label-lower-bound!] / @racket[dmatrix-set-label-upper-bound!]
info fields --- no new API; just two calls on the DMatrix. This example fits the
Veterans' lung-cancer data and watches @racket["aft-nloglik"].

@chunk[<r10-require>
(require ffi/vector
         xgboost
         xgboost/private/demo-utils)]

@chunk[<r10-provide>
(provide run-example)]

@bold{Encoding.} Each raw row becomes numeric features (five numbers plus a
4-way one-hot of @tt{celltype}) and an AFT label interval --- @tt{upper = +inf}
marks a censored row:

@chunk[<r10-encode>
  (define celltype-vocab '("squamous" "smallcell" "adeno" "large"))
  (struct ex (features lower upper) #:transparent)
  (define (encode-row r)
    ;; cols: 0 rownames 1 trt 2 celltype 3 time 4 status 5 karno 6 diagtime 7 age 8 prior
    (define time   (string->number (list-ref r 3)))
    (define status (string->number (list-ref r 4)))
    (define one-hot
      (for/list ([c (in-list celltype-vocab)]) (if (equal? c (list-ref r 2)) 1 0)))
    (define features
      (append (map (lambda (i) (string->number (list-ref r i))) '(1 5 6 7 8)) one-hot))
    (ex features (exact->inexact time) (if (= status 1) (exact->inexact time) +inf.0)))]

@bold{Building DMatrices.} Pack features row-major and attach the censoring
bounds via the two info fields:

@chunk[<r10-build>
  (define ncol 9)
  (define (split-rows ds)
    (for/fold ([tr '()] [te '()] #:result (values (reverse tr) (reverse te)))
              ([x (in-list ds)] [i (in-naturals)])
      (if (zero? (modulo i 5)) (values tr (cons x te)) (values (cons x tr) te))))
  (define (dataset->dmatrix ds)
    (define n (length ds))
    (define features (make-f32vector (* n ncol)))
    (define lower (make-f32vector n))
    (define upper (make-f32vector n))
    (for ([x (in-list ds)] [i (in-naturals)])
      (for ([v (in-list (ex-features x))] [j (in-naturals)])
        (f32vector-set! features (+ (* i ncol) j) (exact->inexact v)))
      (f32vector-set! lower i (ex-lower x))
      (f32vector-set! upper i (ex-upper x)))
    (define dm (make-dmatrix features #:nrow n #:ncol ncol))
    (dmatrix-set-label-lower-bound! dm lower)
    (dmatrix-set-label-upper-bound! dm upper)
    dm)]

@bold{The run.} Train with a normal AFT loss, iterating by hand to collect the
per-round @racket["aft-nloglik"] on both splits, then predict expected survival
times on the held-out set. @racket[run-example] returns the history, sanity
counts, and the per-row predictions:

@chunk[<r10-run>
(define (run-example)
  (define dataset (map encode-row (load-veteran)))
  (define-values (train-set test-set) (split-rows dataset))
  (define dtrain (dataset->dmatrix train-set))
  (define dtest (dataset->dmatrix test-set))
  (define b
    (train dtrain #:evals (list (cons "test" dtest))
           #:objective "survival:aft" #:eval-metric "aft-nloglik"
           #:params '(("aft_loss_distribution" . "normal")
                      ("aft_loss_distribution_scale" . "1.20")
                      ("tree_method" . "hist"))
           #:max-depth 3 #:eta 0.05 #:verbosity 0 #:rounds 0))
  (define eval-set (list (cons "train" dtrain) (cons "test" dtest)))
  (define history
    (for/list ([iter (in-range 100)])
      (booster-update-one-iter! b iter dtrain)
      (parse-eval-line (eval-one-iter b iter eval-set))))
  (define preds (predict b dtest #:as 'f32vector))
  (define n-test (length test-set))
  (define rows
    (for/list ([x (in-list test-set)] [i (in-range n-test)])
      (define p (f32vector-ref preds i))
      (list (ex-lower x) (ex-upper x) (equal? (ex-upper x) +inf.0) p)))
  (hash 'n-train (length train-set) 'n-test n-test
        'n-censored-train (for/sum ([x (in-list train-set)])
                            (if (equal? (ex-upper x) +inf.0) 1 0))
        'history history
        'n-positive (for/sum ([r (in-list rows)]) (if (positive? (cadddr r)) 1 0))
        'n-finite (for/sum ([r (in-list rows)])
                    (define p (cadddr r)) (if (and (not (equal? p +inf.0)) (= p p)) 1 0))
        'rows rows))]

The harness @filepath{test/10-aft-survival.rkt} prints the training log and the
held-out predictions, and asserts the loss falls and every prediction is a
finite, positive survival time.

@chunk[<*>
  <r10-require>
  <r10-provide>
  <r10-encode>
  <r10-build>
  <r10-run>]
