#lang scribble/lp2

@(require (for-label racket/base
                     racket/file
                     xgboost))

@section[#:tag "ex-root-api"]{The high-level API end to end}

A compact smoke test of the everyday @racketmodname[xgboost] surface, using plain
Racket lists for the data: build a @tech{DMatrix}, @racket[train], @racket[predict],
read a metric with @racket[eval-one-iter] / @racket[parse-eval-line], and confirm
a @racket[save-model] / @racket[load-model] round-trip reproduces the predictions.

@chunk[<r13-require>
(require racket/file
         xgboost)]

@chunk[<r13-provide>
(provide run-example)]

@chunk[<r13-run>
(define (run-example)
  (define features '((1.0 2.0 0.5) (2.0 1.0 1.5) (3.0 0.5 0.0) (0.5 3.0 2.0)
                     (4.0 2.0 1.0) (1.5 1.5 0.5) (2.5 3.5 1.5) (0.0 1.0 0.0)))
  (define labels '(3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0))
  (define dtrain (make-dmatrix features #:labels labels))
  (define booster
    (train dtrain #:objective "reg:squarederror" #:eval-metric "rmse"
           #:max-depth 3 #:eta 0.1 #:verbosity 0 #:rounds 20))
  (define preds (predict booster dtrain))
  (define metrics (parse-eval-line (eval-one-iter booster 19 (list (cons "train" dtrain)))))
  (define tmp (make-temporary-file "xgboost-e2e-~a.json"))
  (define loaded-preds
    (dynamic-wind
      void
      (lambda ()
        (save-model booster tmp)
        (predict (load-model tmp) dtrain))
      (lambda () (when (file-exists? tmp) (delete-file tmp)))))
  (hash 'predictions preds 'loaded-predictions loaded-preds 'metrics metrics))]

The harness @filepath{test/13-high-level-root-api.rkt} prints the prediction
count and train RMSE, and asserts the predictions are finite and survive the
save/load round-trip unchanged.

@chunk[<*>
  <r13-require>
  <r13-provide>
  <r13-run>]
