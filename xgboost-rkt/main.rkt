#lang racket/base

(require "private/xgboost-native.rkt")

(provide xgboost-version
         run-regression-demo
         run-classification-demo
         dmatrix?
         dmatrix-create-from-mat
         dmatrix-set-float-info!
         dmatrix-free!
         dmatrix-nrow
         dmatrix-ncol
         dmatrix-get-float-info
         dmatrix-num-non-missing
         dmatrix->list
         dmatrix-show
         booster?
         booster-create
         booster-free!
         booster-set-param!
         booster-update-one-iter!
         booster-predict
         booster-save-model!
         booster-load-model!
         booster-save-model-to-bytes
         booster-load-model-from-bytes!
         booster-eval-one-iter
         parse-eval-line)

(module+ main
  (printf "xgboost version: ~a\n" (xgboost-version))
  (printf "regression first prediction: ~a\n" (run-regression-demo))
  (printf "classification first prediction: ~a\n" (run-classification-demo)))

(module+ test
  (require rackunit)
  (check-regexp-match #rx"^[0-9]+\\.[0-9]+\\.[0-9]+$" (xgboost-version))
  (check-pred rational? (run-regression-demo))
  (define p (run-classification-demo))
  (check-true (<= 0 p 1)))
