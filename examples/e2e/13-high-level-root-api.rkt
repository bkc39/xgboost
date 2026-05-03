#lang racket/base

(require racket/file
         racket/list
         xgboost)

(provide run-example)

(define features
  '((1.0 2.0 0.5)
    (2.0 1.0 1.5)
    (3.0 0.5 0.0)
    (0.5 3.0 2.0)
    (4.0 2.0 1.0)
    (1.5 1.5 0.5)
    (2.5 3.5 1.5)
    (0.0 1.0 0.0)))

(define labels '(3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0))

(define (finite-real? v)
  (and (real? v) (= v v) (not (eqv? v +inf.0)) (not (eqv? v -inf.0))))

(define (run-example)
  (define dtrain (make-dmatrix features #:labels labels))
  (define booster
    (train dtrain
           #:objective "reg:squarederror"
           #:eval-metric "rmse"
           #:max-depth 3
           #:eta 0.1
           #:verbosity 0
           #:rounds 20))
  (define preds (predict booster dtrain))
  (define eval-line (eval-one-iter booster 19 (list (cons "train" dtrain))))
  (define metrics (parse-eval-line eval-line))

  (define tmp (make-temporary-file "xgboost-e2e-~a.json"))
  (define loaded-preds
    (dynamic-wind
      void
      (lambda ()
        (save-model booster tmp)
        (define loaded (load-model tmp))
        (predict loaded dtrain))
      (lambda ()
        (when (file-exists? tmp)
          (delete-file tmp)))))

  (hash 'predictions preds
        'loaded-predictions loaded-preds
        'eval-line eval-line
        'metrics metrics))

(module+ main
  (define result (run-example))
  (printf "predictions: ~a\n" (length (hash-ref result 'predictions)))
  (printf "train-rmse: ~a\n"
          (hash-ref (hash-ref result 'metrics) "train-rmse"))
  (printf "save/load predictions equal: ~a\n"
          (equal? (hash-ref result 'predictions)
                  (hash-ref result 'loaded-predictions))))

(module+ test
  (require rackunit)

  (define result (run-example))
  (define preds (hash-ref result 'predictions))
  (define loaded-preds (hash-ref result 'loaded-predictions))
  (define metrics (hash-ref result 'metrics))

  (check-equal? (length preds) (length labels))
  (check-true (andmap finite-real? preds))
  (check-true (hash-has-key? metrics "train-rmse"))
  (check-true (finite-real? (hash-ref metrics "train-rmse")))
  (check-equal? loaded-preds preds))
