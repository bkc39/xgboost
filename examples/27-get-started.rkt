#lang racket/base

;; Get Started: an iris classifier, the Racket counterpart of XGBoost's
;; Python quickstart at https://xgboost.readthedocs.io/en/stable/get_started.html
;;
;; Mirrors the upstream snippet line for line:
;;
;;   from xgboost import XGBClassifier
;;   from sklearn.datasets import load_iris
;;   from sklearn.model_selection import train_test_split
;;   data = load_iris()
;;   X_train, X_test, y_train, y_test = train_test_split(
;;       data['data'], data['target'], test_size=.2)
;;   bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1)
;;   bst.fit(X_train, y_train)
;;   preds = bst.predict(X_test)
;;
;; Differences from the Python: there is no XGBClassifier (we use `train` /
;; `predict` over a DMatrix); `load-iris` and `train-test-split` come from
;; xgboost/private/demo-utils (load-iris downloads the UCI dataset, falling
;; back to a bundled copy offline); and iris has three classes, so we use
;; `multi:softmax` with #:num-class 3 (so `predict` returns class indices, like
;; sklearn's `.predict` returns labels).
;;
;; Run from the repo root:
;;   nix develop --command racket examples/27-get-started.rkt

(require rackunit
         xgboost
         xgboost/private/demo-utils)

(provide run-example)

(define (accuracy preds labels)
  (/ (for/sum ([p (in-list preds)] [y (in-list labels)])
       (if (= (inexact->exact (round p)) y) 1 0))
     (length labels)))

(define (run-example)
  ;; read data
  (define-values (X y) (load-iris))
  (check-equal? (length X) 150)
  (define-values (X-train X-test y-train y-test)
    (train-test-split X y #:test-size 0.2 #:seed 42))

  ;; create model instance and fit
  (define bst
    (train (make-dmatrix X-train #:labels y-train)
           #:num-class 3
           #:objective "multi:softmax"
           #:max-depth 2
           #:eta 1.0
           #:verbosity 0
           #:rounds 2))

  ;; make predictions
  (define preds (predict bst (make-dmatrix X-test)))

  (check-equal? (length preds) (length y-test))
  (define acc (accuracy preds y-test))
  ;; iris is easily separable; even two shallow rounds clear this floor.
  (check-true (> acc 0.8) (format "expected accuracy > 0.8, got ~a" acc))

  (printf "get-started: ~a train / ~a test, test accuracy ~a\n"
          (length X-train) (length X-test)
          (real->decimal-string acc 3))
  (void))

(module+ main
  (run-example))

(module+ test
  (run-example))
