#lang racket/base

;; The Racket analogue of XGBoost's "Get Started" quickstart:
;;
;;   https://xgboost.readthedocs.io/en/stable/get_started.html
;;
;; Mirrors the booster-level flow shown there (and in the Python Package
;; Introduction): load LIBSVM-format data into a DMatrix, set a small
;; parameter dict, train for a couple of rounds, predict, then save and
;; reload the model.  The Racket binding has no scikit-style estimator, so
;; this DMatrix + `train` path is the direct 1:1 mapping of the upstream
;; `xgb.train` example.
;;
;; Run from the repo root:
;;   nix develop --command racket examples/27-get-started.rkt

(require racket/file
         racket/string
         rackunit
         xgboost)

(provide run-example)

;; Build a small, linearly-separable binary problem in LIBSVM text format:
;;   <label> <feature>:<value> ...
;; Class 1 rows carry a strong positive feature 0; class 0 rows carry a
;; strong feature 1 instead.  We synthesize enough rows that ordinary
;; boosting splits clear the default min_child_weight (a handful of rows
;; would not), the same reason the upstream example uses a real dataset.
(define (libsvm-dataset n)
  (string-join
   (for/list ([i (in-range n)])
     (define class1? (even? i))
     ;; deterministic spread so the two classes don't perfectly overlap
     (define jitter (/ (modulo i 5) 50.0))
     (if class1?
         (format "1 0:~a 1:~a" (+ 0.8 jitter) (* 0.2 jitter))
         (format "0 0:~a 1:~a" (* 0.2 jitter) (+ 0.8 jitter))))
   "\n"
   #:after-last "\n"))

(define (run-example)
  ;; Stage the LIBSVM files on disk, the way the upstream example reads
  ;; `train.svm.txt` / `test.svm.txt`.
  (define train-path (make-temporary-file "xgb-train-~a.svm.txt"))
  (define test-path  (make-temporary-file "xgb-test-~a.svm.txt"))
  (dynamic-wind
    void
    (lambda ()
      (display-to-file (libsvm-dataset 60) train-path #:exists 'truncate)
      (display-to-file "1 0:1.0 1:0.1\n0 0:0.1 1:1.0\n" test-path #:exists 'truncate)

      ;; 1. Load data into DMatrices.  (Python: xgb.DMatrix('train.svm.txt'))
      (define dtrain (make-dmatrix-from-uri train-path #:format "libsvm"))
      (define dtest  (make-dmatrix-from-uri test-path  #:format "libsvm"))

      ;; 2. Set parameters + 3. train for a few rounds.
      ;;    (Python: param = {...}; bst = xgb.train(param, dtrain, num_round))
      (define bst
        (train dtrain
               #:params '((max_depth . 2)
                          (eta       . 1.0)
                          (objective . "binary:logistic"))
               #:verbosity 0
               #:rounds 4))

      ;; 4. Predict.  (Python: preds = bst.predict(dtest))
      (define preds (predict bst dtest))
      (check-equal? (length preds) 2)
      ;; Class-1 probability should be high for the first row, low for the second.
      (check-true (> (car preds) 0.5) "row 0 predicted class 1")
      (check-true (< (cadr preds) 0.5) "row 1 predicted class 0")

      ;; 5. Save and reload the model; predictions must match.
      ;;    (Python: bst.save_model('model.json'); ... load_model(...))
      (define model-path (make-temporary-file "xgb-model-~a.json"))
      (save-model bst model-path)
      (define reloaded (load-model model-path))
      (check-equal? (predict reloaded dtest) preds "reloaded model matches")
      (delete-file model-path)

      (printf "get-started OK: preds ~a\n" preds))
    (lambda ()
      (delete-file train-path)
      (delete-file test-path)))
  (void))

(module+ main
  (run-example))

(module+ test
  (run-example))
