#lang racket/base

;; Small helpers shared by the narrative examples.  These stand in for the
;; "batteries" a Python user gets from scikit-learn — a dataset loader and a
;; train/test split — so the examples can read like a line-for-line translation
;; of the upstream XGBoost quickstart.
;;
;; This module is NOT part of the installed `xgboost` collection; it lives under
;; examples/ purely to support the demos, so it adds no dependencies to the
;; package itself.

(require net/url
         racket/port
         racket/list
         racket/string
         racket/runtime-path)

(provide load-iris
         train-test-split)

;; The classic UCI iris dataset.  We try the network first and fall back to a
;; committed copy so the examples run offline (nix sandbox, pkg-build CI).
(define iris-url
  "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")

(define-runtime-path iris-fallback "data/iris.csv")

(define species->label
  (hash "Iris-setosa"     0
        "Iris-versicolor" 1
        "Iris-virginica"  2))

;; Best-effort download; returns the CSV text, or #f on any failure/timeout.
(define (try-download url #:timeout [timeout 5])
  (with-handlers ([(lambda (_) #t) (lambda (_) #f)])
    (define ch (make-channel))
    (define worker
      (thread
       (lambda ()
         (channel-put
          ch
          (with-handlers ([(lambda (_) #t) (lambda (_) #f)])
            (port->string (get-pure-port (string->url url))))))))
    (define result (sync/timeout timeout ch))
    (unless result (kill-thread worker))
    result))

;; Parse the iris CSV text into (values X y): X is a list of 4-element feature
;; lists, y is a list of integer class labels.  Trivial parse — iris has no
;; quoting or embedded commas.
(define (parse-iris text)
  (for/fold ([xs '()] [ys '()] #:result (values (reverse xs) (reverse ys)))
            ([line (in-list (string-split text "\n"))]
             #:when (positive? (string-length (string-trim line))))
    (define cols (string-split (string-trim line) ","))
    (values (cons (map string->number (take cols 4)) xs)
            (cons (hash-ref species->label (list-ref cols 4)) ys))))

;; load-iris : -> (values X y)
;; Mirrors sklearn's `data = load_iris()` / `data['data'], data['target']`.
(define (load-iris)
  (define text (or (try-download iris-url)
                   (call-with-input-file iris-fallback port->string)))
  (parse-iris text))

;; train-test-split : X y [#:test-size frac] [#:seed seed]
;;                 -> (values X-train X-test y-train y-test)
;; Mirrors sklearn's train_test_split argument and return order.  The split is
;; a seeded shuffle so examples are reproducible (sklearn's is random).
(define (train-test-split X y #:test-size [test-size 0.2] #:seed [seed 0])
  (unless (= (length X) (length y))
    (error 'train-test-split "X and y must have the same length"))
  (define rng (make-pseudo-random-generator))
  (parameterize ([current-pseudo-random-generator rng])
    (random-seed seed)
    (define n (length X))
    (define n-test (inexact->exact (round (* test-size n))))
    (define test-set
      (for/hash ([i (in-list (take (shuffle (range n)) n-test))])
        (values i #t)))
    ;; Tag each (row . label) with its original index, then split once.
    (define tagged
      (for/list ([row (in-list X)] [lbl (in-list y)] [i (in-naturals)])
        (list i row lbl)))
    (define-values (test train)
      (partition (lambda (e) (hash-ref test-set (car e) #f)) tagged))
    (values (map cadr train) (map cadr test)
            (map caddr train) (map caddr test))))
