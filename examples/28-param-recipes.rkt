#lang racket/base

;; Parameter-driven modeling recipes, all expressed through `#:params`.
;;
;; These mirror several upstream XGBoost tutorials that are really just
;; parameter settings on top of ordinary training:
;;
;;   * DART booster                     (booster=dart + drop rates)
;;   * Monotonic Constraints            (monotone_constraints)
;;   * Feature Interaction Constraints  (interaction_constraints)
;;   * Random Forests in XGBoost        (num_parallel_tree + subsampling)
;;
;; Each recipe trains a small regressor on a synthetic dataset and the
;; `module+ test` submodule asserts the expected behavior — most usefully
;; that the monotone constraint actually produces a non-decreasing response
;; in the constrained feature.
;;
;; Run from the repo root:
;;   nix develop --command racket examples/28-param-recipes.rkt

(require ffi/vector
         racket/list
         rackunit
         xgboost)

(provide run-example)

;; ---- Synthetic data ------------------------------------------------------
;; y is increasing in feature 0, mildly dependent on feature 1, and feature 2
;; is noise.  Deterministic via a fixed seed.
(define ncol 3)
(define nrow 200)

(define (make-data)
  (random-seed 20260531)
  (define (rnd) (- (* 2.0 (random)) 1.0))
  (define xs (for/list ([_ (in-range nrow)]) (list (rnd) (rnd) (rnd))))
  (define ys (for/list ([x (in-list xs)])
               (+ (* 2.0 (first x)) (* 0.5 (second x)) (* 0.2 (rnd)))))
  (values xs ys))

(define (rows->dmatrix xs ys)
  (make-dmatrix (list->f32vector (map exact->inexact (append* xs)))
                #:nrow (length xs) #:ncol ncol
                #:labels (list->f32vector (map exact->inexact ys))))

(define (finite? v) (= v v))
(define (all-finite? preds) (for/and ([p (in-list preds)]) (finite? p)))

;; ---- Recipes -------------------------------------------------------------

(define (train-recipe xs ys extra-params)
  (train (rows->dmatrix xs ys)
         #:params (cons '(objective . "reg:squarederror") extra-params)
         #:max-depth 3 #:eta 0.1 #:verbosity 0 #:rounds 30))

;; DART drops trees during boosting to regularize.
(define (recipe-dart xs ys)
  (train-recipe xs ys '((booster    . "dart")
                        (rate_drop  . "0.1")
                        (skip_drop  . "0.5"))))

;; Force the response to be non-decreasing in feature 0, unconstrained elsewhere.
(define (recipe-monotone xs ys)
  (train-recipe xs ys '((monotone_constraints . "(1,0,0)"))))

;; Only allow features 0 and 1 to interact; feature 2 stays on its own.
(define (recipe-interaction xs ys)
  (train-recipe xs ys '((interaction_constraints . "[[0,1],[2]]"))))

;; A random forest is one boosting round of many parallel subsampled trees.
(define (recipe-random-forest xs ys)
  (train (rows->dmatrix xs ys)
         #:params '((objective         . "reg:squarederror")
                    (num_parallel_tree . "20")
                    (subsample         . "0.8")
                    (colsample_bynode  . "0.8"))
         #:max-depth 4 #:eta 1.0 #:verbosity 0 #:rounds 1))

;; Probe monotonicity: sweep feature 0 with features 1,2 held at 0 and check
;; predictions never decrease.
(define (monotone-violations bst)
  (define grid (for/list ([k (in-range 21)]) (+ -1.0 (* 0.1 k))))
  (define probe (for/list ([x0 (in-list grid)]) (list x0 0.0 0.0)))
  (define preds
    (predict-from-dense bst (list->f32vector (map exact->inexact (append* probe)))
                        #:nrow (length probe) #:ncol ncol))
  (for/sum ([a (in-list preds)] [b (in-list (cdr preds))])
    (if (< b (- a 1e-6)) 1 0)))

(define (run-example)
  (define-values (xs ys) (make-data))

  (define dart   (recipe-dart xs ys))
  (define mono   (recipe-monotone xs ys))
  (define inter  (recipe-interaction xs ys))
  (define rf     (recipe-random-forest xs ys))

  (for ([name '("dart" "monotone" "interaction" "random-forest")]
        [bst (list dart mono inter rf)])
    (define preds (predict bst (rows->dmatrix xs ys)))
    (check-true (all-finite? preds) (format "~a produced finite predictions" name))
    (printf "~a: boosted-rounds=~a  pred[0]=~a\n"
            name (booster-boosted-rounds bst)
            (real->decimal-string (car preds) 3)))

  ;; The headline guarantee: the monotone model is non-decreasing in feature 0.
  (define violations (monotone-violations mono))
  (check-equal? violations 0 "monotone constraint holds across the feature-0 sweep")
  (printf "monotone sweep: ~a violations (expected 0)\n" violations)
  (void))

(module+ main
  (run-example))

(module+ test
  (run-example))
