#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     racket/list
                     xgboost))

@section[#:tag "ex-param-recipes"]{Parameter recipes}

Several upstream XGBoost tutorials are really just parameter settings on top of
ordinary training. This example collects four, all expressed through
@racket[#:params]: the DART booster (@racket["booster=dart"] with drop rates),
monotonic constraints (@racket["monotone_constraints"]), feature-interaction
constraints (@racket["interaction_constraints"]), and random-forest mode
(@racket["num_parallel_tree"] with subsampling, one boosting round). The headline
check is that the monotone model really is non-decreasing in the constrained
feature.

@chunk[<r28-require>
(require ffi/vector
         racket/list
         xgboost)]

@chunk[<r28-provide>
(provide run-example)]

@bold{Synthetic data.} A deterministic 200-row set where @tt{y} increases in
feature 0, depends mildly on feature 1, and feature 2 is noise:

@chunk[<r28-data>
  (define ncol 3)
  (define (make-data)
    (random-seed 20260531)
    (define (rnd) (- (* 2.0 (random)) 1.0))
    (define xs (for/list ([_ (in-range 200)]) (list (rnd) (rnd) (rnd))))
    (values xs (for/list ([x (in-list xs)])
                 (+ (* 2.0 (first x)) (* 0.5 (second x)) (* 0.2 (rnd))))))
  (define (rows->dmatrix xs ys)
    (make-dmatrix (list->f32vector (map exact->inexact (append* xs)))
                  #:nrow (length xs) #:ncol ncol
                  #:labels (list->f32vector (map exact->inexact ys))))]

@bold{The recipes.} Each is a thin wrapper over @racket[train] with a different
@racket[#:params]:

@chunk[<r28-recipes>
  (define (train-recipe xs ys extra)
    (train (rows->dmatrix xs ys)
           #:params (cons '(objective . "reg:squarederror") extra)
           #:max-depth 3 #:eta 0.1 #:verbosity 0 #:rounds 30))
  (define (recipe-dart xs ys)
    (train-recipe xs ys '((booster . "dart") (rate_drop . "0.1") (skip_drop . "0.5"))))
  (define (recipe-monotone xs ys)
    (train-recipe xs ys '((monotone_constraints . "(1,0,0)"))))
  (define (recipe-interaction xs ys)
    (train-recipe xs ys '((interaction_constraints . "[[0,1],[2]]"))))
  (define (recipe-random-forest xs ys)
    (train (rows->dmatrix xs ys)
           #:params '((objective . "reg:squarederror") (num_parallel_tree . "20")
                      (subsample . "0.8") (colsample_bynode . "0.8"))
           #:max-depth 4 #:eta 1.0 #:verbosity 0 #:rounds 1))]

@bold{Monotonicity probe.} Sweep feature 0 with the others held at 0 and count
any decrease in the prediction:

@chunk[<r28-probe>
  (define (monotone-violations bst)
    (define probe (for/list ([k (in-range 21)]) (list (+ -1.0 (* 0.1 k)) 0.0 0.0)))
    (define preds
      (predict-from-dense bst (list->f32vector (map exact->inexact (append* probe)))
                          #:nrow (length probe) #:ncol ncol))
    (for/sum ([a (in-list preds)] [b (in-list (cdr preds))])
      (if (< b (- a 1e-6)) 1 0)))]

@bold{The run.} Train all four, record their round counts and a sample
prediction, and probe the monotone model. @racket[run-example] returns the
per-recipe summary plus the violation count (expected 0):

@chunk[<r28-run>
(define (run-example)
  (define-values (xs ys) (make-data))
  (define recipes
    (list (cons "dart" (recipe-dart xs ys))
          (cons "monotone" (recipe-monotone xs ys))
          (cons "interaction" (recipe-interaction xs ys))
          (cons "random-forest" (recipe-random-forest xs ys))))
  (define summary
    (for/list ([r (in-list recipes)])
      (define preds (predict (cdr r) (rows->dmatrix xs ys)))
      (list (car r) (booster-boosted-rounds (cdr r)) (car preds)
            (for/and ([p (in-list preds)]) (= p p)))))
  (define mono (cdr (assoc "monotone" recipes)))
  (hash 'summary summary 'violations (monotone-violations mono)))]

The harness @filepath{test/28-param-recipes.rkt} prints each recipe's round
count and a sample prediction, and asserts every recipe gives finite
predictions and the monotone constraint holds across the sweep (zero
violations).

@chunk[<*>
  <r28-require>
  <r28-provide>
  <r28-data>
  <r28-recipes>
  <r28-probe>
  <r28-run>]
