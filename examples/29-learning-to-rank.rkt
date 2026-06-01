#lang racket/base

;; Learning to Rank (LambdaMART), the Racket analogue of XGBoost's
;;
;;   https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html
;;
;; Ranking differs from plain regression/classification in one way: rows are
;; grouped into *queries*, and the model learns to order documents *within*
;; each query rather than to predict an absolute score.  You declare the query
;; layout with `dmatrix-set-group!` (a list of per-query row counts), then use
;; a `rank:*` objective.
;;
;; This demo synthesizes a relevance dataset where feature 0 carries the
;; relevance signal, trains `rank:ndcg`, and checks that the model orders each
;; held-out query by relevance (high nDCG).
;;
;; Run from the repo root:
;;   nix develop --command racket examples/29-learning-to-rank.rkt

(require ffi/vector
         racket/list
         rackunit
         xgboost)

(provide run-example)

(define ncol 3)
(define docs-per-query 8)

;; Build `nq` queries of `docs-per-query` documents.  Each document has a
;; graded relevance in 0..3; feature 0 is the relevance plus noise (the
;; signal), features 1 and 2 are noise.  Returns (values flat-features
;; labels group-sizes).
(define (make-queries nq seed)
  (random-seed seed)
  (define (noise) (- (* 2.0 (random)) 1.0))
  (define feats '())
  (define labels '())
  (for ([q (in-range nq)])
    (for ([d (in-range docs-per-query)])
      (define rel (exact->inexact (modulo (+ d (* 3 q)) 4)))
      (set! labels (cons rel labels))
      (set! feats (cons (list (+ rel (* 0.4 (noise)))   ; informative
                              (noise)                     ; noise
                              (* 0.5 (noise)))            ; noise
                        feats))))
  (values (map exact->inexact (append* (reverse feats)))
          (reverse labels)
          (make-list nq docs-per-query)))

(define (build-dmatrix flat labels groups)
  (define nrow (* (length groups) docs-per-query))
  (define dm (make-dmatrix (list->f32vector flat)
                           #:nrow nrow #:ncol ncol
                           #:labels (list->f32vector labels)))
  (dmatrix-set-group! dm groups)
  dm)

;; Normalized discounted cumulative gain of one query, given the model's
;; predicted scores and the true relevances for that query's documents.
(define (query-ndcg preds rels)
  (define (dcg order)
    (for/sum ([rel (in-list order)] [i (in-naturals)])
      (/ (- (expt 2.0 rel) 1.0) (/ (log (+ i 2.0)) (log 2.0)))))
  (define by-pred (map cdr (sort (map cons preds rels) > #:key car)))
  (define ideal   (sort rels >))
  (define idcg (dcg ideal))
  (if (zero? idcg) 1.0 (/ (dcg by-pred) idcg)))

(define (run-example)
  (define-values (train-feats train-labels train-groups) (make-queries 40 20260531))
  (define-values (test-feats  test-labels  test-groups)  (make-queries 10 99))

  (define dtrain (build-dmatrix train-feats train-labels train-groups))
  (define dtest  (build-dmatrix test-feats  test-labels  test-groups))

  ;; group_ptr is the cumulative offset array XGBoost maintains for the groups.
  (check-equal? (dmatrix-group-ptr dtrain)
                (for/list ([i (in-range (add1 (length train-groups)))])
                  (* i docs-per-query)))

  (define bst
    (train dtrain
           #:params '((objective    . "rank:ndcg")
                      (eval_metric  . "ndcg@8"))
           #:max-depth 4 #:eta 0.1 #:verbosity 0 #:rounds 50))

  ;; Score the held-out queries and average nDCG across them.
  (define preds (predict bst dtest))
  (define ndcgs
    (for/list ([q (in-range (length test-groups))])
      (define lo (* q docs-per-query))
      (define hi (+ lo docs-per-query))
      (query-ndcg (take (drop preds lo) docs-per-query)
                  (take (drop test-labels lo) docs-per-query))))
  (define mean-ndcg (/ (apply + ndcgs) (length ndcgs)))

  (printf "learning-to-rank: ~a queries x ~a docs, mean test nDCG@~a = ~a\n"
          (length test-groups) docs-per-query docs-per-query
          (real->decimal-string mean-ndcg 4))

  ;; A model that has learned the signal ranks far better than chance; random
  ;; ordering of these graded labels averages well below 0.9 nDCG.
  (check-true (> mean-ndcg 0.9)
              (format "expected strong ranking, got mean nDCG ~a" mean-ndcg))
  (void))

(module+ main
  (run-example))

(module+ test
  (run-example))
