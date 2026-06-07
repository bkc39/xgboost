#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     racket/list
                     xgboost))

@section[#:tag "ex-learning-to-rank"]{Learning to rank}

Ranking differs from regression/classification in one way: rows are grouped into
@emph{queries}, and the model learns to order documents @emph{within} each query.
You declare the layout with @racket[dmatrix-set-group!] (a list of per-query row
counts) and use a @racket["rank:*"] objective. This is the Racket analogue of
upstream's
@hyperlink["https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html"]{Learning
to Rank} tutorial: it synthesizes a graded-relevance dataset where feature 0
carries the signal, trains @racket["rank:ndcg"], and checks held-out nDCG.

@chunk[<r29-require>
(require ffi/vector
         racket/list
         xgboost)]

@chunk[<r29-provide>
(provide run-example)]

@bold{Synthetic queries.} Each query has eight documents with graded relevance
@tt{0..3}; feature 0 is relevance plus noise, features 1 and 2 are noise:

@chunk[<r29-data>
  (define ncol 3)
  (define docs-per-query 8)
  (define (make-queries nq seed)
    (random-seed seed)
    (define (noise) (- (* 2.0 (random)) 1.0))
    (define feats '()) (define labels '())
    (for ([q (in-range nq)])
      (for ([d (in-range docs-per-query)])
        (define rel (exact->inexact (modulo (+ d (* 3 q)) 4)))
        (set! labels (cons rel labels))
        (set! feats (cons (list (+ rel (* 0.4 (noise))) (noise) (* 0.5 (noise))) feats))))
    (values (map exact->inexact (append* (reverse feats)))
            (reverse labels) (make-list nq docs-per-query)))
  (define (build-dmatrix flat labels groups)
    (define dm (make-dmatrix (list->f32vector flat)
                             #:nrow (* (length groups) docs-per-query) #:ncol ncol
                             #:labels (list->f32vector labels)))
    (dmatrix-set-group! dm groups)
    dm)]

@bold{nDCG.} Normalized discounted cumulative gain for one query, given the
model's predicted order and the true relevances:

@chunk[<r29-ndcg>
  (define (query-ndcg preds rels)
    (define (dcg order)
      (for/sum ([rel (in-list order)] [i (in-naturals)])
        (/ (- (expt 2.0 rel) 1.0) (/ (log (+ i 2.0)) (log 2.0)))))
    (define by-pred (map cdr (sort (map cons preds rels) > #:key car)))
    (define idcg (dcg (sort rels >)))
    (if (zero? idcg) 1.0 (/ (dcg by-pred) idcg)))]

@bold{The run.} Train @racket["rank:ndcg"] on 40 queries, then average nDCG over
10 held-out queries. @racket[run-example] also returns the training matrix's
group-pointer (the cumulative offsets XGBoost derives from the group sizes) so
the harness can check the layout:

@chunk[<r29-run>
(define (run-example)
  (define-values (tr-f tr-l tr-g) (make-queries 40 20260531))
  (define-values (te-f te-l te-g) (make-queries 10 99))
  (define dtrain (build-dmatrix tr-f tr-l tr-g))
  (define dtest (build-dmatrix te-f te-l te-g))
  (define bst
    (train dtrain #:params '((objective . "rank:ndcg") (eval_metric . "ndcg@8"))
           #:max-depth 4 #:eta 0.1 #:verbosity 0 #:rounds 50))
  (define preds (predict bst dtest))
  (define ndcgs
    (for/list ([q (in-range (length te-g))])
      (define lo (* q docs-per-query))
      (query-ndcg (take (drop preds lo) docs-per-query)
                  (take (drop te-l lo) docs-per-query))))
  (hash 'group-ptr (dmatrix-group-ptr dtrain)
        'expected-group-ptr (for/list ([i (in-range (add1 (length tr-g)))])
                              (* i docs-per-query))
        'n-test-queries (length te-g)
        'docs-per-query docs-per-query
        'mean-ndcg (/ (apply + ndcgs) (length ndcgs))))]

The harness @filepath{test/29-learning-to-rank.rkt} prints the mean held-out
nDCG and asserts the group pointer matches the declared layout and the model
ranks far better than chance (mean nDCG > 0.9).

@chunk[<*>
  <r29-require>
  <r29-provide>
  <r29-data>
  <r29-ndcg>
  <r29-run>]
