#lang scribble/manual

@(require (for-label ffi/vector
                     racket/base
                     racket/contract
                     "../../main.rkt"))

@title[#:tag "ranking"]{Learning to Rank}

This chapter is the Racket counterpart of XGBoost's
@hyperlink["https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html"]{Learning
to Rank} tutorial. Ranking (LambdaMART) differs from regression and
classification in one structural way: rows are partitioned into @deftech{query
groups}, and the model learns to order the documents @emph{within} each group
rather than to predict an absolute target.

@section{Declaring query groups}

A ranking DMatrix carries the usual features and per-row relevance
@racket[#:labels], plus a @emph{group} layout: how many consecutive rows belong
to each query. Set it with @racket[dmatrix-set-group!], passing one row count
per query (the rows must already be laid out query-by-query):

@racketblock[
(require xgboost ffi/vector)

(code:comment "12 rows = three queries of four documents each")
(define dranking
  (make-dmatrix features #:nrow 12 #:ncol 3
                #:labels '(0 1 2 3   0 1 2 3   0 1 2 3)))

(dmatrix-set-group! dranking '(4 4 4))
]

XGBoost stores the cumulative offsets internally; read them back with
@racket[dmatrix-group-ptr], which returns @racket['(0 4 8 12)] for the example
above (one more entry than the number of queries).

Relevance labels are typically small non-negative integers — graded relevance,
where a larger label means a more relevant document.

@section{Training a ranker}

Train with one of the @racket["rank:*"] objectives.
@racket["rank:ndcg"] (LambdaMART optimizing nDCG) is the usual default;
@racket["rank:pairwise"] optimizes the pairwise loss directly. Pass an
@racket["eval_metric"] such as @racket["ndcg@"] @racket[N] to watch ranking
quality during training:

@racketblock[
(define ranker
  (train dranking
         #:params '((objective   . "rank:ndcg")
                    (eval_metric . "ndcg@8"))
         #:max-depth 4 #:eta 0.1 #:verbosity 0 #:rounds 50))
]

@margin-note{@racket["rank:map"] optimizes mean average precision and therefore
requires @emph{binary} (0/1) relevance labels; graded labels raise an error.
@racket["rank:ndcg"] and @racket["rank:pairwise"] accept graded relevance.}

@section{Predicting and scoring}

@racket[predict] returns one score per row. The scores are only meaningful
@emph{relative to other documents in the same query}: sort each query's
documents by descending score to get the ranking.

@racketblock[
(define scores (predict ranker dranking))

(code:comment "rank the first query (rows 0–3) by descending score")
(define query-0 (for/list ([i (in-range 4)]) (list-ref scores i)))
]

To measure quality, compute a ranking metric such as nDCG per query and average
across queries. A fully worked version — synthesizing graded-relevance queries,
training @racket["rank:ndcg"], and asserting a high held-out nDCG — is
@filepath{examples/29-learning-to-rank.rkt} in the package source.
