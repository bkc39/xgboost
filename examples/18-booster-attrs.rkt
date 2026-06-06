#lang scribble/lp2

@(require (for-label racket/base
                     xgboost))

@section[#:tag "ex-booster-attrs"]{Booster attributes}

A booster carries an arbitrary string→string attribute dictionary --- handy for
stamping a model with provenance. This example sets two attributes with
@racket[booster-set-attr!], reads them back with @racket[booster-attr], lists the
keys with @racket[booster-attr-names], then removes one with
@racket[booster-delete-attr!] (a missing key reads back as @racket[#f]).

@chunk[<r18-require>
(require xgboost)]

@chunk[<r18-provide>
(provide run-example)]

@chunk[<r18-run>
(define (run-example)
  (define b (make-booster))
  (booster-set-attr! b "owner" "racket")
  (booster-set-attr! b "purpose" "example")
  (define before-delete
    (hash 'owner (booster-attr b "owner")
          'purpose (booster-attr b "purpose")
          'names (sort (booster-attr-names b) string<?)))
  (booster-delete-attr! b "purpose")
  (hash 'before-delete before-delete
        'purpose-after-delete (booster-attr b "purpose")
        'names-after-delete (booster-attr-names b)))]

The harness @filepath{test/18-booster-attrs.rkt} prints the attributes and
asserts the round-trip and the post-delete state.

@chunk[<*>
  <r18-require>
  <r18-provide>
  <r18-run>]
