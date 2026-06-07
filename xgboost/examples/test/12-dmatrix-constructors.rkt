#lang racket/base

;; Runner + tests for the literate example ../12-dmatrix-constructors.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require "../12-dmatrix-constructors.rkt")

(define expected-dense '((1.0 2.0 3.0) (4.0 5.0 6.0)))
(define expected-sparse '((1.0 +nan.0 3.0) (+nan.0 5.0 6.0)))

(define (nan? v) (not (= v v)))
(define (same-cell? e g) (if (nan? e) (nan? g) (< (abs (- g e)) 1e-6)))
(define (same-matrix? expected got)
  (and (= (length expected) (length got))
       (for/and ([er (in-list expected)] [gr (in-list got)])
         (and (= (length er) (length gr))
              (for/and ([e (in-list er)] [g (in-list gr)]) (same-cell? e g))))))

(module+ main
  (define result (run-example))
  (for ([name (in-list '(dense csr csc columnar libsvm))])
    (define s (hash-ref result name))
    (printf "~a: ~ax~a\n" name (hash-ref s 'rows) (hash-ref s 'cols))))

(module+ test
  (require rackunit)
  (define result (run-example))
  (for ([name (in-list '(dense columnar libsvm))])
    (define s (hash-ref result name))
    (check-equal? (hash-ref s 'rows) 2)
    (check-equal? (hash-ref s 'cols) 3)
    (check-true (same-matrix? expected-dense (hash-ref s 'values))))
  (for ([name (in-list '(csr csc))])
    (define s (hash-ref result name))
    (check-equal? (hash-ref s 'rows) 2)
    (check-equal? (hash-ref s 'cols) 3)
    (check-true (same-matrix? expected-sparse (hash-ref s 'values)))))
