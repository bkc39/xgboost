#lang racket/base

;; Runner + tests for the literate example ../03-save-load.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require racket/format
         "../03-save-load.rkt")

(module+ main
  (define r (run-example))
  (printf "json file: ~a bytes\n" (hash-ref r 'file-bytes))
  (printf "ubj  blob: ~a bytes\n" (hash-ref r 'ubj-bytes))
  (printf "json blob: ~a bytes\n" (hash-ref r 'json-bytes))
  (printf "\n  ~a  matches baseline?\n" (~a "source" #:width 24))
  (for ([row (in-list `(("loaded from file" ,(hash-ref r 'file-match?))
                        ("loaded from ubj bytes" ,(hash-ref r 'ubj-match?))
                        ("loaded from json bytes" ,(hash-ref r 'json-match?))))])
    (printf "  ~a  ~a\n" (~a (car row) #:width 24) (if (cadr row) "yes" "NO")))
  (printf "\nbaseline predictions:\n")
  (for ([v (in-list (hash-ref r 'baseline))] [i (in-naturals)])
    (printf "  i=~a  ~a\n" i (~r v #:precision '(= 4) #:min-width 8))))

(module+ test
  (require rackunit)
  (define r (run-example))
  (check-true (> (hash-ref r 'file-bytes) 0))
  (check-true (hash-ref r 'file-match?))
  (check-true (hash-ref r 'ubj-match?))
  (check-true (hash-ref r 'json-match?)))
