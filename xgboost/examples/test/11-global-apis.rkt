#lang racket/base

;; Runner + tests for the literate example ../11-global-apis.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require json
         racket/port
         "../11-global-apis.rkt")

(define (json->jsexpr s) (with-input-from-string s read-json))
(define (json-object-string? s) (hash? (json->jsexpr s)))

(module+ main
  (define result (run-example))
  (printf "build info JSON: ~a\n" (json-object-string? (hash-ref result 'build-info)))
  (printf "verbosity set during example: ~a\n"
          (hash-ref (json->jsexpr (hash-ref result 'during-config)) 'verbosity))
  (printf "global config restored: ~a\n"
          (equal? (json->jsexpr (hash-ref result 'before-config))
                  (json->jsexpr (hash-ref result 'after-config)))))

(module+ test
  (require rackunit)
  (define result (run-example))
  (check-true (json-object-string? (hash-ref result 'build-info)))
  (check-true (json-object-string? (hash-ref result 'before-config)))
  (check-true (json-object-string? (hash-ref result 'during-config)))
  (check-true (json-object-string? (hash-ref result 'after-config)))
  (check-equal? (hash-ref (json->jsexpr (hash-ref result 'during-config)) 'verbosity) 0)
  (check-equal? (json->jsexpr (hash-ref result 'after-config))
                (json->jsexpr (hash-ref result 'before-config))))
