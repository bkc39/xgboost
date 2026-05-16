#lang racket/base

;; JSON array-interface builders and parsers.
;;
;; XGBoost's modern data-ingest APIs take buffers described by a JSON
;; "__array_interface__" string: a raw pointer integer, a typestr, and a
;; shape.  These helpers encode Racket f32/u32/u64 vectors into that form
;; and decode the JSON XGBoost hands back.

(require ffi/unsafe
         ffi/vector
         json
         racket/string)

(provide json-number
         missing-config
         dims->json
         array-interface-json
         f32-array-interface
         u64-array-interface
         u32-array-interface
         parse-array-interface
         array-interface->u64vector
         array-interface->f32vector)

(define (json-number v)
  (cond
    [(not (= v v)) "NaN"]
    [(eqv? v +inf.0) "Infinity"]
    [(eqv? v -inf.0) "-Infinity"]
    [else (number->string v)]))

(define (missing-config missing)
  (format "{\"missing\":~a}" (json-number missing)))

(define (dims->json dims)
  (string-join (map number->string dims) ","))

(define (array-interface-json ptr typestr dims)
  (format "{\"data\":[~a,false],\"typestr\":\"~a\",\"shape\":[~a],\"version\":3}"
          (cast ptr _pointer _uintptr)
          typestr
          (dims->json dims)))

(define (f32-array-interface vec dims)
  (array-interface-json (f32vector->cpointer vec) "<f4" dims))

(define (u64-array-interface vec)
  (array-interface-json (u64vector->cpointer vec) "<u8"
                        (list (u64vector-length vec))))

(define (u32-array-interface vec)
  (array-interface-json (u32vector->cpointer vec) "<u4"
                        (list (u32vector-length vec))))

(define (parse-array-interface who s expected-typestr)
  (define j (string->jsexpr s))
  (define ptr-int (car (hash-ref j 'data)))
  (define typestr (hash-ref j 'typestr))
  (define shape (hash-ref j 'shape))
  (unless (equal? typestr expected-typestr)
    (error who "expected typestr ~s, got ~s" expected-typestr typestr))
  (values (cast ptr-int _uintptr _pointer)
          (apply * shape)))

(define (array-interface->u64vector s)
  (define-values (ptr n) (parse-array-interface 'array-interface->u64vector s "<u8"))
  (define vec (make-u64vector n))
  (memcpy (u64vector->cpointer vec) ptr (* n 8))
  vec)

(define (array-interface->f32vector s)
  (define-values (ptr n) (parse-array-interface 'array-interface->f32vector s "<f4"))
  (define vec (make-f32vector n))
  (memcpy (f32vector->cpointer vec) ptr (* n 4))
  vec)
