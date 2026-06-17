#lang racket/base

;; Input coercion shared by the high-level `core` layer.
;;
;; The public API accepts ordinary Racket data — lists, vectors, nested
;; rows — and these helpers normalize it into the flat f32/u32 vectors the
;; foreign layer expects, inferring shape where possible.

(require ffi/vector)

(provide sequence->f32vector
         sequence->u32vector
         row-sequence?
         rows->matrix
         coerce-matrix)

(define (sequence->f32vector who xs)
  (cond
    [(f32vector? xs) xs]
    [(list? xs) (list->f32vector (map exact->inexact xs))]
    [(vector? xs) (list->f32vector (map exact->inexact (vector->list xs)))]
    [else (raise-argument-error who "list, vector, or f32vector" xs)]))

(define (sequence->u32vector who xs)
  (cond
    [(u32vector? xs) xs]
    [(list? xs) (list->u32vector xs)]
    [(vector? xs) (list->u32vector (vector->list xs))]
    [else (raise-argument-error who "list, vector, or u32vector" xs)]))

(define (row-sequence? v)
  (or (list? v) (vector? v)))

;; Flatten a list-of-lists / vector-of-vectors into a row-major f32vector,
;; returning the inferred (vec nrow ncol).  Rejects ragged input.
(define (rows->matrix who rows0)
  (define rows (if (vector? rows0) (vector->list rows0) rows0))
  (unless (and (list? rows) (andmap row-sequence? rows))
    (raise-argument-error who "list-of-lists or vector-of-vectors" rows0))
  (define nrow (length rows))
  (define ncol
    (cond
      [(zero? nrow) (raise-argument-error who "at least one row" rows0)]
      [else (length (if (vector? (car rows)) (vector->list (car rows)) (car rows)))]))
  (when (zero? ncol)
    (raise-argument-error who "at least one column" rows0))
  (define flat
    (for/fold ([acc '()] #:result (reverse acc))
              ([row0 (in-list rows)])
      (define row (if (vector? row0) (vector->list row0) row0))
      (unless (= (length row) ncol)
        (error who "ragged matrix: expected ~a columns, got ~a" ncol (length row)))
      (append (reverse row) acc)))
  (values (list->f32vector (map exact->inexact flat)) nrow ncol))

;; Coerce arbitrary `data` into (vec nrow ncol).  Nested rows infer their own
;; shape (and cross-check any explicitly supplied #:nrow / #:ncol); flat data
;; requires both dimensions to be given.
(define (coerce-matrix data nrow ncol)
  (cond
    [(and (or (list? data) (vector? data))
          (positive? (if (vector? data) (vector-length data) (length data)))
          (row-sequence? (if (vector? data) (vector-ref data 0) (car data))))
     (define-values (vec inferred-nrow inferred-ncol)
       (rows->matrix 'make-dmatrix data))
     (when (and nrow (not (= nrow inferred-nrow)))
       (error 'make-dmatrix "given #:nrow ~a does not match inferred row count ~a"
              nrow inferred-nrow))
     (when (and ncol (not (= ncol inferred-ncol)))
       (error 'make-dmatrix "given #:ncol ~a does not match inferred column count ~a"
              ncol inferred-ncol))
     (values vec inferred-nrow inferred-ncol)]
    [else
     (unless (and nrow ncol)
       (error 'make-dmatrix
              "flat data requires both #:nrow and #:ncol"))
     (values (sequence->f32vector 'make-dmatrix data) nrow ncol)]))
