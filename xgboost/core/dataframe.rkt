#lang racket/base

;; Polars DataFrame -> DMatrix bridge.
;;
;; This is the conversion layer that lets a Polars `dataframe?` stand in for
;; ordinary Racket data.  It extracts numeric columns into `f32vector`s and
;; reuses the existing columnar DMatrix constructor, so a DataFrame "just
;; works" wherever feature data is expected.  Wiring this into `make-dmatrix`
;; / `train` / `predict` (auto-detecting a DataFrame) is handled separately.
;;
;; We import only the polars bindings we use. A blanket `(require polars)` would
;; also pull in its many `racket/base`-shadowing exports (`=`, `<`, `min`, `max`,
;; `filter`, `sort`, `first`, `last`, `count`, ...) as ambiguous bindings, so we
;; name exactly what we need instead.

(require ffi/vector
         (only-in polars ref len column-names polars-null?)
         "coerce.rkt"
         "dmatrix.rkt")

(provide series->f32vector
         dataframe-column->f32vector
         dataframe->dmatrix)

;; Read a Polars series of numbers into an `f32vector`.  Non-numeric cells
;; (string columns, or Polars nulls) raise rather than silently coercing, so
;; type/missing-data problems surface at conversion time.
(define (series->f32vector s #:who [who 'series->f32vector])
  (define n (len s))
  (define v (make-f32vector n))
  (for ([i (in-range n)])
    (define x (ref s i))
    (unless (real? x)
      (error who
             "column is not purely numeric: got ~v at row ~a~a"
             x i
             (if (polars-null? x) " (null)" "")))
    (f32vector-set! v i (exact->inexact x)))
  v)

;; Extract one column (by name or index) of a DataFrame as an `f32vector`.
(define (dataframe-column->f32vector df col)
  (series->f32vector (ref df col) #:who 'dataframe-column->f32vector))

;; Convert a DataFrame to a DMatrix.
;;
;;  - #:labels / #:weights accept a column name (string), in which case that
;;    column is pulled out of the frame and excluded from the features; or any
;;    ordinary sequence (list/vector/f32vector), passed through as-is.
;;  - #:feature-columns pins the feature set (a list of column-name strings);
;;    otherwise every column except the label/weight columns is used.
;;  - The DMatrix is built columnar (one f32vector per column, no transpose)
;;    and its feature names are set from the chosen columns.
(define (dataframe->dmatrix df
                            #:labels [labels #f]
                            #:weights [weights #f]
                            #:feature-columns [feature-columns #f]
                            #:missing [missing +nan.0])
  (define label-col (and (string? labels) labels))
  (define weight-col (and (string? weights) weights))
  (define feat-cols
    (or feature-columns
        (for/list ([c (in-list (column-names df))]
                   #:unless (or (equal? c label-col) (equal? c weight-col)))
          c)))
  (when (null? feat-cols)
    (error 'dataframe->dmatrix
           "no feature columns remain after excluding label/weight columns"))
  (define columns
    (map (lambda (c) (dataframe-column->f32vector df c)) feat-cols))
  (define dm (make-dmatrix-from-columnar columns missing))
  (dmatrix-set-feature-names! dm feat-cols)
  (define (resolve who spec)
    (cond
      [(string? spec) (dataframe-column->f32vector df spec)]
      [spec (sequence->f32vector who spec)]
      [else #f]))
  (let ([lv (resolve 'dataframe->dmatrix labels)])
    (when lv (dmatrix-set-label! dm lv)))
  (let ([wv (resolve 'dataframe->dmatrix weights)])
    (when wv (dmatrix-set-weight! dm wv)))
  dm)

(module+ test
  (require rackunit
           (only-in polars dataframe series)
           (only-in "../foreign.rkt" dmatrix-rows dmatrix-cols))

  (define df
    (dataframe
     (list (series '(1.0 2.0 3.0 4.0) #:name "f0")
           (series '(10.0 20.0 30.0 40.0) #:name "f1")
           (series '(0.0 1.0 1.0 0.0) #:name "y"))))

  (test-case "series->f32vector reads numeric column"
    (check-equal? (f32vector->list (series->f32vector (ref df "f0")))
                  '(1.0 2.0 3.0 4.0)))

  (test-case "dataframe->dmatrix with label column"
    (define dm (dataframe->dmatrix df #:labels "y"))
    (check-equal? (dmatrix-rows dm) 4)
    (check-equal? (dmatrix-cols dm) 2 "label column excluded from features")
    (check-equal? (dmatrix-feature-names dm) '("f0" "f1"))
    (check-equal? (dmatrix-label dm) '(0.0 1.0 1.0 0.0)))

  (test-case "explicit feature-columns and external label sequence"
    (define dm (dataframe->dmatrix df #:feature-columns '("f1") #:labels '(0 1 1 0)))
    (check-equal? (dmatrix-cols dm) 1)
    (check-equal? (dmatrix-feature-names dm) '("f1"))
    (check-equal? (dmatrix-label dm) '(0.0 1.0 1.0 0.0)))

  (test-case "non-numeric feature column raises"
    (define sdf
      (dataframe (list (series '("a" "b") #:name "s")
                          (series '(1.0 2.0) #:name "y"))))
    (check-exn exn:fail? (lambda () (dataframe->dmatrix sdf #:labels "y")))))
