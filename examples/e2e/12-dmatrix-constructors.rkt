#lang racket/base

(require ffi/vector
         racket/file
         racket/list
         xgboost/ffi)

(provide run-example)

(define expected-dense
  '((1.0 2.0 3.0)
    (4.0 5.0 6.0)))

(define expected-sparse
  '((1.0 +nan.0 3.0)
    (+nan.0 5.0 6.0)))

(define (nan? v)
  (not (= v v)))

(define (same-cell? expected got)
  (if (nan? expected)
      (nan? got)
      (< (abs (- got expected)) 1e-6)))

(define (same-matrix? expected got)
  (and (= (length expected) (length got))
       (for/and ([expected-row (in-list expected)]
                 [got-row (in-list got)])
         (and (= (length expected-row) (length got-row))
              (for/and ([expected-cell (in-list expected-row)]
                        [got-cell (in-list got-row)])
                (same-cell? expected-cell got-cell))))))

(define (matrix-summary dm)
  (hash 'rows (dmatrix-nrow dm)
        'cols (dmatrix-ncol dm)
        'values (dmatrix->list dm)))

(define (make-libsvm-dmatrix)
  (define tmp (make-temporary-file "xgboost-e2e-~a.libsvm"))
  (dynamic-wind
    void
    (lambda ()
      (call-with-output-file tmp
        (lambda (out)
          (displayln "0 0:1 1:2 2:3" out)
          (displayln "1 0:4 1:5 2:6" out))
        #:exists 'truncate)
      (dmatrix-create-from-uri
       (format "{\"uri\":\"~a?format=libsvm\",\"silent\":1}"
               (path->string tmp))))
    (lambda ()
      (when (file-exists? tmp)
        (delete-file tmp)))))

(define (run-example)
  (define dense
    (dmatrix-create-from-dense
     (f32vector 1.0 2.0 3.0
                4.0 5.0 6.0)
     2
     3
     -1.0))
  (define csr
    (dmatrix-create-from-csr
     (u64vector 0 2 4)
     (u32vector 0 2 1 2)
     (f32vector 1.0 3.0 5.0 6.0)
     3
     -1.0))
  (define csc
    (dmatrix-create-from-csc
     (u64vector 0 1 2 4)
     (u32vector 0 1 0 1)
     (f32vector 1.0 5.0 3.0 6.0)
     2
     -1.0))
  (define columnar
    (dmatrix-create-from-columnar
     (list (f32vector 1.0 4.0)
           (f32vector 2.0 5.0)
           (f32vector 3.0 6.0))
     -1.0))
  (define libsvm (make-libsvm-dmatrix))

  (define result
    (hash 'dense (matrix-summary dense)
          'csr (matrix-summary csr)
          'csc (matrix-summary csc)
          'columnar (matrix-summary columnar)
          'libsvm (matrix-summary libsvm)))

  (dmatrix-free! dense)
  (dmatrix-free! csr)
  (dmatrix-free! csc)
  (dmatrix-free! columnar)
  (dmatrix-free! libsvm)
  result)

(module+ main
  (define result (run-example))
  (for ([name (in-list '(dense csr csc columnar libsvm))])
    (define summary (hash-ref result name))
    (printf "~a: ~ax~a\n"
            name
            (hash-ref summary 'rows)
            (hash-ref summary 'cols))))

(module+ test
  (require rackunit)

  (define result (run-example))

  (for ([name (in-list '(dense columnar libsvm))])
    (define summary (hash-ref result name))
    (check-equal? (hash-ref summary 'rows) 2)
    (check-equal? (hash-ref summary 'cols) 3)
    (check-true (same-matrix? expected-dense (hash-ref summary 'values))))

  (for ([name (in-list '(csr csc))])
    (define summary (hash-ref result name))
    (check-equal? (hash-ref summary 'rows) 2)
    (check-equal? (hash-ref summary 'cols) 3)
    (check-true (same-matrix? expected-sparse (hash-ref summary 'values)))))
