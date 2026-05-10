#lang racket/base

(require ffi/vector
         racket/file
         racket/list
         xgboost)

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
  (hash 'rows (dmatrix-rows dm)
        'cols (dmatrix-cols dm)
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
      (make-dmatrix-from-uri tmp #:format "libsvm"))
    (lambda ()
      (when (file-exists? tmp)
        (delete-file tmp)))))

(define (run-example)
  (define dense
    (make-dmatrix
     (f32vector 1.0 2.0 3.0
                4.0 5.0 6.0)
     #:nrow 2
     #:ncol 3
     #:missing -1.0))
  (define csr
    (make-dmatrix-from-csr
     (u64vector 0 2 4)
     (u32vector 0 2 1 2)
     (f32vector 1.0 3.0 5.0 6.0)
     3
     -1.0))
  (define csc
    (make-dmatrix-from-csc
     (u64vector 0 1 2 4)
     (u32vector 0 1 0 1)
     (f32vector 1.0 5.0 3.0 6.0)
     2
     -1.0))
  (define columnar
    (make-dmatrix-from-columnar
     (list (f32vector 1.0 4.0)
           (f32vector 2.0 5.0)
           (f32vector 3.0 6.0))
     -1.0))
  (define libsvm (make-libsvm-dmatrix))

  (hash 'dense (matrix-summary dense)
        'csr (matrix-summary csr)
        'csc (matrix-summary csc)
        'columnar (matrix-summary columnar)
        'libsvm (matrix-summary libsvm)))

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
