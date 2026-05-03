#lang racket/base

(require ffi/vector
         racket/file
         xgboost/ffi)

(provide run-example)

(define (same-matrix? expected got)
  (and (= (length expected) (length got))
       (for/and ([erow (in-list expected)]
                 [grow (in-list got)])
         (and (= (length erow) (length grow))
              (for/and ([e (in-list erow)]
                        [g (in-list grow)])
                (< (abs (- g e)) 1e-6))))))

(define (run-example)
  (define dm
    (dmatrix-create-from-dense
     (f32vector 1.0 2.0
                3.0 4.0
                5.0 6.0)
     3
     2
     -1.0))
  (define sliced (dmatrix-slice dm '(2 0)))
  (define sliced-values (dmatrix->list sliced))
  (define tmp (make-temporary-file "xgboost-dmatrix-~a.buffer"))
  (when (file-exists? tmp)
    (delete-file tmp))
  (define loaded-summary
    (dynamic-wind
      void
      (lambda ()
        (dmatrix-save-binary! sliced tmp)
        (define loaded
          (dmatrix-create-from-uri
           (format "{\"uri\":\"~a\",\"silent\":1}" (path->string tmp))))
        (begin0
          (hash 'rows (dmatrix-nrow loaded)
                'cols (dmatrix-ncol loaded)
                'values (dmatrix->list loaded))
          (dmatrix-free! loaded)))
      (lambda ()
        (when (file-exists? tmp)
          (delete-file tmp)))))

  (define result
    (hash 'sliced-values sliced-values
          'loaded-summary loaded-summary))
  (dmatrix-free! sliced)
  (dmatrix-free! dm)
  result)

(module+ main
  (define result (run-example))
  (define loaded (hash-ref result 'loaded-summary))
  (printf "sliced values: ~a\n" (hash-ref result 'sliced-values))
  (printf "loaded binary shape: ~ax~a\n"
          (hash-ref loaded 'rows)
          (hash-ref loaded 'cols)))

(module+ test
  (require rackunit)

  (define expected '((5.0 6.0) (1.0 2.0)))
  (define result (run-example))
  (define loaded (hash-ref result 'loaded-summary))
  (check-true (same-matrix? expected (hash-ref result 'sliced-values)))
  (check-equal? (hash-ref loaded 'rows) 2)
  (check-equal? (hash-ref loaded 'cols) 2)
  (check-true (same-matrix? expected (hash-ref loaded 'values))))
