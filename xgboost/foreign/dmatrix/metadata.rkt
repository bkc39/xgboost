#lang racket/base

;; DMatrix metadata: setting and getting float / uint / string-feature info
;; fields (label, weight, base_margin, group, feature_name, feature_type, …).

(require ffi/unsafe
         ffi/vector
         "../array-interface.rkt"
         "../error.rkt"
         "../raw/dmatrix.rkt")

(provide dmatrix-set-float-info!
         dmatrix-set-uint-info!
         dmatrix-set-info-from-interface!
         dmatrix-set-feature-info!
         dmatrix-get-float-info
         dmatrix-get-uint-info
         dmatrix-get-feature-info)

(define (dmatrix-set-float-info! h field vals)
  (check-ok (xgb-dmatrix-set-float-info/raw h field vals)
            'dmatrix-set-float-info!))

(define (dmatrix-set-uint-info! h field vals)
  (check-ok (xgb-dmatrix-set-info-from-interface/raw
             h field (u32-array-interface vals))
            'dmatrix-set-uint-info!))

(define (dmatrix-set-info-from-interface! h field data-json)
  (check-ok (xgb-dmatrix-set-info-from-interface/raw h field data-json)
            'dmatrix-set-info-from-interface!))

(define (dmatrix-set-feature-info! h field vals)
  (check-ok (xgb-dmatrix-set-str-feature-info/raw h field vals)
            'dmatrix-set-feature-info!))

;; Copy a float info field into a fresh f32vector.  The pointer returned by
;; the raw call is borrowed from the DMatrix; we memcpy out of it before
;; letting control flow near anything that could invalidate it.  An unset
;; field returns an empty f32vector (rc is still 0).
(define (dmatrix-get-float-info h field)
  (define-values (rc len ptr) (xgb-dmatrix-get-float-info/raw h field))
  (check-ok rc 'dmatrix-get-float-info)
  (define result (make-f32vector len))
  (when (and ptr (> len 0))
    (memcpy (f32vector->cpointer result) ptr len _float))
  result)

(define (dmatrix-get-uint-info h field)
  (define-values (rc len ptr) (xgb-dmatrix-get-uint-info/raw h field))
  (check-ok rc 'dmatrix-get-uint-info)
  (define result (make-u32vector len))
  (when (and ptr (> len 0))
    (memcpy (u32vector->cpointer result) ptr len _uint32))
  result)

(define (dmatrix-get-feature-info h field)
  (define-values (rc len _count)
    (xgb-dmatrix-get-str-feature-info/raw h field 0 (make-bytes 0)))
  (cond
    [(zero? rc) '()]
    [(= rc 2)
     (define buf (make-bytes len))
     (define-values (rc2 len2 count2)
       (xgb-dmatrix-get-str-feature-info/raw h field len buf))
     (check-ok rc2 'dmatrix-get-feature-info)
     (unless (= len2 len)
       (error 'dmatrix-get-feature-info
              "expected ~a bytes, got ~a" len len2))
     (nul-separated-bytes->strings buf count2)]
    [else (check-ok rc 'dmatrix-get-feature-info)]))
