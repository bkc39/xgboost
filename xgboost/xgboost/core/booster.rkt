#lang racket/base

;; High-level Booster inspection: shape queries, lifecycle (slice / reset),
;; attributes, feature names/types, JSON config, model dumps, and feature
;; importance scores.

(require (prefix-in foreign: "../foreign.rkt"))

(provide booster-num-feature
         booster-boosted-rounds
         booster-reset!
         booster-slice
         booster-set-attr!
         booster-attr
         booster-attr-names
         booster-delete-attr!
         booster-set-feature-names!
         booster-set-feature-types!
         booster-feature-names
         booster-feature-types
         booster-config
         booster-set-config!
         booster-dump
         booster-feature-score)

(define (booster-num-feature b)
  (foreign:booster-num-feature b))

(define (booster-boosted-rounds b)
  (foreign:booster-boosted-rounds b))

(define (booster-reset! b)
  (foreign:booster-reset! b))

(define (booster-slice b begin-layer end-layer [step 1])
  (foreign:booster-slice b begin-layer end-layer step))

(define (booster-set-attr! b key value)
  (foreign:booster-set-attr! b key value))

(define (booster-attr b key)
  (foreign:booster-get-attr b key))

(define (booster-attr-names b)
  (foreign:booster-get-attr-names b))

(define (booster-delete-attr! b key)
  (foreign:booster-delete-attr! b key))

(define (booster-set-feature-names! b names)
  (foreign:booster-set-feature-info! b "feature_name" names))

(define (booster-set-feature-types! b types)
  (foreign:booster-set-feature-info! b "feature_type" types))

(define (booster-feature-names b)
  (foreign:booster-get-feature-info b "feature_name"))

(define (booster-feature-types b)
  (foreign:booster-get-feature-info b "feature_type"))

(define (booster-config b)
  (foreign:booster-save-json-config b))

(define (booster-set-config! b cfg)
  (foreign:booster-load-json-config! b cfg))

(define (booster-dump b
                     #:format [fmt "text"]
                     #:with-stats? [stats? #f]
                     #:feature-names [names #f]
                     #:feature-types [types #f])
  (cond
    [(or names types)
     (unless (and names types)
       (error 'booster-dump
              "#:feature-names and #:feature-types must be provided together"))
     (foreign:booster-dump-model-with-features b names types
                                               #:format fmt
                                               #:with-stats? stats?)]
    [else
     (foreign:booster-dump-model b
                                 #:format fmt
                                 #:with-stats? stats?)]))

(define (booster-feature-score b
                               #:importance-type [importance-type "weight"]
                               #:feature-names [names #f]
                               #:config [config #f])
  (foreign:booster-feature-score b
                                 #:importance-type importance-type
                                 #:feature-names names
                                 #:config config))
