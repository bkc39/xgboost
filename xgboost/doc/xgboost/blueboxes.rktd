937
((3) 0 () 1 ((q lib "xgboost/main.rkt")) () (h ! (equal) ((c def c (c (? . 0) q load-model-from-bytes)) q (2745 . 3)) ((c def c (c (? . 0) q predict)) q (1537 . 14)) ((c def c (c (? . 0) q xgboost-register-log-callback!)) q (3036 . 3)) ((c def c (c (? . 0) q dmatrix?)) q (0 . 3)) ((c def c (c (? . 0) q parse-eval-line)) q (2304 . 3)) ((c def c (c (? . 0) q xgboost-build-info)) q (2860 . 2)) ((c def c (c (? . 0) q load-model)) q (2489 . 3)) ((c def c (c (? . 0) q xgboost-version)) q (2819 . 2)) ((c def c (c (? . 0) q booster?)) q (594 . 3)) ((c def c (c (? . 0) q train)) q (648 . 21)) ((c def c (c (? . 0) q xgboost-set-global-config!)) q (2955 . 3)) ((c def c (c (? . 0) q save-model)) q (2389 . 4)) ((c def c (c (? . 0) q xgboost-get-global-config)) q (2904 . 2)) ((c def c (c (? . 0) q eval-one-iter)) q (2138 . 5)) ((c def c (c (? . 0) q save-model-to-bytes)) q (2558 . 5)) ((c def c (c (? . 0) q make-dmatrix)) q (54 . 13))))
procedure
(dmatrix? v) -> boolean?
  v : any/c
procedure
(make-dmatrix  data                    
              [#:nrow nrow             
               #:ncol ncol             
               #:missing missing       
               #:labels labels         
               #:weights weights]) -> dmatrix?
  data : any/c
  nrow : (or/c #f exact-positive-integer?) = #f
  ncol : (or/c #f exact-positive-integer?) = #f
  missing : real? = +nan.0
  labels : (or/c #f any/c) = #f
  weights : (or/c #f any/c) = #f
procedure
(booster? v) -> boolean?
  v : any/c
procedure
(train  dtrain                        
       [#:params params               
        #:rounds rounds               
        #:evals evals                 
        #:objective objective         
        #:eta eta                     
        #:max-depth max-depth         
        #:num-class num-class         
        #:eval-metric eval-metric     
        #:verbosity verbosity])   -> booster?
  dtrain : dmatrix?
  params : any/c = '()
  rounds : exact-nonnegative-integer? = 10
  evals : (listof (cons/c string? dmatrix?)) = '()
  objective : (or/c #f any/c) = #f
  eta : (or/c #f any/c) = #f
  max-depth : (or/c #f any/c) = #f
  num-class : (or/c #f any/c) = #f
  eval-metric : (or/c #f any/c) = #f
  verbosity : (or/c #f any/c) = #f
procedure
(predict  booster                       
          dmat                          
         [#:output output               
          #:iteration-end iteration-end 
          #:as as])                     
 -> (or/c (listof real?) f32vector?)
  booster : booster?
  dmat : dmatrix?
  output : (or/c 'value 'margin 'contribs        = 'value
                  'approx-contribs 'interactions
                  'approx-interactions 'leaf)
  iteration-end : exact-nonnegative-integer? = 0
  as : (or/c 'list 'f32vector) = 'list
procedure
(eval-one-iter booster iter evals) -> string?
  booster : booster?
  iter : exact-integer?
  evals : (listof (cons/c string? dmatrix?))
procedure
(parse-eval-line line) -> (hash/c string? real?)
  line : string?
procedure
(save-model booster path) -> void?
  booster : booster?
  path : path-string?
procedure
(load-model path) -> booster?
  path : path-string?
procedure
(save-model-to-bytes  booster               
                     [#:format format]) -> bytes?
  booster : booster?
  format : (or/c "json" "ubj") = "ubj"
procedure
(load-model-from-bytes data) -> booster?
  data : bytes?
procedure
(xgboost-version) -> string?
procedure
(xgboost-build-info) -> string?
procedure
(xgboost-get-global-config) -> string?
procedure
(xgboost-set-global-config! config) -> void?
  config : string?
procedure
(xgboost-register-log-callback! callback) -> void?
  callback : (-> string? any/c)
