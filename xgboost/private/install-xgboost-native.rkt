#lang racket/base

(provide pre-installer)

(require racket/file)

(define (copy-native-libs! dest-dir source-dir pattern)
  (make-directory* dest-dir)
  (for ([f (in-list (directory-list source-dir))])
    (when (regexp-match? pattern (path->string f))
      (define src (build-path source-dir f))
      (define dst (build-path dest-dir f))
      (when (file-exists? dst)
        (delete-file dst))
      (copy-file src dst))))

(define (has-matching-files? dir pattern)
  (and (directory-exists? dir)
       (pair? (filter (lambda (f) (regexp-match? pattern (path->string f)))
                      (directory-list dir)))))

(define (pre-installer collections-top-path this-collection-path user-specific?)
  (define native-libs-dir (build-path this-collection-path "native-libs"))
  (define cpp-lib-path (getenv "XGBOOST_NATIVE_LIB_PATH"))
  (cond
    [cpp-lib-path
     (copy-native-libs! native-libs-dir (build-path cpp-lib-path "lib")
                        #rx"^libxgbcompat\\.")]
    [(has-matching-files? native-libs-dir #rx"^libxgbcompat\\.")
     (void)]
    [else
     (error 'pre-installer
            "xgbcompat library not found. Either:\n  1. Run: nix run .#copy-native-libs\n  2. Build manually and copy to ~a"
            (path->string native-libs-dir))]))
