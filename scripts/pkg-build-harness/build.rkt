#lang racket/base

;; Faithful reproduction of the Racket package-build service
;; (pkg-build.racket-lang.org) for the `xgboost` package, using the *same*
;; Docker images and the official `pkg-build` harness. This is what finally
;; surfaces the test transcript the public build server failed to write.
;;
;; Adapted from racket/pkg-build's examples/docker/build.rkt, narrowed to a
;; single package via `#:only-packages`.
;;
;; Knobs (environment variables):
;;   XGB_CATALOG  package catalog to build from.
;;                Default: the public catalog (builds xgboost@master — i.e.
;;                reproduces the currently published failure).
;;                Point it at a local file:// catalog to test a branch.
;;   XGB_VERS     Racket release to install in the VM (default "9.0").
;;
;; Run: racket scripts/pkg-build-harness/build.rkt
;; Results land under workdir/server/built/{test-fail,test-success,install}/.

(require pkg-build
         racket/format)

;; Don't run as a test (this module shells out to Docker):
(module test racket/base)

(define vers (or (getenv "XGB_VERS") "9.0"))
(define catalog (or (getenv "XGB_CATALOG") "https://pkgs.racket-lang.org/"))

;; Match the build server's 1 GB container cap (×2 counting swap):
(define memory-mb 1024)

;; The non-minimal deps image carries xvfb; run programs under it so any GUI
;; deps work, exactly as the server does.
(define xvfb-shell '("/usr/bin/xvfb-run" "--auto-servernum" "/bin/sh" "-c"))

(define (make-vm name)
  (docker-vm
   #:name name
   #:from-image "racket/pkg-build:deps-x86_64"
   #:shell xvfb-shell
   #:memory-mb memory-mb
   #:minimal-variant (docker-vm #:name (string-append name "-min")
                                #:from-image "racket/pkg-build:deps-min-x86_64"
                                #:memory-mb memory-mb)))

(module+ main
  (printf "pkg-build harness: catalog=~a racket=~a\n" catalog vers)
  (build-pkgs
   #:work-dir "workdir"
   #:snapshot-url (~a "https://mirror.racket-lang.org/releases/" vers "/")
   #:installer-name (~a "racket-" vers "-x86_64-linux-natipkg-pkg-build.sh")
   #:compile-any? #t
   #:pkgs-for-version vers
   #:pkg-catalogs (list catalog)
   #:only-packages '("xgboost")
   #:timeout 2400
   #:vms (list (make-vm "pkg-build"))
   #:steps (steps-in 'download 'summary)))
