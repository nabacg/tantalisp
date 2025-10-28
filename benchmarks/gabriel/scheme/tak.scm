#!/usr/bin/env racket
#lang racket

;;; TAK benchmark for Scheme (Racket, Chicken, Guile)

(define (tak x y z)
  (if (< y x)
      (tak (tak (- x 1) y z)
           (tak (- y 1) z x)
           (tak (- z 1) x y))
      z))

(displayln (tak 18 12 6))
