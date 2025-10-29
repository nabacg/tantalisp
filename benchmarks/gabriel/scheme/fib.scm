#!/usr/bin/env racket
#lang racket

(define (fib n)
  (if (< n 2)
      1
      (+ (fib (- n 1)) (fib (- n 2)))))

(displayln (fib 30))
