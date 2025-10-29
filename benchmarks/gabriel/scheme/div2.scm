#!/usr/bin/env racket
#lang racket

(define (div2-iter n acc)
  (if (<= n 1)
      acc
      (div2-iter (- n 2) (+ acc 1))))

(define (div2 n)
  (div2-iter n 0))

(displayln (div2 1000000))
