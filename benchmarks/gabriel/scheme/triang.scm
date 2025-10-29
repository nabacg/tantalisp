#!/usr/bin/env racket
#lang racket

(define (tri n)
  (if (= n 0)
      0
      (+ n (tri (- n 1)))))

(displayln (tri 100))
