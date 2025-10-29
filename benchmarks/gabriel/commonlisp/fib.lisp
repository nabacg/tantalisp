;;; Fibonacci benchmark for Common Lisp

(defun fib (n)
  (declare (fixnum n))
  (if (< n 2)
      1
      (the fixnum (+ (fib (the fixnum (1- n)))
                     (fib (the fixnum (- n 2)))))))

(print (fib 30))
