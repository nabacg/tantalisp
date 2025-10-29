;;; TAK benchmark for Common Lisp (SBCL, CCL)

(defun tak (x y z)
  (declare (fixnum x y z))
  (if (< y x)
      (tak (tak (the fixnum (1- x)) y z)
           (tak (the fixnum (1- y)) z x)
           (tak (the fixnum (1- z)) x y))
      z))

(print (tak 18 12 6))
