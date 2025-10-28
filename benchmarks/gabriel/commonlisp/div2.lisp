;;; DIV2 benchmark for Common Lisp (SBCL, CCL)

(defun div2-iter (n acc)
  (declare (fixnum n acc))
  (if (<= n 1)
      acc
      (div2-iter (- n 2) (+ acc 1))))

(defun div2 (n)
  (declare (fixnum n))
  (div2-iter n 0))

(print (div2 1000000))
