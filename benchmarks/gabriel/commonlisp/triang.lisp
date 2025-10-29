;;; TRIANG benchmark for Common Lisp (SBCL, CCL)

(defun tri (n)
  (declare (fixnum n))
  (if (= n 0)
      0
      (+ n (tri (1- n)))))

(print (tri 100))
