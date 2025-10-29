;;; TAKL benchmark for Common Lisp (SBCL, CCL)

(defun listn (n)
  (declare (fixnum n))
  (if (= n 0)
      nil
      (cons n (listn (1- n)))))

(defun shorterp (x y)
  (declare (list x y))
  (and y (or (null x)
             (shorterp (cdr x) (cdr y)))))

(defun mas (x y z)
  (declare (list x y z))
  (if (shorterp y x)
      (mas (mas (cdr x) y z)
           (mas (cdr y) z x)
           (mas (cdr z) x y))
      z))

(let ((l18 (listn 18))
      (l12 (listn 12))
      (l6 (listn 6)))
  (print (mas l18 l12 l6)))
