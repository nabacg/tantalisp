;;; TAKL benchmark for Clojure

(defn listn [^long n]
  (if (= n 0)
    '()
    (cons n (listn (dec n)))))

(defn shorterp [x y]
  (if (seq y)
    (if (seq x)
      (shorterp (rest x) (rest y))
      true)
    false))

(defn mas [x y z]
  (if (shorterp y x)
    (mas (mas (rest x) y z)
         (mas (rest y) z x)
         (mas (rest z) x y))
    z))

(let [l18 (listn 18)
      l12 (listn 12)
      l6 (listn 6)]
  (println (mas l18 l12 l6)))
