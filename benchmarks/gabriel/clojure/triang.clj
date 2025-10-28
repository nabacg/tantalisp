;;; TRIANG benchmark for Clojure

(defn tri [^long n]
  (if (= n 0)
    0
    (+ n (tri (dec n)))))

(println (tri 100))
