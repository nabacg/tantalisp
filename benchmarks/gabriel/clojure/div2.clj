;;; DIV2 benchmark for Clojure

(defn div2-iter [^long n ^long acc]
  (if (<= n 1)
    acc
    (recur (- n 2) (+ acc 1))))

(defn div2 [^long n]
  (div2-iter n 0))

(println (div2 1000000))
