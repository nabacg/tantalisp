;;; TAK benchmark for Clojure

(defn tak [^long x ^long y ^long z]
  (if (< y x)
    (tak (tak (dec x) y z)
         (tak (dec y) z x)
         (tak (dec z) x y))
    z))

(println (tak 18 12 6))
