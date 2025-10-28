;;; Fibonacci benchmark for Clojure

(defn fib [^long n]
  (if (< n 2)
    1
    (+ (fib (dec n)) (fib (- n 2)))))

(println (fib 30))
