# Tantalisp Standard Library (Prelude)

The prelude is automatically loaded when starting the REPL. It provides common functional programming utilities.

## List Processing Functions

### `map`
Apply a function to each element of a list.

**Usage**: `(map (fn [x] (+ x 1)) (list 1 2 3))` → `(2 3 4)`

### `filter`
Keep only elements that satisfy a predicate.

**Usage**: `(filter (fn [x] (> x 2)) (list 1 2 3 4))` → `(3 4)`

### `reduce`
Fold a list from the left with an accumulator.

**Usage**: `(reduce + 0 (list 1 2 3 4))` → `10`

### `append`
Concatenate two lists.

**Usage**: `(append (list 1 2) (list 3 4))` → `(1 2 3 4)`

### `reverse`
Reverse a list.

**Usage**: `(reverse (list 1 2 3))` → `(3 2 1)`

## List Query Functions

### `length`
Count the number of elements in a list.

**Usage**: `(length (list 1 2 3))` → `3`

### `sum`
Add all numbers in a list.

**Usage**: `(sum (list 1 2 3 4))` → `10`

### `any`
Check if any element satisfies a predicate.

**Usage**: `(any (fn [x] (> x 3)) (list 1 2 3 4))` → `:true`

### `all`
Check if all elements satisfy a predicate.

**Usage**: `(all (fn [x] (> x 0)) (list 1 2 3))` → `:true`

## List Slicing Functions

### `take`
Take the first n elements from a list.

**Usage**: `(take 2 (list 1 2 3 4))` → `(1 2)`

### `drop`
Drop the first n elements from a list.

**Usage**: `(drop 2 (list 1 2 3 4))` → `(3 4)`

## List Generation Functions

### `range`
Generate a list of numbers from 0 to n (exclusive).

**Usage**: `(range 5)` → `(0 1 2 3 4)`

**Usage**: `(range 10)` → `(0 1 2 3 4 5 6 7 8 9)`

## Examples

```lisp
# Generate a range of numbers
(range 0 10)
# → (0 1 2 3 4 5 6 7 8 9)

# Double all numbers
(map (fn [x] (* x 2)) (list 1 2 3 4 5))
# → (2 4 6 8 10)

# Double numbers in a range
(map (fn [x] (* x 2)) (range 5))
# → (0 2 4 6 8)

# Get only positive numbers
(filter (fn [x] (> x 0)) (list -2 -1 0 1 2))
# → (1 2)

# Calculate factorial using reduce
(reduce * 1 (list 1 2 3 4 5))
# → 120

# Check if all numbers are even
(all (fn [x] (= 0 (- x (* 2 (/ x 2))))) (list 2 4 6))
# → :true

# Complex example: sum of squares of even numbers
(sum (map (fn [x] (* x x)) (filter (fn [x] (= 0 (- x (* 2 (/ x 2))))) (list 1 2 3 4 5))))
# → 20  (2² + 4² = 4 + 16 = 20)
```
