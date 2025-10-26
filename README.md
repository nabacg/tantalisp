# Tantalisp

A JIT-compiled Lisp interpreter written in Rust with LLVM backend via Inkwell.

## Features

- **Dual Execution Modes**
  - JIT compilation to native code (default, via LLVM)
  - Tree-walking interpreter (fallback mode)
- **Reference Counting Garbage Collection**
  - Automatic memory management for compiled code
  - GC monitoring and debugging support
- **Rich Language Features**
  - First-class functions and closures
  - Recursive functions
  - List operations (car, cdr, cons)
  - Standard library (prelude) with map, filter, reduce, etc.
- **Interactive REPL**
  - Command history with rustyline
  - Persistent variables across evaluations
  - Multiple input modes (interactive, file, piped)

## Installation

### Prerequisites

- Rust (edition 2024)
- LLVM 18.x
- Cargo

### Build

```bash
cargo build --release
```

### Run Tests

Due to LLVM thread-safety limitations, tests must be run single-threaded:

```bash
cargo test --lib -- --test-threads=1
```

## Usage

### Interactive REPL

Start an interactive session with command history:

```bash
cargo run
```

Example session:

```lisp
tantalisp> (def x 42)
42
tantalisp> (+ x 8)
50
tantalisp> (def fib (fn [n] (if (< n 2) 1 (+ (fib (- n 1)) (fib (- n 2))))))
#<lambda>
tantalisp> (fib 10)
89
tantalisp> (map (fn [x] (* x 2)) (range 5))
(0 2 4 6 8)
```

Exit with `Ctrl+C`, `Ctrl+D`, or `:quit`.

### File Execution

Run a Lisp file:

```bash
cargo run -- --file example.lisp
```

### Piped Input

Execute code from stdin:

```bash
echo "(+ 1 2)" | cargo run
```

## Command Line Flags

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--file <PATH>` | `-f` | Execute Lisp code from file | None (REPL mode) |
| `--debug` | `-d` | Enable debug output (tokens, AST, LLVM IR) | `false` |
| `--interpreter` | `-i` | Use tree-walking interpreter instead of JIT | `false` (JIT mode) |

### Examples

```bash
# Run file with debug output
cargo run -- --file example.lisp --debug

# Use interpreter mode instead of JIT
cargo run -- --interpreter

# Combine flags
cargo run -- --file test.lisp --debug --interpreter
```

## Environment Variables

### `GC_DEBUG=1` - Garbage Collector Monitoring

Enable GC debugging to monitor memory allocation and deallocation in real-time.

**Usage:**

```bash
GC_DEBUG=1 cargo run
```

When enabled:
- A background thread logs memory statistics every 100ms to `gc_debug.log`
- Shows total allocations, bytes allocated/deallocated, and memory leaks
- Useful for performance analysis and memory leak detection

**Log Format:**

```
[timestamp] allocs=<count> allocated=<bytes> deallocated=<bytes> leaked=<bytes>
```

**Example:**

```bash
# Run with GC monitoring
GC_DEBUG=1 cargo run

# In another terminal, watch stats in real-time
tail -f gc_debug.log
```

## Scripts

The `bin/` directory contains helpful scripts:

### `bin/test_all.sh` - Run Full Test Suite

Runs all tests with proper LLVM thread-safety flags:

```bash
./bin/test_all.sh
```

This automatically uses `--test-threads=1` to avoid LLVM-related crashes.

### `bin/run_perf_test.sh` - Performance Testing

Runs the performance test suite with GC monitoring and visualization:

```bash
./bin/run_perf_test.sh
```

This runs `perf_test.tlsp` which includes:
- Map/reduce operations on large ranges (up to 5,000 elements)
- Recursive fibonacci calculations
- Nested list operations
- Deep recursion tests

The script automatically:
- Builds in release mode for maximum performance
- Enables GC monitoring (logs to `gc_debug.log`)
- Times execution
- Reports memory leak statistics
- **Opens a gnuplot window** showing memory usage over time (if gnuplot is installed)

**Install gnuplot for visualization:**

```bash
brew install gnuplot
```

**Manual performance testing:**

```bash
GC_DEBUG=1 cargo run --release -- --file perf_test.tlsp
```

## Language Reference

### Data Types

- **Integers**: `42`, `-17`
- **Booleans**: `:true`, `:false`
- **Strings**: `"hello world"`
- **Lists**: `(1 2 3)`, `'(a b c)`
- **Nil**: `nil` or `'()`
- **Lambdas**: `(fn [x] (+ x 1))`

### Special Forms

| Form | Syntax | Description |
|------|--------|-------------|
| `def` | `(def name value)` | Define global variable |
| `fn` | `(fn [params] body)` | Create lambda function |
| `if` | `(if pred then else)` | Conditional expression |
| `quote` | `'expr` or `(quote expr)` | Quote expression |

### Builtin Operators

**Arithmetic:** `+`, `-`, `*`, `/`

**Comparison:** `=`, `!=`, `<`, `>`, `<=`, `>=`

**List Operations:**
- `(list ...)` - Create list
- `(car lst)` / `(head lst)` - Get first element
- `(cdr lst)` / `(tail lst)` - Get rest of list
- `(cons elem lst)` - Prepend element to list

### Standard Library (Prelude)

Automatically loaded on startup from `prelude.tlsp`:

- `(map fn lst)` - Apply function to each element
- `(filter pred lst)` - Keep elements matching predicate
- `(reduce fn acc lst)` - Fold list with accumulator
- `(length lst)` - Get list length
- `(sum lst)` - Sum all numbers in list
- `(reverse lst)` - Reverse list
- `(append xs ys)` - Concatenate lists
- `(take n lst)` - Take first n elements
- `(drop n lst)` - Drop first n elements
- `(any pred lst)` - Test if any element matches
- `(all pred lst)` - Test if all elements match
- `(range n)` - Generate list [0..n)

## Project Structure

```
tantalisp/
├── src/
│   ├── main.rs              # CLI entry point
│   ├── lib.rs               # REPL implementation
│   ├── lexer.rs             # Tokenization
│   ├── parser.rs            # S-expression parsing
│   ├── evaluator.rs         # Tree-walking interpreter
│   └── codegen/
│       ├── mod.rs           # LLVM JIT compiler
│       └── runtime/
│           ├── mod.rs       # Runtime support functions
│           └── garbage_collector.rs  # Reference counting GC
├── prelude.tlsp             # Standard library
├── perf_test.tlsp           # Performance test suite
├── run_perf_test.sh         # Performance test runner
└── CLAUDE.md                # Detailed implementation notes
```

## Architecture

### JIT Compilation Pipeline

```
Source Code → Lexer → Parser → CodeGen → LLVM IR → JIT → Native Code
```

### Type System

All values are represented as tagged unions (`LispVal`) with reference counting:

```rust
struct LispVal {
    tag: u8,        // Type discriminator
    refcount: i32,  // Reference count for GC
    data: i64,      // Value or pointer
}
```

**Tags:**
- `0` - Integer
- `1` - Boolean
- `2` - String
- `3` - List (cons cell)
- `4` - Nil
- `5` - Lambda (function pointer)

### Memory Management

- **Compiled Mode**: Reference counting with automatic cleanup
- **Interpreted Mode**: Rust's ownership system
- **Global Variables**: Stored in `HashMap<String, *mut LispVal>` in runtime

## Known Issues

- Tests must run with `--test-threads=1` due to LLVM thread-safety
- Piped input mode doesn't load prelude (use file mode for prelude functions)
- No tail call optimization yet
- Segfault on undefined function calls (needs better error handling)

## Development

### Debug LLVM IR

Enable debug mode to see generated LLVM IR:

```bash
cargo run -- --debug --file example.lisp
```

### Memory Leak Analysis

Check for memory leaks with GC monitoring:

```bash
# Run program with GC debug
GC_DEBUG=1 cargo run

# Analyze final leak
tail -1 gc_debug.log
```

Typical leak is ~20KB (0.02%) from LLVM infrastructure - this is expected and acceptable.

## Examples

### Fibonacci

```lisp
(def fib (fn [n]
  (if (< n 2)
    1
    (+ (fib (- n 1)) (fib (- n 2))))))

(fib 20)  ; => 10946
```

### List Processing

```lisp
; Sum of squares of even numbers in range [0, 10)
(reduce +
  0
  (map (fn [x] (* x x))
    (filter (fn [x] (= (% x 2) 0))
      (range 10))))
```

### Higher-Order Functions

```lisp
(def compose (fn [f g]
  (fn [x] (f (g x)))))

(def add1 (fn [x] (+ x 1)))
(def double (fn [x] (* x 2)))

(def add1-then-double (compose double add1))

(add1-then-double 5)  ; => 12
```

## Contributing

This is an educational project exploring JIT compilation and garbage collection in Lisp.

## License

See LICENSE file for details.

## References

- [Inkwell Documentation](https://thedan64.github.io/inkwell/)
- [LLVM Language Reference](https://llvm.org/docs/LangRef.html)
- [Crafting Interpreters](https://craftinginterpreters.com/)
