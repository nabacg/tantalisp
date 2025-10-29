# Gabriel Benchmark Results

## Quick Summary

✅ **Successfully implemented** 5 Gabriel benchmarks:
- TAK (Takeuchi Function)
- TAKL (Takeuchi with Lists)
- DIV2 (Division by 2)
- TRIANG (Triangle Numbers)
- FIB (Fibonacci)

## Running Benchmarks

**First, build Tantalisp in release mode:**

```bash
cargo build --release
```

**Then run the benchmarks:**

```bash
cd benchmarks
./run_gabriel.sh
```

**Important:** The benchmark script requires a pre-built release binary to avoid including compilation time in measurements. It will exit with an error if `target/release/tantalisp` is not found.

## Current Results (Sample Run)

| Implementation | TAK | TAKL | DIV2 | TRIANG | FIB |
|---------------|-----|------|------|--------|-----|
| Tantalisp (JIT) | ~600ms | ~2s | ~90ms | ~12ms | ~350ms |
| Racket | TBD | TBD | TBD | TBD | TBD |
| Chicken (Compiled) | TBD | TBD | TBD | TBD | TBD |
| Clojure | TBD | TBD | TBD | TBD | TBD |

*Note: First run includes JIT compilation overhead. Times shown are averages of 5 iterations.*

## Implementation Status

### ✅ Implemented Benchmarks

All 5 benchmarks work correctly with Tantalisp's current feature set:

- **Integers**: ✅
- **Arithmetic**: ✅ (`+`, `-`, `*`, `/`)
- **Comparison**: ✅ (`<`, `>`, `<=`, `>=`, `=`)
- **Lists**: ✅ (`cons`, `car`, `cdr`, `nil`)
- **Recursion**: ✅ (Full support)
- **Lambdas**: ✅ (First-class functions)

### ❌ Not Yet Implemented

Gabriel benchmarks we **cannot** implement yet:

- **DERIV** (Symbolic Differentiation) - Needs: symbols, `eq?`, property lists
- **BROWSE** (Database Traversal) - Needs: symbols, `member`, `assoc`
- **DESTRUCTIVE** - Needs: mutable list operations (`rplaca`, `rplacd`)
- **PUZZLE** - Needs: vectors/arrays, mutation
- **STAK** - Needs: dynamic variables
- **CTAK** - Needs: continuation-passing style / `call/cc`
- **BOYER** - Needs: symbols, pattern matching, many CL features
- **FFT** - Needs: floats, arrays, complex numbers

## Performance Notes

### What the Benchmarks Test

1. **TAK**: Function call overhead, deep recursion, integer arithmetic
2. **TAKL**: List allocation, GC performance, recursive data structures
3. **DIV2**: Tail recursion, simple iteration
4. **TRIANG**: Basic recursion, arithmetic
5. **FIB**: Binary recursion tree, function calls

### Expected Performance Characteristics

- **TAK/FIB**: JIT compilation should make these very fast
- **TAKL**: May show RefCount GC overhead (especially with lambda capture leak)
- **DIV2/TRIANG**: Should be extremely fast (simple arithmetic)

### Known Performance Issues

1. **Lambda Variable Capture Leak** (documented in CLAUDE.md):
   - Variables accessed in lambdas get incref'd but not decref'd on lambda exit
   - Causes memory leaks proportional to call count
   - Affects TAKL performance

2. **Prelude Loading Overhead**:
   - Standard library loaded on every REPL startup
   - Adds ~50-100ms to first benchmark

## Comparison Setup

### Install Comparison Lisps

```bash
# macOS
brew install sbcl racket chicken clojure

# Ubuntu/Debian
sudo apt install sbcl racket chicken-bin clojure

# Fedora
sudo dnf install sbcl racket chicken clojure
```

### Why These Implementations?

- **SBCL**: Industry-standard compiled Common Lisp, heavily optimized
- **Racket**: Modern Scheme, JIT compiled
- **Chicken**: Scheme-to-C compiler, produces fast native code
- **Clojure**: JVM-based Lisp, good for comparison despite startup overhead

## Future Work

To implement more Gabriel benchmarks, we need:

1. **Symbol type** (not just strings)
2. **Floating-point numbers**
3. **Vectors/arrays**
4. **Mutable data structures**
5. **Property lists / associative data**
6. **More complete standard library**

See main `CLAUDE.md` for implementation roadmap.
