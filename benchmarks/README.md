# Tantalisp Benchmarks

This directory contains benchmark suites for comparing Tantalisp's performance against other Lisp implementations.

## Gabriel Benchmarks

The Gabriel benchmarks are a classic suite from Richard Gabriel's 1985 book "Performance and Evaluation of Lisp Systems". We've implemented a subset that works with Tantalisp's current feature set.

### Benchmarks Included

1. **TAK (Takeuchi Function)** - Classic triply-recursive integer benchmark
   - Tests: Function call overhead, recursion depth, integer arithmetic
   - Standard test: `(tak 18 12 6)`

2. **TAKL (Takeuchi with Lists)** - TAK variant using list lengths instead of integers
   - Tests: List operations, cons cell allocation, recursion with complex data
   - Standard test: `(mas (listn 18) (listn 12) (listn 6))`

3. **DIV2 (Division by 2)** - Simple iteration benchmark
   - Tests: Tail recursion, integer arithmetic, iteration performance
   - Standard test: `(div2 1000000)`

4. **TRIANG (Triangle Numbers)** - Sum of integers from 1 to N
   - Tests: Simple recursion, arithmetic operations
   - Standard test: `(tri 100)`

5. **FIB (Fibonacci)** - Classic doubly-recursive benchmark
   - Tests: Binary recursion, function call overhead
   - Standard test: `(fib 30)`

### Running the Benchmarks

#### Quick Start

**First, build Tantalisp in release mode:**

```bash
cargo build --release
```

**Then run the benchmarks:**

```bash
cd benchmarks
./run_gabriel.sh
```

**For faster development/testing (Tantalisp only):**

```bash
cd benchmarks
./run_gabriel.sh --tantalisp-only
```

This will:
- Automatically detect available Lisp implementations (or skip them with `--tantalisp-only`)
- Check that Tantalisp is built in release mode (exits if not)
- Run each benchmark 5 times and average the results
- Generate a CSV file with results
- Display a summary table
- Create a performance chart (if gnuplot is available)

**Important:** The benchmark script requires a pre-built release binary to avoid including compilation time in measurements.

**Tip:** Use `--tantalisp-only` during development to get quick feedback. Run full comparisons periodically to see how Tantalisp compares to other implementations.

#### Requirements

**Mandatory:**
- Cargo/Rust (for Tantalisp)

**Optional** (for comparison - script auto-detects):
- SBCL (Steel Bank Common Lisp)
- Racket
- Chicken Scheme
- Clojure

Install comparison Lisps:

```bash
# macOS (Homebrew)
brew install sbcl racket chicken clojure

# Ubuntu/Debian
sudo apt install sbcl racket chicken-bin clojure

# Fedora
sudo dnf install sbcl racket chicken clojure
```

#### Output

The script produces:

1. **`gabriel_results.csv`** - Raw timing data in CSV format
2. **`gabriel_chart.png`** - Performance comparison chart (if gnuplot available)
3. **Console output** - Real-time progress and summary table

Example output:

```
========================================
   Gabriel Benchmarks Runner
   TAK, TAKL, DIV2, TRIANG, FIB
========================================

Checking available Lisp implementations:

✓ Tantalisp found
✓ SBCL found
✓ Racket found
✗ Chicken Scheme not found (skipping)
✓ Clojure found

========================================
Tantalisp (JIT Compiled)
========================================

Running: Tantalisp - TAK
  Average: 245ms (5 iterations)
  Result: 7

...

Summary Table:
Implementation  Benchmark  Time(ms)  Result
Tantalisp       TAK        245       7
Tantalisp       TAKL       1523      (6)
Tantalisp       DIV2       89        500000
Tantalisp       TRIANG     12        5050
Tantalisp       FIB        342       1346269
SBCL            TAK        156       7
...
```

### Understanding the Results

- **Lower times are better**
- **TAK** and **FIB** are recursion-heavy - JIT compilation shines here
- **TAKL** stresses the garbage collector - RefCount GC overhead may show
- **DIV2** and **TRIANG** are arithmetic-heavy - should be very fast
- **Result** column verifies correctness (all implementations should produce same result)

### Expected Performance

Based on typical JIT compiler performance:

- **vs. Interpreted Lisps**: Tantalisp should be **10-100x faster**
- **vs. Compiled Schemes** (Chicken, Gambit): Should be **competitive** (within 2-3x)
- **vs. SBCL**: SBCL is highly optimized, may be **faster** on some benchmarks
- **vs. Clojure**: Should be **significantly faster** (Clojure has JVM startup overhead)
- **vs. Racket**: Should be **competitive to faster**

### Benchmark Files Structure

```
benchmarks/
├── README.md                      # This file
├── run_gabriel.sh                 # Main benchmark runner
└── gabriel/
    ├── *.tlsp                     # Tantalisp implementations
    ├── scheme/*.scm               # Scheme implementations
    ├── commonlisp/*.lisp          # Common Lisp implementations
    └── clojure/*.clj              # Clojure implementations
```

### Customization

You can modify the benchmark parameters:

1. **Change iteration count** (for more stable averages):
   ```bash
   # Edit ITERATIONS=5 in run_gabriel.sh
   ITERATIONS=10
   ```

2. **Run specific implementations only**:
   ```bash
   # Edit the HAVE_* variables in run_gabriel.sh
   HAVE_RACKET=false  # Skip Racket
   ```

3. **Adjust benchmark parameters** (e.g., `fib 35` instead of `fib 30`):
   - Edit the `*.tlsp` files in `gabriel/` directory

### Adding New Benchmarks

To add a new benchmark:

1. Create `benchmarks/gabriel/mybench.tlsp`
2. Create equivalent files in `scheme/`, `commonlisp/`, `clojure/`
3. Add benchmark calls to `run_gabriel.sh`:
   ```bash
   run_benchmark "Tantalisp" "MYBENCH" \
       "$PROJECT_ROOT/target/release/tantalisp --file $GABRIEL_DIR/mybench.tlsp"
   ```

### Known Limitations

- **TAKL** may show RefCount GC overhead (lambda variable capture leak)
- Clojure has significant JVM startup overhead (~1-2 seconds)
- First run may include compilation time for some implementations

### Contributing

Found a bug or want to add more benchmarks? See the main CLAUDE.md file for development notes.

## Future Benchmarks

We're limited by missing language features. When implemented, we could add:

- **DERIV** (symbolic differentiation) - needs symbol type
- **BROWSE** (database traversal) - needs symbols, property lists
- **BOYER** (theorem prover) - needs many features
- **FFT** (Fast Fourier Transform) - needs arrays, floats
- **PUZZLE** (combinatorial search) - needs vectors

See CLAUDE.md TODO section for implementation priorities.
