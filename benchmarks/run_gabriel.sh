#!/bin/bash

# Gabriel Benchmark Runner
# Runs TAK, TAKL, DIV2, and TRIANG benchmarks across multiple Lisp implementations
#
# Usage: ./run_gabriel.sh [--tantalisp-only]
#   --tantalisp-only    Only run Tantalisp benchmarks (skip other languages)

set -e

# Parse command line arguments
TANTALISP_ONLY=false
if [ "$1" = "--tantalisp-only" ]; then
    TANTALISP_ONLY=true
fi

BENCHMARKS_DIR="$(cd "$(dirname "$0")" && pwd)"
GABRIEL_DIR="$BENCHMARKS_DIR/gabriel"
PROJECT_ROOT="$(dirname "$BENCHMARKS_DIR")"

# Number of iterations for averaging
ITERATIONS=5

# Results file
RESULTS_FILE="$BENCHMARKS_DIR/gabriel_results.csv"

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "========================================"
echo "   Gabriel Benchmarks Runner"
echo "   TAK, TAKL, DIV2, TRIANG, FIB"
echo "========================================"
echo ""

# Check which Lisp implementations are available
check_implementation() {
    local impl=$1
    local cmd=$2
    if command -v $cmd &> /dev/null; then
        echo -e "${GREEN}✓${NC} $impl found"
        return 0
    else
        echo -e "${YELLOW}✗${NC} $impl not found (skipping)"
        return 1
    fi
}

echo "Checking available Lisp implementations:"
echo ""

HAVE_TANTALISP=true
HAVE_SBCL=false
HAVE_RACKET=false
HAVE_CHICKEN=false
HAVE_CLOJURE=false

check_implementation "Tantalisp" "cargo" || HAVE_TANTALISP=false

if [ "$TANTALISP_ONLY" = true ]; then
    echo -e "${YELLOW}--tantalisp-only${NC} mode: Skipping other implementations"
    HAVE_SBCL=false
    HAVE_RACKET=false
    HAVE_CHICKEN=false
    HAVE_CLOJURE=false
else
    check_implementation "SBCL" "sbcl" && HAVE_SBCL=true
    check_implementation "Racket" "racket" && HAVE_RACKET=true
    # Skip Chicken - csc conflicts with C# compiler
    # check_implementation "Chicken Scheme" "csc" && HAVE_CHICKEN=true
    echo -e "${YELLOW}✗${NC} Chicken Scheme skipped (csc conflicts with C# compiler)"
    HAVE_CHICKEN=false
    check_implementation "Clojure" "clojure" && HAVE_CLOJURE=true
fi

echo ""
echo "========================================"
echo ""

# Initialize results file
echo "Implementation,Benchmark,Time(ms),Result" > "$RESULTS_FILE"

# Function to run a benchmark multiple times and get average
run_benchmark() {
    local impl=$1
    local bench_name=$2
    local cmd=$3

    echo -e "${BLUE}Running:${NC} $impl - $bench_name"

    local total_time=0
    local result=""
    local failed=false

    for i in $(seq 1 $ITERATIONS); do
        # Run in subshell with timeout and error handling
        (
            set +e  # Don't exit on error in subshell
            if command -v gdate &> /dev/null; then
                local start=$(gdate +%s%N)
                output=$(timeout 30 bash -c "$cmd" 2>&1)
                local exit_code=$?
                local end=$(gdate +%s%N)
                local time_ms=$(( (end - start) / 1000000 ))
            else
                local start_ms=$(python3 -c "import time; print(int(time.time() * 1000))")
                output=$(timeout 30 bash -c "$cmd" 2>&1)
                local exit_code=$?
                local end_ms=$(python3 -c "import time; print(int(time.time() * 1000))")
                local time_ms=$((end_ms - start_ms))
            fi

            # Check for errors
            if [ $exit_code -ne 0 ]; then
                echo "ERROR:$exit_code"
                exit 1
            fi

            echo "$time_ms"
            echo "$output"
        ) > /tmp/benchmark_run_$$.txt 2>&1

        local subshell_result=$?

        if [ $subshell_result -ne 0 ]; then
            echo -e "  ${RED}FAILED:${NC} Benchmark crashed or timed out (30s)"
            if grep -q "ERROR:" /tmp/benchmark_run_$$.txt 2>/dev/null; then
                local error_code=$(grep "ERROR:" /tmp/benchmark_run_$$.txt | cut -d: -f2)
                echo -e "  ${RED}Exit code:${NC} $error_code"
            fi
            failed=true
            rm -f /tmp/benchmark_run_$$.txt
            break
        fi

        # Parse output
        local time_ms=$(head -1 /tmp/benchmark_run_$$.txt)
        local output=$(tail -n +2 /tmp/benchmark_run_$$.txt)

        total_time=$((total_time + time_ms))

        # Capture result from first iteration
        if [ $i -eq 1 ]; then
            result=$(echo "$output" | tail -1 | tr -d '\n' | tr -d ' ')
        fi

        rm -f /tmp/benchmark_run_$$.txt
    done

    if [ "$failed" = true ]; then
        echo "$impl,$bench_name,FAILED,N/A" >> "$RESULTS_FILE"
        echo ""
        return 1
    fi

    # Calculate average
    local avg_time=$((total_time / ITERATIONS))

    echo -e "  ${GREEN}Average:${NC} ${avg_time}ms (${ITERATIONS} iterations)"
    echo -e "  ${GREEN}Result:${NC} $result"
    echo ""

    # Write to CSV
    echo "$impl,$bench_name,$avg_time,$result" >> "$RESULTS_FILE"
}

# Check if Tantalisp is built in release mode
if [ "$HAVE_TANTALISP" = true ]; then
    if [ ! -f "$PROJECT_ROOT/target/release/tantalisp" ]; then
        echo -e "${RED}ERROR:${NC} Tantalisp not found in release mode!"
        echo ""
        echo "Please build Tantalisp first:"
        echo "  cd $PROJECT_ROOT"
        echo "  cargo build --release"
        echo ""
        exit 1
    fi
    echo -e "${GREEN}✓${NC} Tantalisp release binary found"
    echo ""
fi

# Run SBCL benchmarks (compiled)
if [ "$HAVE_SBCL" = true ]; then
    echo "========================================"
    echo -e "${YELLOW}SBCL (Compiled)${NC}"
    echo "========================================"
    echo ""

    run_benchmark "SBCL" "TAK" \
        "sbcl --script $GABRIEL_DIR/commonlisp/tak.lisp"

    run_benchmark "SBCL" "DIV2" \
        "sbcl --script $GABRIEL_DIR/commonlisp/div2.lisp"

    run_benchmark "SBCL" "TRIANG" \
        "sbcl --script $GABRIEL_DIR/commonlisp/triang.lisp"

    run_benchmark "SBCL" "FIB" \
        "sbcl --script $GABRIEL_DIR/commonlisp/fib.lisp"

    run_benchmark "SBCL" "TAKL" \
        "sbcl --script $GABRIEL_DIR/commonlisp/takl.lisp"
fi

# Run Racket benchmarks
if [ "$HAVE_RACKET" = true ]; then
    echo "========================================"
    echo -e "${YELLOW}Racket${NC}"
    echo "========================================"
    echo ""

    run_benchmark "Racket" "TAK" \
        "racket $GABRIEL_DIR/scheme/tak.scm"

    run_benchmark "Racket" "DIV2" \
        "racket $GABRIEL_DIR/scheme/div2.scm"

    run_benchmark "Racket" "TRIANG" \
        "racket $GABRIEL_DIR/scheme/triang.scm"

    run_benchmark "Racket" "FIB" \
        "racket $GABRIEL_DIR/scheme/fib.scm"

    run_benchmark "Racket" "TAKL" \
        "racket $GABRIEL_DIR/scheme/takl.scm"
fi

# Run Chicken Scheme benchmarks (compiled)
if [ "$HAVE_CHICKEN" = true ]; then
    echo "========================================"
    echo -e "${YELLOW}Chicken Scheme (Compiled)${NC}"
    echo "========================================"
    echo ""

    # Compile the benchmarks first
    mkdir -p /tmp/chicken_bench

    for bench in tak takl div2 triang fib; do
        csc -O3 -o /tmp/chicken_bench/$bench $GABRIEL_DIR/scheme/$bench.scm 2>/dev/null
    done

    run_benchmark "Chicken" "TAK" \
        "/tmp/chicken_bench/tak"

    run_benchmark "Chicken" "TAKL" \
        "/tmp/chicken_bench/takl"

    run_benchmark "Chicken" "DIV2" \
        "/tmp/chicken_bench/div2"

    run_benchmark "Chicken" "TRIANG" \
        "/tmp/chicken_bench/triang"

    run_benchmark "Chicken" "FIB" \
        "/tmp/chicken_bench/fib"

    # Cleanup
    rm -rf /tmp/chicken_bench
fi

# Run Clojure benchmarks
if [ "$HAVE_CLOJURE" = true ]; then
    echo "========================================"
    echo -e "${YELLOW}Clojure${NC}"
    echo "========================================"
    echo ""

    run_benchmark "Clojure" "TAK" \
        "clojure $GABRIEL_DIR/clojure/tak.clj"

    run_benchmark "Clojure" "DIV2" \
        "clojure $GABRIEL_DIR/clojure/div2.clj"

    run_benchmark "Clojure" "TRIANG" \
        "clojure $GABRIEL_DIR/clojure/triang.clj"

    run_benchmark "Clojure" "FIB" \
        "clojure $GABRIEL_DIR/clojure/fib.clj"

    run_benchmark "Clojure" "TAKL" \
        "clojure $GABRIEL_DIR/clojure/takl.clj"
fi

# Run Tantalisp benchmarks (last, in case of crashes)
if [ "$HAVE_TANTALISP" = true ]; then
    echo "========================================"
    echo -e "${YELLOW}Tantalisp (JIT Compiled)${NC}"
    echo "========================================"
    echo ""

    run_benchmark "Tantalisp" "TAK" \
        "$PROJECT_ROOT/target/release/tantalisp --file $GABRIEL_DIR/tak.tlsp"

    echo -e "${RED}Skipping:${NC} Tantalisp - DIV2 (stack overflow - no TCO, see CLAUDE.md)"
    echo ""

    run_benchmark "Tantalisp" "TRIANG" \
        "$PROJECT_ROOT/target/release/tantalisp --file $GABRIEL_DIR/triang.tlsp"

    run_benchmark "Tantalisp" "FIB" \
        "$PROJECT_ROOT/target/release/tantalisp --file $GABRIEL_DIR/fib.tlsp"

    echo -e "${RED}Skipping:${NC} Tantalisp - TAKL (segfault bug, see CLAUDE.md)"
    echo ""
fi

echo "========================================"
echo -e "${GREEN}Benchmarks Complete!${NC}"
echo "========================================"
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""

# Display summary table
echo "Summary Table:"
echo ""
column -t -s',' "$RESULTS_FILE"
echo ""

# Generate comparison chart (if gnuplot is available)
if command -v gnuplot &> /dev/null; then
    echo "Generating performance chart..."

    # Create gnuplot script
    cat > /tmp/gabriel_plot.gp << 'EOF'
set terminal png size 1400,800
set output '/tmp/gabriel_chart.png'
set title "Gabriel Benchmarks - Performance Comparison"
set ylabel "Time (ms)"
set xlabel "Benchmark"
set style data histograms
set style histogram clustered gap 1
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
set grid y
set key outside right top
set yrange [0:*]

set datafile separator ","
plot for [i=2:*] '/tmp/gabriel_data.txt' using i:xtic(1) title columnheader
EOF

    # Transform CSV to gnuplot format
    awk -F',' '
    NR==1 {next}
    {
        if (!seen_impl[$1]) {
            impl[++impl_count] = $1
            seen_impl[$1] = 1
        }
        if (!seen_bench[$2]) {
            bench[++bench_count] = $2
            seen_bench[$2] = 1
        }
        data[$2","$1] = $3
    }
    END {
        # Print header
        printf "Benchmark"
        for (i=1; i<=impl_count; i++) printf ",%s", impl[i]
        printf "\n"
        # Print data (skip FAILED entries)
        for (b=1; b<=bench_count; b++) {
            printf "%s", bench[b]
            for (i=1; i<=impl_count; i++) {
                val = data[bench[b]","impl[i]]
                # Use 0 for missing data, skip FAILED
                if (val == "FAILED" || val == "") val = "0"
                printf ",%s", val
            }
            printf "\n"
        }
    }' "$RESULTS_FILE" > /tmp/gabriel_data.txt

    gnuplot /tmp/gabriel_plot.gp 2>/dev/null

    if [ -f /tmp/gabriel_chart.png ]; then
        mv /tmp/gabriel_chart.png "$BENCHMARKS_DIR/gabriel_chart.png"
        echo -e "${GREEN}Chart saved to:${NC} $BENCHMARKS_DIR/gabriel_chart.png"

        # Try to open it
        if command -v open &> /dev/null; then
            open "$BENCHMARKS_DIR/gabriel_chart.png"
        fi
    fi
fi

echo ""
echo "Done!"
