#!/bin/bash

# Performance test runner for Tantalisp
# Runs perf_test.tlsp with GC monitoring enabled

echo "=== Tantalisp Performance Test ==="
echo "Running perf_test.tlsp with GC monitoring..."
echo ""

# Clean up old log
rm -f gc_debug.log

# Run the test with GC_DEBUG enabled
echo "Starting test (GC stats will be logged to gc_debug.log)..."
time GC_DEBUG=1 cargo run --release -- --file perf_test.tlsp

echo ""
echo "=== GC Statistics Summary ==="
echo ""

# Show first few lines
echo "Initial allocations:"
head -3 gc_debug.log

echo ""
echo "Final allocations:"
tail -3 gc_debug.log

echo ""
echo "=== Leak Analysis ==="

# Get final stats
FINAL_LINE=$(tail -1 gc_debug.log)
echo "Final: $FINAL_LINE"

# Extract leaked bytes (use word boundaries to avoid matching "deallocated")
LEAKED=$(echo "$FINAL_LINE" | grep -o 'leaked=[0-9]*' | cut -d= -f2)
ALLOCATED=$(echo "$FINAL_LINE" | grep -o ' allocated=[0-9]*' | cut -d= -f2)

if [ -n "$LEAKED" ] && [ -n "$ALLOCATED" ]; then
    LEAK_PCT=$(awk -v leaked="$LEAKED" -v allocated="$ALLOCATED" 'BEGIN {printf "%.4f", (leaked / allocated) * 100}')
    echo ""
    echo "Total leaked: $LEAKED bytes"
    echo "Total allocated: $ALLOCATED bytes"
    echo "Leak percentage: $LEAK_PCT%"
fi

echo ""
echo "Full GC log saved to: gc_debug.log"

# Generate plot if gnuplot is available
if command -v gnuplot &> /dev/null; then
    echo ""
    echo "Generating memory usage plot..."

    # Extract data for plotting using grep and sed
    grep -o 'allocs=[0-9]* allocated=[0-9]* deallocated=[0-9]* leaked=[0-9]*' gc_debug.log | \
        sed 's/allocs=//; s/allocated=/,/; s/deallocated=/,/; s/leaked=/,/' | \
        awk -F, '{print NR, $1, $2, $3, $4}' > /tmp/gc_plot_data.txt

    # Create gnuplot script
    cat > /tmp/gc_plot.gp << 'EOF'
set terminal qt size 1200,800 title "Tantalisp GC Memory Usage"
set title "Memory Usage Over Time"
set xlabel "Sample Number (100ms intervals)"
set ylabel "Bytes (log scale)"
set logscale y
set grid
set key outside right top

plot '/tmp/gc_plot_data.txt' using 1:3 with lines lw 2 title "Allocated", \
     '/tmp/gc_plot_data.txt' using 1:4 with lines lw 2 title "Deallocated", \
     '/tmp/gc_plot_data.txt' using 1:5 with lines lw 2 title "Leaked"

pause mouse close
EOF

    # Run gnuplot
    gnuplot /tmp/gc_plot.gp &
    echo "Plot window opened (close it to continue)"
else
    echo ""
    echo "Install gnuplot to see memory usage plots: brew install gnuplot"
fi
