#!/bin/bash

# Test runner for Tantalisp
# Runs all tests with --test-threads=1 to avoid LLVM thread-safety issues

echo "=== Tantalisp Test Suite ==="
echo ""
echo "Running all tests (single-threaded due to LLVM limitations)..."
echo ""

cargo test --lib -- --test-threads=1

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ All tests passed!"
else
    echo "✗ Some tests failed (exit code: $EXIT_CODE)"
fi

exit $EXIT_CODE
