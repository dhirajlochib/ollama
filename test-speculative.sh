#!/bin/bash

echo "========================================="
echo "Testing SPECULATIVE DECODING"
echo "========================================="
echo ""

echo "Test 1: Model WITH draft (speculative decoding)"
echo "------------------------------------------------"
time ./ollama run my-model-with-draft "Count from 1 to 20" --verbose 2>&1 | tee test-with-draft.log

echo ""
echo ""
echo "Test 2: Model WITHOUT draft (normal)"
echo "------------------------------------------------"
time ./ollama run my-model-no-draft "Count from 1 to 20" --verbose 2>&1 | tee test-no-draft.log

echo ""
echo ""
echo "========================================="
echo "RESULTS COMPARISON"
echo "========================================="
echo ""
echo "WITH DRAFT:"
grep "eval rate" test-with-draft.log
echo ""
echo "WITHOUT DRAFT:"
grep "eval rate" test-no-draft.log
echo ""

rm test-with-draft.log test-no-draft.log
