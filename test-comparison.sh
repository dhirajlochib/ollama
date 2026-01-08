#!/bin/bash

PROMPT="Write a detailed explanation of machine learning in 5 paragraphs"

echo "=== Testing BASELINE (no draft) ==="
time ./ollama run qwen-baseline "$PROMPT" --verbose

echo ""
echo "=== Testing SPECULATIVE (with draft) ==="
time ./ollama run qwen-speculative "$PROMPT" --verbose
