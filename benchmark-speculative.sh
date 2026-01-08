#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Speculative Decoding Benchmark ===${NC}\n"

# Test prompts
PROMPTS=(
    "Write a short story about a robot"
    "Explain quantum computing in simple terms"
    "Write a poem about the ocean"
)

# Function to extract tokens/s from verbose output
extract_speed() {
    grep "eval rate:" | awk '{print $3}' | head -1
}

echo -e "${GREEN}Creating test models...${NC}"
./ollama create qwen-baseline -f Modelfile-no-draft > /dev/null 2>&1 || true
./ollama create qwen-speculative -f Modelfile-with-draft > /dev/null 2>&1 || true

echo ""

for i in "${!PROMPTS[@]}"; do
    PROMPT="${PROMPTS[$i]}"
    echo -e "${YELLOW}Test $((i+1)): ${NC}${PROMPT:0:50}..."
    
    # Baseline test
    echo -n "  Baseline:     "
    BASELINE_OUT=$(./ollama run qwen-baseline "$PROMPT" --verbose 2>&1)
    BASELINE_SPEED=$(echo "$BASELINE_OUT" | extract_speed)
    echo -e "${BASELINE_SPEED} tokens/s"
    
    # Speculative test
    echo -n "  Speculative:  "
    SPEC_OUT=$(./ollama run qwen-speculative "$PROMPT" --verbose 2>&1)
    SPEC_SPEED=$(echo "$SPEC_OUT" | extract_speed)
    echo -e "${SPEC_SPEED} tokens/s"
    
    # Calculate speedup
    if [[ -n "$BASELINE_SPEED" && -n "$SPEC_SPEED" ]]; then
        SPEEDUP=$(echo "scale=2; $SPEC_SPEED / $BASELINE_SPEED" | bc)
        echo -e "  ${GREEN}Speedup: ${SPEEDUP}x${NC}"
    fi
    
    echo ""
done

echo -e "${YELLOW}=== Server Logs (last 20 lines) ===${NC}"
tail -20 server.log | grep -E "speculative|draft|acceptance" || echo "No speculative logs found"
