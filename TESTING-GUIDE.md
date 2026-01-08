# Speculative Decoding - Local Testing Guide

## Quick Start

### 1. Build and Start Server

```bash
# Build
go build -o ollama .

# Start server with debug logging
OLLAMA_DEBUG=1 ./ollama serve 2>&1 | tee server.log &

# Wait for startup
sleep 3
```

### 2. Verify Server is Running

```bash
# Check if server is responding
./ollama list

# If you see models listed, server is ready!
```

### 3. Create Test Models

You should have these Modelfiles already:

- `Modelfile-no-draft` - Baseline model
- `Modelfile-with-draft` - Model with draft for speculative decoding

```bash
# Create baseline model
./ollama create test-baseline -f Modelfile-no-draft

# Create speculative model
./ollama create test-spec -f Modelfile-with-draft
```

### 4. Run Simple Test

```bash
# Test baseline
echo "=== BASELINE ===" 
time ./ollama run test-baseline "Write a short joke"

# Test speculative
echo "=== SPECULATIVE ===" 
time ./ollama run test-spec "Write a short joke"
```

### 5. Check Server Logs

```bash
# See what's happening with speculative decoding
tail -50 server.log | grep -E "speculative|draft|acceptance"
```

## Expected Output (Current State)

### ✅ Infrastructure Working:
- Both models load successfully
- Server logs: `"speculative decoding ready (infrastructure)"` or `"draft model loaded, using normal completion"`
- Full responses generated (no truncation)

### ⚠️ No Speedup Yet:
- Both baseline and speculative have **similar performance**
- This is expected! The infrastructure is complete, but the actual speculative algorithm runs through normal `Completion()` path
- To get speedup, need runner-level integration (see below)

## Server Log Messages to Look For

### Current State:
```
msg="speculative decoding ready (infrastructure)" draft_model=qwen2.5:0.5b
msg="draft model loaded, using normal completion" draft=qwen2.5:0.5b
```

### When Draft Model Loads:
```
msg="loading draft model" draft=qwen2.5:0.5b
msg="draft runner ready" draft=qwen2.5:0.5b
```

## Benchmarking

### Run Automated Benchmark:
```bash
./benchmark-speculative.sh
```

This will:
- Test multiple prompts
- Compare baseline vs speculative
- Extract tokens/s from verbose output
- Show speedup ratio

### Manual Timing:
```bash
# Get detailed stats with --verbose
./ollama run test-spec "Write a detailed story" --verbose
```

Look for:
- `eval rate:` - tokens per second
- Total tokens generated
- Total eval duration

## What's Next for Real Speedup

The current implementation has **all the infrastructure** but uses normal `Completion()` path. For actual 2-3x speedup, we need:

### Runner-Level Integration:
1. **Modify `runner/ollamarunner/runner.go`**
   - Hook into token-by-token generation loop
   - Add speculative batch generation

2. **Shared KV Cache Management**
   - Draft and target share KV cache state
   - Efficiently rewind on rejection

3. **Batch Verification**
   - Target verifies K draft tokens in one forward pass
   - Accept matching prefix

### Testing After Runner Integration:
You should see:
```
# In logs:
msg="using speculative decoding for completion"
msg="speculative decoding completed" total_draft_tokens=20 total_accepted=12 acceptance_rate=60.0%

# In performance:
Baseline:    45 tokens/s
Speculative: 95 tokens/s  <-- 2.1x speedup!
```

## Troubleshooting

### Server Won't Start:
```bash
# Kill existing servers
pkill -f "ollama serve"

# Check port is free
lsof -i :11434

# Try again
OLLAMA_DEBUG=1 ./ollama serve 2>&1 | tee server.log &
```

### Model Not Found:
```bash
# List available models
./ollama list

# Recreate if needed
./ollama create test-spec -f Modelfile-with-draft
```

### No Draft Model Loaded:
Check server.log for:
```
msg="draft model not yet loaded for speculative decoding"
```

This means the draft model is specified in Modelfile but hasn't loaded yet. Should load automatically on next request.

## Performance Expectations

### Current Implementation (Infrastructure Only):
- **Baseline**: ~40-50 tokens/s
- **Speculative**: ~40-50 tokens/s (same, uses normal path)
- ✅ Models load correctly
- ✅ Full responses work
- ⚠️ No speedup yet

### With Full Runner Integration (Future):
- **Baseline**: ~40-50 tokens/s  
- **Speculative**: ~80-120 tokens/s (2-3x faster!)
- ✅ Draft generates K tokens ahead
- ✅ Target verifies in batch
- ✅ Accept matching prefix, repeat

## Files to Check

1. **Server Logs** - `server.log` or stdout from `ollama serve`
2. **Model Config** - Run `./ollama show test-spec` to see draft model
3. **API Response** - Models should include `"draft": "qwen2.5:0.5b"` field

## Next Steps

1. ✅ Verify infrastructure works (you're here!)
2. 🔄 Implement runner-level integration for speedup
3. 🎯 Test and achieve 2-3x performance improvement
4. 📊 Benchmark with various model sizes
5. 🚀 Submit PR to Ollama

---

**Current Status**: Infrastructure complete, foundation ready for runner integration.
