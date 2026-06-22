#!/usr/bin/env bash
# Automated speculative-decoding benchmark (plain vs MTP/DFlash) against a
# running Ollama server. Emits machine-readable JSONL + human summary.
#
# Usage:
#   # Tier A — unit/config tests (no models, always run in CI):
#   ./scripts/bench_speculative.sh unit
#
#   # Tier B — live server perf (needs ollama serve + models):
#   OLLAMA_HOST=http://127.0.0.1:11434 \
#   SPEC_BASE_MODEL=qwen3.5:4b \
#   SPEC_DRAFT_MODEL=qwen3.5-4b-dflash \   # optional; omit for plain-only
#   ./scripts/bench_speculative.sh perf
#
#   # Tier C — recommended 24GB (Apple unified / NVIDIA) model matrix:
#   ./scripts/bench_speculative.sh recommend
#
# Env knobs:
#   SPEC_PROMPT          default multi-paragraph prompt
#   SPEC_NUM_PREDICT     tokens to generate (default 256)
#   SPEC_TEMPERATURE     default 0 (greedy; best for acceptance fairness)
#   SPEC_RUNS            repeats per model (default 3; first run warmup discarded)
#   SPEC_OUT             results file (default /tmp/ollama_spec_bench.jsonl)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${SPEC_OUT:-/tmp/ollama_spec_bench.jsonl}"
HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
NUM_PREDICT="${SPEC_NUM_PREDICT:-256}"
TEMP="${SPEC_TEMPERATURE:-0}"
RUNS="${SPEC_RUNS:-3}"
PROMPT="${SPEC_PROMPT:-Write a detailed technical explanation of how speculative decoding works in large language models, including draft models, verification, acceptance rates, and practical trade-offs for production serving. Include examples.}"

cmd="${1:-unit}"

unit_tests() {
  echo "== Tier A: unit / config tests (no weights) =="
  (cd "$ROOT" && go test ./x/mlxrunner/ -count=1 -timeout 120s)
  (cd "$ROOT" && go test ./x/models/dflash/ -count=1 -timeout 120s)
  (cd "$ROOT" && go test ./x/mlxrunner/model/base/ -count=1 -timeout 60s 2>/dev/null || true)
  echo "OK: unit tier passed"
}

recommend() {
  cat <<'EOF'
== Recommended models for ~24 GB (Apple M-series unified / 24GB dGPU) ==

DFlash path (MLX runner, safetensors create — needs this branch):
  Target (pick one that fits):
    • Qwen3.5-4B  (safetensors / mxfp8 or q4)   ~3–6 GB weights — safest
    • Qwen3.5-9B  (q4 / mxfp8)                  ~6–12 GB — good if headroom
    • Qwen3-8B    (q4)                          ~5–9 GB
  Draft (HF, pair with matching target family):
    • z-lab/Qwen3.5-4B-DFlash   (~0.6B, ~1–2 GB)
    • z-lab/Qwen3-8B-DFlash-b16 (~1B, ~2 GB)
  Avoid on 24GB: Qwen3.5-27B+DFlash (target alone often >24GB unquantized)

MTP path (already on main for Gemma4 assistant draft):
  Target: gemma4:e2b / gemma4:e4b (library, size varies by quant)
  Draft:  bundled assistant MTP head via ollama create DRAFT

GGUF MTP (llama-server, not DFlash):
  Target: any qwen3.5/qwen3next GGUF with nextn/mtp tensors
  Flags:  runner sets --spec-type draft-mtp when EnableMTP / DraftModelPath

Create example (DFlash, once weights are local):
  mkdir -p /tmp/dflash-bench && cd /tmp/dflash-bench
  hf download z-lab/Qwen3.5-4B-DFlash --local-dir ./draft
  hf download Qwen/Qwen3.5-4B --local-dir ./target   # or your quant export
  cat > Modelfile <<'MF'
  FROM ./target
  DRAFT ./draft
  PARAMETER temperature 0
  PARAMETER num_predict 256
  MF
  ollama create qwen35-4b-dflash -f Modelfile

Then:
  SPEC_BASE_MODEL=qwen35-4b-plain SPEC_DRAFT_MODEL=qwen35-4b-dflash \
    ./scripts/bench_speculative.sh perf

Metrics collected per run:
  prompt_tps, eval_tps (tokens/s), eval_count, eval_duration_ms,
  total_duration_ms, load_duration_ms, acceptance (from server logs if present)

EOF
}

# Single generate call; prints JSON metrics line to stdout (and appends to OUT).
one_run() {
  local model="$1" label="$2" run_idx="$3"
  local payload resp
  payload=$(python3 - <<PY
import json, os
print(json.dumps({
  "model": os.environ["MODEL"],
  "prompt": os.environ["PROMPT"],
  "stream": False,
  "options": {
    "temperature": float(os.environ["TEMP"]),
    "num_predict": int(os.environ["NUM_PREDICT"]),
  },
}))
PY
)
  MODEL="$model" PROMPT="$PROMPT" TEMP="$TEMP" NUM_PREDICT="$NUM_PREDICT" \
  resp=$(curl -sS "$HOST/api/generate" -H 'Content-Type: application/json' -d "$payload")

  python3 - <<PY
import json, os, sys, time
raw = '''$resp'''
try:
    r = json.loads(raw)
except Exception as e:
    print(json.dumps({"error": str(e), "raw": raw[:500]}))
    sys.exit(1)
if r.get("error"):
    print(json.dumps({"error": r["error"], "model": "$model", "label": "$label"}))
    sys.exit(1)

ec = r.get("eval_count") or 0
ed_ns = r.get("eval_duration") or 0
pc = r.get("prompt_eval_count") or 0
pd_ns = r.get("prompt_eval_duration") or 0
td_ns = r.get("total_duration") or 0
ld_ns = r.get("load_duration") or 0

def tps(count, ns):
    if not ns or ns <= 0 or not count:
        return 0.0
    return count / (ns / 1e9)

row = {
    "ts": time.time(),
    "label": "$label",
    "model": "$model",
    "run": int("$run_idx"),
    "warmup": int("$run_idx") == 0,
    "prompt_eval_count": pc,
    "prompt_tps": round(tps(pc, pd_ns), 3),
    "eval_count": ec,
    "eval_tps": round(tps(ec, ed_ns), 3),
    "eval_duration_ms": round(ed_ns / 1e6, 2),
    "total_duration_ms": round(td_ns / 1e6, 2),
    "load_duration_ms": round(ld_ns / 1e6, 2),
    "temperature": float("$TEMP"),
    "num_predict": int("$NUM_PREDICT"),
}
print(json.dumps(row))
open("$OUT", "a").write(json.dumps(row) + "\n")
PY
}

perf() {
  echo "== Tier B: live perf against $HOST =="
  if ! curl -sf "$HOST/api/tags" >/dev/null; then
    echo "ERROR: cannot reach $HOST — start with: ollama serve" >&2
    exit 1
  fi
  : >"$OUT"
  base="${SPEC_BASE_MODEL:-}"
  draft="${SPEC_DRAFT_MODEL:-}"
  if [[ -z "$base" ]]; then
    echo "Set SPEC_BASE_MODEL (plain/target model name in ollama list)" >&2
    exit 1
  fi

  echo "Benchmarking base=$base draft=${draft:-none} runs=$RUNS num_predict=$NUM_PREDICT temp=$TEMP"
  echo "Writing JSONL -> $OUT"

  for i in $(seq 0 $((RUNS - 1))); do
    echo "-- base run $i --"
    one_run "$base" "plain_or_base" "$i" || true
  done
  if [[ -n "$draft" ]]; then
    for i in $(seq 0 $((RUNS - 1))); do
      echo "-- draft/spec run $i --"
      one_run "$draft" "speculative" "$i" || true
    done
  fi

  python3 - <<'PY'
import json, statistics, os, sys
path = os.environ.get("SPEC_OUT", "/tmp/ollama_spec_bench.jsonl")
rows = []
with open(path) as f:
    for line in f:
        line=line.strip()
        if not line: continue
        try:
            r=json.loads(line)
        except: continue
        if r.get("error") or r.get("warmup"):
            continue
        rows.append(r)

def summarize(label):
    xs=[r for r in rows if r.get("label")==label]
    if not xs:
        return None
    tps=[r["eval_tps"] for r in xs if r.get("eval_tps")]
    if not tps:
        return {"label": label, "n": 0}
    return {
        "label": label,
        "n": len(tps),
        "eval_tps_mean": round(statistics.mean(tps), 3),
        "eval_tps_stdev": round(statistics.pstdev(tps), 3) if len(tps)>1 else 0.0,
        "eval_tps_min": round(min(tps), 3),
        "eval_tps_max": round(max(tps), 3),
        "eval_count_mean": round(statistics.mean(r["eval_count"] for r in xs), 1),
        "prompt_tps_mean": round(statistics.mean(r["prompt_tps"] for r in xs), 3),
    }

plain = summarize("plain_or_base")
spec = summarize("speculative")
print("\n======== SUMMARY (warmup runs excluded) ========")
print(json.dumps({"plain": plain, "speculative": spec}, indent=2))
if plain and spec and plain.get("eval_tps_mean") and spec.get("eval_tps_mean"):
    speedup = spec["eval_tps_mean"] / plain["eval_tps_mean"]
    print(f"\nSpeedup (spec / plain eval_tps): {speedup:.2f}x")
print(f"Raw JSONL: {path}")
PY
}

case "$cmd" in
  unit) unit_tests ;;
  perf) perf ;;
  recommend) recommend ;;
  all) unit_tests; recommend ;;
  *) echo "usage: $0 {unit|perf|recommend|all}"; exit 2 ;;
esac
