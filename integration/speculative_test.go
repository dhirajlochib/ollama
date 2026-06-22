//go:build integration && speculative

// Speculative decoding integration tests. Opt-in because they need a running
// server plus locally created models with draft heads.
//
// Run:
//
//	OLLAMA_TEST_EXISTING=1 \
//	SPEC_BASE_MODEL=qwen35-4b-plain \
//	SPEC_DRAFT_MODEL=qwen35-4b-dflash \
//	go test -tags='integration,speculative' -count=1 -timeout 30m \
//	  ./integration -run TestSpeculativeGenerate -v
package integration

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// TestSpeculativeGenerate compares plain vs draft-backed models on the same
// prompt under greedy sampling, reporting eval tokens/s for both.
func TestSpeculativeGenerate(t *testing.T) {
	if os.Getenv("OLLAMA_TEST_EXISTING") == "" {
		t.Skip("set OLLAMA_TEST_EXISTING=1 and start ollama serve")
	}
	base := os.Getenv("SPEC_BASE_MODEL")
	draft := os.Getenv("SPEC_DRAFT_MODEL")
	if base == "" {
		t.Skip("SPEC_BASE_MODEL required")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Minute)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	prompt := os.Getenv("SPEC_PROMPT")
	if prompt == "" {
		prompt = "Explain speculative decoding, multi-token prediction, and DFlash block diffusion in depth."
	}
	stream := false
	numPredict := 128

	runOnce := func(t *testing.T, model, label string) (evalTPS float64, evalCount int) {
		t.Helper()
		var final api.GenerateResponse
		err := client.Generate(ctx, &api.GenerateRequest{
			Model:  model,
			Prompt: prompt,
			Stream: &stream,
			Options: map[string]any{
				"temperature":    0,
				"num_predict":    numPredict,
				"repeat_penalty": 1.0,
			},
		}, func(r api.GenerateResponse) error {
			final = r
			return nil
		})
		if err != nil {
			t.Fatalf("%s generate failed: %v", label, err)
		}
		if final.EvalCount == 0 || final.EvalDuration == 0 {
			t.Fatalf("%s: missing eval metrics: %+v", label, final)
		}
		tps := float64(final.EvalCount) / final.EvalDuration.Seconds()
		promptTPS := 0.0
		if final.PromptEvalDuration > 0 {
			promptTPS = float64(final.PromptEvalCount) / final.PromptEvalDuration.Seconds()
		}
		t.Logf("%s model=%s eval_count=%d eval_tps=%.2f prompt_tps=%.2f total=%s",
			label, model, final.EvalCount, tps, promptTPS, final.TotalDuration)
		return tps, final.EvalCount
	}

	// Warmup base
	_, _ = runOnce(t, base, "warmup_base")
	baseTPS, baseN := runOnce(t, base, "plain")

	if draft == "" {
		t.Logf("SPEC_DRAFT_MODEL unset; only plain measured (%.2f tps, %d tokens)", baseTPS, baseN)
		return
	}
	_, _ = runOnce(t, draft, "warmup_draft")
	draftTPS, draftN := runOnce(t, draft, "speculative")

	speedup := draftTPS / baseTPS
	t.Logf("RESULT plain_tps=%.2f spec_tps=%.2f speedup=%.2fx tokens_plain=%d tokens_spec=%d",
		baseTPS, draftTPS, speedup, baseN, draftN)
	if draftTPS <= 0 || baseTPS <= 0 {
		t.Fatal("non-positive throughput")
	}
}
