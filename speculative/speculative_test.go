package speculative

import (
	"testing"
)

func TestDefaultConfig(t *testing.T) {
	config := DefaultConfig()

	if config.NumSpeculativeTokens != 5 {
		t.Errorf("expected NumSpeculativeTokens=5, got %d", config.NumSpeculativeTokens)
	}

	if config.AcceptanceThreshold != 0.9 {
		t.Errorf("expected AcceptanceThreshold=0.9, got %f", config.AcceptanceThreshold)
	}

	if config.Enabled {
		t.Error("expected Enabled=false")
	}
}

func TestConfigFromModel(t *testing.T) {
	tests := []struct {
		name       string
		draftModel string
		wantEnable bool
	}{
		{
			name:       "empty draft model",
			draftModel: "",
			wantEnable: false,
		},
		{
			name:       "with draft model",
			draftModel: "qwen2.5:0.5b",
			wantEnable: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := ConfigFromModel(tt.draftModel)

			if config.Enabled != tt.wantEnable {
				t.Errorf("expected Enabled=%v, got %v", tt.wantEnable, config.Enabled)
			}

			if tt.wantEnable && config.DraftModelName != tt.draftModel {
				t.Errorf("expected DraftModelName=%s, got %s", tt.draftModel, config.DraftModelName)
			}
		})
	}
}

func TestNewEngine(t *testing.T) {
	config := ConfigFromModel("test-draft")
	engine := NewEngine(config)

	if engine == nil {
		t.Fatal("expected non-nil engine")
	}

	if engine.config.DraftModelName != "test-draft" {
		t.Errorf("expected DraftModelName='test-draft', got '%s'", engine.config.DraftModelName)
	}

	if !engine.config.Enabled {
		t.Error("expected config.Enabled=true")
	}
}

func TestEngineIsReady(t *testing.T) {
	config := DefaultConfig()
	engine := NewEngine(config)

	// Not ready - no servers set
	if engine.IsReady() {
		t.Error("expected IsReady()=false with no servers")
	}

	// Still not ready - config not enabled
	config.Enabled = true
	engine = NewEngine(config)
	if engine.IsReady() {
		t.Error("expected IsReady()=false with no servers even if enabled")
	}
}

func TestEngineAcceptTokensBasic(t *testing.T) {
	config := Config{
		AcceptanceThreshold: 0.5,
	}
	engine := NewEngine(config)

	// Test empty inputs
	accepted := engine.acceptTokens(nil, nil)
	if accepted != 0 {
		t.Errorf("expected 0 accepted for nil inputs, got %d", accepted)
	}

	accepted = engine.acceptTokens([]DraftCandidate{}, nil)
	if accepted != 0 {
		t.Errorf("expected 0 accepted for empty draft, got %d", accepted)
	}
}

func TestEngineStats(t *testing.T) {
	config := DefaultConfig()
	engine := NewEngine(config)

	total, accepted, rate := engine.Stats()

	if total != 0 || accepted != 0 || rate != 0 {
		t.Errorf("expected zero stats, got total=%d, accepted=%d, rate=%f", total, accepted, rate)
	}
}
