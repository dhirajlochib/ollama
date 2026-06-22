package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	sampler "github.com/ollama/ollama/x/mlxrunner/sample"
	"github.com/ollama/ollama/x/tokenizer"
)

// --- stubs for DetectSpecKind / selectSpecMode ---

type stubModel struct{}

func (stubModel) Forward(*batch.Batch, []cache.Cache) *mlx.Array { return nil }
func (stubModel) Unembed(*mlx.Array) *mlx.Array                  { return nil }
func (stubModel) NumLayers() int                                 { return 4 }
func (stubModel) Tokenizer() *tokenizer.Tokenizer                { return nil }
func (stubModel) MaxContextLength() int                          { return 128 }
func (stubModel) LoadWeights(map[string]*mlx.Array) error        { return nil }

type stubEmbedModel struct{ stubModel }

func (stubEmbedModel) TokenEmbeddings(*mlx.Array) *mlx.Array { return nil }

type stubDFlashTarget struct{ stubEmbedModel }

func (stubDFlashTarget) ForwardDFlash(*batch.Batch, []cache.Cache, []int) (*mlx.Array, *mlx.Array) {
	return nil, nil
}

type stubMTPDraft struct{}

func (stubMTPDraft) LoadWeights(map[string]*mlx.Array) error { return nil }
func (stubMTPDraft) Draft(*mlx.Array, int32, []cache.Cache) (*mlx.Array, *mlx.Array) {
	return nil, nil
}

type stubDFlashDraft struct{}

func (stubDFlashDraft) LoadWeights(map[string]*mlx.Array) error { return nil }
func (stubDFlashDraft) TargetLayerIDs() []int                   { return []int{1, 2} }
func (stubDFlashDraft) BlockSize() int                          { return 16 }
func (stubDFlashDraft) MaskTokenID() int32                      { return 0 }
func (stubDFlashDraft) NewCaches() []cache.Cache                { return nil }
func (stubDFlashDraft) AppendContext(*mlx.Array, []cache.Cache) {}
func (stubDFlashDraft) Draft(*mlx.Array, []cache.Cache) *mlx.Array { return nil }

func TestDetectSpecKind(t *testing.T) {
	tests := []struct {
		name   string
		target base.Model
		draft  base.DraftModel
		want   base.SpecKind
	}{
		{"nil draft", stubEmbedModel{}, nil, base.SpecKindNone},
		{"mtp only", stubEmbedModel{}, stubMTPDraft{}, base.SpecKindMTP},
		{"dflash full", stubDFlashTarget{}, stubDFlashDraft{}, base.SpecKindDFlash},
		{"dflash draft without target capture", stubEmbedModel{}, stubDFlashDraft{}, base.SpecKindNone},
		{"dflash prefers over mtp shape", stubDFlashTarget{}, stubDFlashDraft{}, base.SpecKindDFlash},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := base.DetectSpecKind(tt.target, tt.draft); got != tt.want {
				t.Fatalf("DetectSpecKind = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSelectSpecModeDFlashGreedy(t *testing.T) {
	r := &Runner{Model: stubDFlashTarget{}, Draft: stubDFlashDraft{}}
	mode, reason := r.selectSpecMode(sampler.Options{Temperature: 0})
	if mode != SpecModeDFlashGreedy {
		t.Fatalf("mode = %v (%s), want dflash_greedy", mode, reason)
	}
}

func TestSelectSpecModeLogprobsDisables(t *testing.T) {
	r := &Runner{Model: stubDFlashTarget{}, Draft: stubDFlashDraft{}}
	mode, reason := r.selectSpecMode(sampler.Options{Logprobs: true})
	if mode != SpecModeNone {
		t.Fatalf("mode = %v, want none", mode)
	}
	if reason != "logprobs_requested" {
		t.Fatalf("reason = %q, want logprobs_requested", reason)
	}
}

func TestDFlashDraftLimit(t *testing.T) {
	d := stubDFlashDraft{}
	if got := dflashDraftLimit(d, 100); got != 15 {
		t.Fatalf("draft limit = %d, want 15 (block_size-1)", got)
	}
	if got := dflashDraftLimit(d, 3); got != 3 {
		t.Fatalf("draft limit = %d, want 3 (remaining)", got)
	}
	if got := dflashDraftLimit(nil, 10); got != 0 {
		t.Fatalf("draft limit = %d, want 0", got)
	}
}

func TestDepthControllerPrefersDeeperWhenAcceptHigh(t *testing.T) {
	c := newDepthController()
	// Cheap deep drafts with perfect acceptance should beat depth 0.
	for range 20 {
		c.observe(4, 4, 1.0) // depth 4 always fully accepted, cost 1.0
		c.observe(0, 0, 1.0) // depth 0 same wall cost
	}
	sel := c.selected()
	if sel < 1 {
		t.Fatalf("selected depth = %d, want >= 1 after high acceptance", sel)
	}
}

func TestDepthControllerAcceptanceInherit(t *testing.T) {
	c := newDepthController()
	c.observe(2, 1, 1.0) // position 1 accept, position 2 reject
	if a := c.acceptance(1); a != 1.0 {
		t.Fatalf("acceptance(1) = %v, want 1", a)
	}
	if a := c.acceptance(2); a != 0.0 {
		t.Fatalf("acceptance(2) = %v, want 0", a)
	}
	// Unseen position inherits lowest measured (0).
	if a := c.acceptance(3); a != 0.0 {
		t.Fatalf("acceptance(3) = %v, want 0 (inherited)", a)
	}
}
