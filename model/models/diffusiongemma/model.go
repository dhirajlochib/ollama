package diffusiongemma

import (
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/tokenizer"
)

// BlockSize is the default number of tokens generated in parallel per diffusion block.
const BlockSize = 256

// Model implements the DiffusionGemma block-diffusion architecture.
// It reuses the Gemma4 MoE transformer backbone but replaces autoregressive
// decoding with iterative parallel denoising over fixed-size token blocks.
type Model struct {
	model.Base
	tokenizer.Tokenizer

	*TextModel

	DiffusionOptions
}

// DiffusionOptions holds diffusion-specific hyperparameters read from GGUF metadata.
type DiffusionOptions struct {
	BlockSize     int     // tokens per diffusion block (default 256)
	EBMaxSteps    int     // maximum denoising iterations per block (default 48)
	EntropyBound  float64 // entropy threshold for unmasking (default 0.1)
	MaskTokenID   int32   // token ID used for [MASK]
	NumVocab      int     // vocabulary size
}

func New(c fs.Config) (model.Model, error) {
	vocabulary := tokenizer.Vocabulary{
		Values: c.Strings("tokenizer.ggml.tokens"),
		Scores: c.Floats("tokenizer.ggml.scores"),
		Types:  c.Ints("tokenizer.ggml.token_type"),
		Merges: c.Strings("tokenizer.ggml.merges"),
		AddBOS: c.Bool("tokenizer.ggml.add_bos_token", false),
		BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
		AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
		EOS: append(
			[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
			c.Ints("tokenizer.ggml.eos_token_ids")...,
		),
	}

	t := tokenizer.NewBytePairEncodingWithOptions(&vocabulary, []string{},
		tokenizer.WithSentencePieceNormalizer())

	// Look up [MASK] token ID; fall back to metadata key
	maskTokenID := int32(c.Uint("tokenizer.ggml.mask_token_id", 0))
	if maskTokenID == 0 {
		for i, tok := range vocabulary.Values {
			if tok == "[MASK]" || tok == "<mask>" {
				maskTokenID = int32(i)
				break
			}
		}
	}

	blockSize := int(c.Uint("diffusion.block_size", BlockSize))
	ebMaxSteps := int(c.Uint("diffusion.eb_max_steps", 48))
	entropyBound := float64(c.Float("diffusion.entropy_bound", 0.1))

	m := Model{
		Tokenizer: t,
		TextModel: newTextModel(c),
		DiffusionOptions: DiffusionOptions{
			BlockSize:    blockSize,
			EBMaxSteps:   ebMaxSteps,
			EntropyBound: entropyBound,
			MaskTokenID:  maskTokenID,
			NumVocab:     len(vocabulary.Values),
		},
	}

	slidingWindowLen := int32(c.Uint("attention.sliding_window", 0))
	if slidingWindowLen > 0 {
		m.Cache = kvcache.NewWrapperCache(
			kvcache.NewSWAMemCache(slidingWindowLen, 4096, m.Shift),
			kvcache.NewCausalCache(m.Shift),
		)
	} else {
		m.Cache = kvcache.NewCausalCache(m.Shift)
	}

	return &m, nil
}

// Forward performs a single denoising step over the current batch.
// For diffusion, the batch contains an entire block of (possibly masked) tokens.
// The model computes logits for all positions using bidirectional attention
// within the current block and causal attention to previously finalized blocks.
func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	hiddenState := m.TextModel.Forward(ctx, batch, m.Cache)
	hiddenState = m.TextModel.Output.Forward(ctx, hiddenState)

	if m.TextModel.TextOptions.finalLogitSoftcap > 0.0 {
		hiddenState = hiddenState.Scale(ctx, 1.0/float64(m.TextModel.TextOptions.finalLogitSoftcap))
		hiddenState = hiddenState.Tanh(ctx)
		hiddenState = hiddenState.Scale(ctx, float64(m.TextModel.TextOptions.finalLogitSoftcap))
	}

	return hiddenState, nil
}

func (m *Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	ropeBase, ropeDims := m.TextModel.ropeForLayer(layer)
	return nn.RoPE(ctx, key, shift, ropeDims, ropeBase, 1.0, rope.WithTypeNeoX()), nil
}

func init() {
	model.Register("diffusiongemma", New)
}

// --- Utility functions for diffusion decoding ---

// ComputeEntropy calculates per-token entropy from a logits slice.
// logits is [vocab_size] for a single token position.
func ComputeEntropy(logits []float32) float64 {
	// Numerically stable softmax + entropy
	maxVal := float32(-math.MaxFloat32)
	for _, v := range logits {
		if v > maxVal {
			maxVal = v
		}
	}

	sumExp := float64(0)
	for _, v := range logits {
		sumExp += math.Exp(float64(v - maxVal))
	}
	logSumExp := math.Log(sumExp) + float64(maxVal)

	entropy := float64(0)
	for _, v := range logits {
		p := math.Exp(float64(v) - logSumExp)
		if p > 0 {
			entropy -= p * math.Log(p)
		}
	}
	return entropy
}

// ArgMax returns the index of the maximum value in a float32 slice.
func ArgMax(logits []float32) int {
	bestIdx := 0
	bestVal := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > bestVal {
			bestVal = logits[i]
			bestIdx = i
		}
	}
	return bestIdx
}
