package diffusiongemma

import (
	"math"
	"math/rand"
	"testing"
)

// --- Mock LogitsProvider ---

// mockLogitsProvider simulates a model that produces logits for diffusion decoding.
// It uses a tiny randomly-initialized "model" with the specified config.
type mockLogitsProvider struct {
	vocabSize  int
	blockSize  int
	hiddenSize int
	numLayers  int
	numExperts int

	// weights simulates random weight matrices [numLayers][hiddenSize][vocabSize]
	weights [][][]float32

	rng *rand.Rand
}

type mockConfig struct {
	vocabSize  int
	blockSize  int
	hiddenSize int
	numLayers  int
	numExperts int
	seed       int64
}

func newMockLogitsProvider(cfg mockConfig) *mockLogitsProvider {
	rng := rand.New(rand.NewSource(cfg.seed))

	// Initialize random weights for each layer
	weights := make([][][]float32, cfg.numLayers)
	for l := range cfg.numLayers {
		weights[l] = make([][]float32, cfg.hiddenSize)
		for h := 0; h < cfg.hiddenSize; h++ {
			weights[l][h] = make([]float32, cfg.vocabSize)
			for v := 0; v < cfg.vocabSize; v++ {
				weights[l][h][v] = float32(rng.NormFloat64()) * 0.02
			}
		}
	}

	return &mockLogitsProvider{
		vocabSize:  cfg.vocabSize,
		blockSize:  cfg.blockSize,
		hiddenSize: cfg.hiddenSize,
		numLayers:  cfg.numLayers,
		numExperts: cfg.numExperts,
		weights:    weights,
		rng:        rng,
	}
}

// ComputeBlockLogits implements LogitsProvider for the mock model.
// For each token position, it computes logits by hashing the token ID
// through the "weight matrices" to produce a distribution over the vocabulary.
func (m *mockLogitsProvider) ComputeBlockLogits(prefixTokens, blockTokens []int32) ([]float32, error) {
	logits := make([]float32, len(blockTokens)*m.vocabSize)

	for pos, tokenID := range blockTokens {
		// Simple embedding: use token ID to index into hidden dimension
		hidden := make([]float32, m.hiddenSize)
		for h := 0; h < m.hiddenSize; h++ {
			// Hash-based embedding: deterministic but varied per token+position
			hidden[h] = float32(math.Sin(float64(tokenID)*0.1+float64(h)*0.3+float64(pos)*0.7)) * 0.5
		}

		// Pass through each layer
		for l := 0; l < m.numLayers; l++ {
			newHidden := make([]float32, m.hiddenSize)
			for h := 0; h < m.hiddenSize; h++ {
				sum := float32(0)
				for hh := 0; hh < m.hiddenSize && hh < len(m.weights[l]); hh++ {
					if h < len(m.weights[l][hh]) {
						sum += hidden[hh] * m.weights[l][hh][h%m.vocabSize]
					}
				}
				// GELU-like activation
				newHidden[h] = sum * float32(1.0/(1.0+math.Exp(-float64(sum)*1.702)))
			}
			// Residual connection
			for h := 0; h < m.hiddenSize; h++ {
				hidden[h] += newHidden[h]
			}
		}

		// Project to vocabulary logits
		for v := 0; v < m.vocabSize; v++ {
			sum := float32(0)
			for h := 0; h < m.hiddenSize; h++ {
				if h < len(m.weights[0]) && v < len(m.weights[0][h]) {
					sum += hidden[h] * m.weights[0][h][v]
				}
			}
			logits[pos*m.vocabSize+v] = sum
		}
	}

	return logits, nil
}

// --- Tests ---

func TestComputeEntropy(t *testing.T) {
	tests := []struct {
		name     string
		logits   []float32
		wantLow  float64 // entropy should be >= this
		wantHigh float64 // entropy should be <= this
	}{
		{
			name:     "uniform distribution has high entropy",
			logits:   []float32{1.0, 1.0, 1.0, 1.0},
			wantLow:  1.3, // ln(4) ≈ 1.386
			wantHigh: 1.4,
		},
		{
			name:     "peaked distribution has low entropy",
			logits:   []float32{100.0, 0.0, 0.0, 0.0},
			wantLow:  0.0,
			wantHigh: 0.001,
		},
		{
			name:     "moderately peaked distribution",
			logits:   []float32{5.0, 1.0, 0.0, 0.0},
			wantLow:  0.0,
			wantHigh: 0.2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			entropy := ComputeEntropy(tt.logits)
			if entropy < tt.wantLow || entropy > tt.wantHigh {
				t.Errorf("ComputeEntropy() = %f, want in [%f, %f]", entropy, tt.wantLow, tt.wantHigh)
			}
		})
	}
}

func TestArgMax(t *testing.T) {
	tests := []struct {
		name   string
		logits []float32
		want   int
	}{
		{"first element", []float32{5.0, 1.0, 2.0, 3.0}, 0},
		{"last element", []float32{1.0, 2.0, 3.0, 5.0}, 3},
		{"middle element", []float32{1.0, 5.0, 2.0, 3.0}, 1},
		{"single element", []float32{42.0}, 0},
		{"negative values", []float32{-1.0, -0.5, -2.0}, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ArgMax(tt.logits)
			if got != tt.want {
				t.Errorf("ArgMax() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestNewBlockState(t *testing.T) {
	maskID := int32(999)
	bs := NewBlockState(8, maskID)

	if len(bs.Tokens) != 8 {
		t.Fatalf("expected 8 tokens, got %d", len(bs.Tokens))
	}
	if bs.NumMasked != 8 {
		t.Fatalf("expected 8 masked, got %d", bs.NumMasked)
	}
	for i, tok := range bs.Tokens {
		if tok != maskID {
			t.Errorf("token %d = %d, want %d", i, tok, maskID)
		}
		if !bs.Masked[i] {
			t.Errorf("position %d should be masked", i)
		}
	}
	if bs.IsFullyUnmasked() {
		t.Error("fresh block should not be fully unmasked")
	}
}

func TestDenoiseBlockMonotonicUnmasking(t *testing.T) {
	// Config: tiny model, 2 layers, hidden size 64, 2 experts, block size 8
	cfg := mockConfig{
		vocabSize:  32,
		blockSize:  8,
		hiddenSize: 64,
		numLayers:  2,
		numExperts: 2,
		seed:       42,
	}
	provider := newMockLogitsProvider(cfg)
	maskID := int32(31) // last token in vocab as mask

	block := NewBlockState(cfg.blockSize, maskID)

	// Track unmasking history to verify monotonicity
	prevMaskedCount := block.NumMasked
	stepHistory := []int{}

	callback := func(blockIndex, step int, tokens []int32, masked []bool) bool {
		maskedCount := 0
		for _, m := range masked {
			if m {
				maskedCount++
			}
		}
		// Monotonic check: masked count should never increase
		if maskedCount > prevMaskedCount {
			t.Errorf("step %d: masked count increased from %d to %d (re-masking detected!)",
				step, prevMaskedCount, maskedCount)
		}
		prevMaskedCount = maskedCount
		stepHistory = append(stepHistory, maskedCount)
		return true
	}

	err := DenoiseBlock(
		provider,
		nil, // no prefix
		block,
		cfg.vocabSize,
		0.1, // entropy bound
		48,  // max steps
		callback,
		0,
	)
	if err != nil {
		t.Fatalf("DenoiseBlock failed: %v", err)
	}

	// Assert 1: all tokens should be unmasked
	if !block.IsFullyUnmasked() {
		t.Errorf("block should be fully unmasked, %d still masked", block.NumMasked)
	}

	// Assert 2: no token should be the mask token
	for i, tok := range block.Tokens {
		if tok == maskID {
			t.Errorf("position %d still has mask token", i)
		}
	}

	// Assert 3: step count should be within max_steps
	if block.Step > 48 {
		t.Errorf("block took %d steps, exceeds max of 48", block.Step)
	}

	t.Logf("Block denoised in %d steps, history: %v", block.Step, stepHistory)
}

func TestDenoiseBlockTerminatesWithinMaxSteps(t *testing.T) {
	cfg := mockConfig{
		vocabSize:  16,
		blockSize:  8,
		hiddenSize: 32,
		numLayers:  2,
		numExperts: 2,
		seed:       123,
	}
	provider := newMockLogitsProvider(cfg)
	maskID := int32(15)

	block := NewBlockState(cfg.blockSize, maskID)

	maxSteps := 48
	err := DenoiseBlock(provider, nil, block, cfg.vocabSize, 0.1, maxSteps, nil, 0)
	if err != nil {
		t.Fatalf("DenoiseBlock failed: %v", err)
	}

	if block.Step > maxSteps {
		t.Errorf("block took %d steps, should be <= %d", block.Step, maxSteps)
	}

	// After denoising, all positions must be unmasked (force-unmask fallback)
	for i, m := range block.Masked {
		if m {
			t.Errorf("position %d still masked after denoising", i)
		}
	}
}

func TestRunDiffusionDecodeOutputLength(t *testing.T) {
	cfg := mockConfig{
		vocabSize:  32,
		blockSize:  8,
		hiddenSize: 64,
		numLayers:  2,
		numExperts: 2,
		seed:       99,
	}
	provider := newMockLogitsProvider(cfg)
	maskID := int32(31)

	numBlocks := 3
	result, err := RunDiffusionDecode(
		provider,
		numBlocks,
		cfg.blockSize,
		maskID,
		cfg.vocabSize,
		0.1,
		48,
		nil,
	)
	if err != nil {
		t.Fatalf("RunDiffusionDecode failed: %v", err)
	}

	// Assert 1: output length is exactly numBlocks * blockSize
	expectedLen := numBlocks * cfg.blockSize
	if len(result.Tokens) != expectedLen {
		t.Errorf("expected %d tokens, got %d", expectedLen, len(result.Tokens))
	}

	// Assert 2: correct number of blocks generated
	if result.NumBlocks != numBlocks {
		t.Errorf("expected %d blocks, got %d", numBlocks, result.NumBlocks)
	}

	// Assert 3: no mask tokens in output
	for i, tok := range result.Tokens {
		if tok == maskID {
			t.Errorf("position %d still has mask token in final output", i)
		}
	}

	// Assert 4: each block terminated within max steps
	for i, steps := range result.StepsPerBlock {
		if steps > 48 {
			t.Errorf("block %d took %d steps, exceeds max of 48", i, steps)
		}
	}

	t.Logf("Generated %d tokens in %d blocks, steps per block: %v, total steps: %d",
		len(result.Tokens), result.NumBlocks, result.StepsPerBlock, result.TotalSteps)
}

func TestRunDiffusionDecodeMonotonicAcrossBlocks(t *testing.T) {
	cfg := mockConfig{
		vocabSize:  16,
		blockSize:  4,
		hiddenSize: 32,
		numLayers:  2,
		numExperts: 2,
		seed:       77,
	}
	provider := newMockLogitsProvider(cfg)
	maskID := int32(15)

	// Track all step callbacks
	type stepRecord struct {
		blockIdx    int
		step        int
		maskedCount int
	}
	var records []stepRecord

	callback := func(blockIndex, step int, tokens []int32, masked []bool) bool {
		maskedCount := 0
		for _, m := range masked {
			if m {
				maskedCount++
			}
		}
		records = append(records, stepRecord{blockIndex, step, maskedCount})
		return true
	}

	result, err := RunDiffusionDecode(
		provider,
		4, // 4 blocks
		cfg.blockSize,
		maskID,
		cfg.vocabSize,
		0.1,
		48,
		callback,
	)
	if err != nil {
		t.Fatalf("RunDiffusionDecode failed: %v", err)
	}

	// Verify monotonicity within each block
	prevByBlock := make(map[int]int)
	for _, r := range records {
		if prev, ok := prevByBlock[r.blockIdx]; ok {
			if r.maskedCount > prev {
				t.Errorf("block %d step %d: masked increased from %d to %d",
					r.blockIdx, r.step, prev, r.maskedCount)
			}
		}
		prevByBlock[r.blockIdx] = r.maskedCount
	}

	// Verify output
	expectedLen := 4 * cfg.blockSize
	if len(result.Tokens) != expectedLen {
		t.Errorf("expected %d tokens, got %d", expectedLen, len(result.Tokens))
	}
}

func TestSoftmaxProbs(t *testing.T) {
	logits := []float32{1.0, 2.0, 3.0}
	probs := SoftmaxProbs(logits)

	// Probabilities should sum to ~1.0
	sum := 0.0
	for _, p := range probs {
		sum += p
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("softmax probs sum to %f, want 1.0", sum)
	}

	// Higher logits should get higher probabilities
	if probs[0] >= probs[1] || probs[1] >= probs[2] {
		t.Errorf("expected monotonically increasing probs, got %v", probs)
	}
}

func TestNewConfig(t *testing.T) {
	cfg := testConfig{
		"diffusiongemma.block_count":                      uint32(2),
		"diffusiongemma.embedding_length":                 uint32(64),
		"diffusiongemma.attention.key_length":             uint32(16),
		"diffusiongemma.attention.head_count":             uint32(4),
		"diffusiongemma.attention.head_count_kv":          uint32(2),
		"diffusiongemma.attention.layer_norm_rms_epsilon": float32(1e-6),
		"diffusiongemma.expert_count":                     uint32(2),
		"diffusiongemma.expert_used_count":                uint32(1),
		"diffusiongemma.diffusion.block_size":             uint32(8),
		"diffusiongemma.diffusion.eb_max_steps":           uint32(48),
		"diffusiongemma.diffusion.entropy_bound":          float32(0.1),
	}

	tm := newTextModel(cfg)

	if tm.hiddenSize != 64 {
		t.Errorf("hiddenSize = %d, want 64", tm.hiddenSize)
	}
	if tm.numHeads != 4 {
		t.Errorf("numHeads = %d, want 4", tm.numHeads)
	}
	if tm.numKVHeads != 2 {
		t.Errorf("numKVHeads = %d, want 2", tm.numKVHeads)
	}
	if tm.hiddenLayers != 2 {
		t.Errorf("hiddenLayers = %d, want 2", tm.hiddenLayers)
	}
	if tm.numExperts != 2 {
		t.Errorf("numExperts = %d, want 2", tm.numExperts)
	}
	if tm.numExpertsUsed != 1 {
		t.Errorf("numExpertsUsed = %d, want 1", tm.numExpertsUsed)
	}
	if len(tm.Layers) != 2 {
		t.Errorf("layers = %d, want 2", len(tm.Layers))
	}
}

func TestDenoiseBlockEarlyStop(t *testing.T) {
	// Use a provider that always produces very peaked logits (low entropy)
	// so all tokens unmask in a single step
	provider := &peakedLogitsProvider{vocabSize: 8}
	maskID := int32(7)
	blockSize := 4

	block := NewBlockState(blockSize, maskID)

	err := DenoiseBlock(provider, nil, block, 8, 0.1, 48, nil, 0)
	if err != nil {
		t.Fatalf("DenoiseBlock failed: %v", err)
	}

	// Should finish in very few steps due to adaptive stopping
	if block.Step > 3 {
		t.Errorf("expected early stop (<=3 steps), got %d steps", block.Step)
	}
	if !block.IsFullyUnmasked() {
		t.Error("block should be fully unmasked")
	}
}

// peakedLogitsProvider always returns highly peaked logits so entropy is near zero.
type peakedLogitsProvider struct {
	vocabSize int
}

func (p *peakedLogitsProvider) ComputeBlockLogits(prefix, block []int32) ([]float32, error) {
	logits := make([]float32, len(block)*p.vocabSize)
	for pos := range block {
		for v := 0; v < p.vocabSize; v++ {
			if v == pos%p.vocabSize {
				logits[pos*p.vocabSize+v] = 100.0 // very peaked
			} else {
				logits[pos*p.vocabSize+v] = -100.0
			}
		}
	}
	return logits, nil
}

func TestDiffusionOptionsDefaults(t *testing.T) {
	cfg := testConfig{
		"diffusiongemma.block_count":             uint32(1),
		"diffusiongemma.embedding_length":        uint32(32),
		"diffusiongemma.attention.key_length":    uint32(8),
		"diffusiongemma.attention.head_count":    uint32(2),
		"diffusiongemma.attention.head_count_kv": uint32(1),
	}

	// Test that defaults are applied when GGUF metadata is missing
	m, err := New(cfg)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	dm := m.(*Model)

	if dm.BlockSize != BlockSize {
		t.Errorf("BlockSize = %d, want default %d", dm.BlockSize, BlockSize)
	}
	if dm.EBMaxSteps != 48 {
		t.Errorf("EBMaxSteps = %d, want 48", dm.EBMaxSteps)
	}
	if math.Abs(dm.EntropyBound-0.1) > 1e-6 {
		t.Errorf("EntropyBound = %f, want ~0.1", dm.EntropyBound)
	}
}
