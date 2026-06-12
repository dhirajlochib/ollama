// Package diffusiongemma implements the Entropy Bounded Denoising (EBD)
// decode loop for DiffusionGemma block-diffusion models.
//
// Instead of generating tokens one by one (autoregressive), this loop
// generates an entire block of BLOCK_SIZE tokens in parallel by:
//
//  1. Starting with all positions in the block set to [MASK]
//  2. Running a forward pass to get logits for all positions
//  3. Computing per-token entropy from the logits
//  4. Unmasking tokens whose entropy falls below the entropy bound
//  5. Repeating until the block is fully unmasked or max steps is reached
//  6. Advancing to the next block and repeating
//
// Tokens unmask monotonically: once a token is unmasked it is never re-masked.
// The Adaptive Stopping criterion terminates early when all positions are resolved.
package diffusiongemma

import (
	"fmt"
	"math"
)

// BlockState tracks the state of a single diffusion block during denoising.
type BlockState struct {
	// Tokens holds the current token IDs for this block.
	// Masked positions contain the mask token ID.
	Tokens []int32

	// Masked tracks which positions are still masked (true = masked).
	Masked []bool

	// NumMasked is the count of positions still masked.
	NumMasked int

	// Step is the current denoising iteration within this block.
	Step int
}

// NewBlockState creates a fully-masked block of the given size.
func NewBlockState(blockSize int, maskTokenID int32) *BlockState {
	tokens := make([]int32, blockSize)
	masked := make([]bool, blockSize)
	for i := range tokens {
		tokens[i] = maskTokenID
		masked[i] = true
	}
	return &BlockState{
		Tokens:    tokens,
		Masked:    masked,
		NumMasked: blockSize,
		Step:      0,
	}
}

// IsFullyUnmasked returns true if all positions have been resolved.
func (bs *BlockState) IsFullyUnmasked() bool {
	return bs.NumMasked == 0
}

// DiffusionResult holds the output of a full diffusion decode run.
type DiffusionResult struct {
	// Tokens contains all generated tokens across all blocks.
	Tokens []int32

	// NumBlocks is how many blocks were generated.
	NumBlocks int

	// StepsPerBlock records how many denoising steps each block took.
	StepsPerBlock []int

	// TotalSteps is the sum of all denoising steps across all blocks.
	TotalSteps int
}

// StepCallback is called after each denoising step with the current block state.
// The callback receives: blockIndex, step, current tokens, masked positions.
// Return false to abort generation.
type StepCallback func(blockIndex int, step int, tokens []int32, masked []bool) bool

// LogitsProvider abstracts the model forward pass for the decode loop.
// Given a sequence of token IDs (prefix + current block), it returns
// logits of shape [blockSize][vocabSize] for the current block positions.
type LogitsProvider interface {
	// ComputeBlockLogits runs the model on the full sequence and returns
	// logits for the current block positions.
	// prefixTokens: finalized tokens from previous blocks
	// blockTokens: current block tokens (may include masks)
	// Returns logits as a flattened [blockSize * vocabSize] slice.
	ComputeBlockLogits(prefixTokens, blockTokens []int32) ([]float32, error)
}

// DenoiseBlock runs the Entropy Bounded Denoising loop on a single block.
//
// Parameters:
//   - provider: computes logits for the current block given prefix+block tokens
//   - prefixTokens: finalized tokens from all previous blocks
//   - block: the BlockState to denoise (modified in place)
//   - vocabSize: vocabulary size for indexing into logits
//   - entropyBound: entropy threshold below which tokens are unmasked (default 0.1)
//   - maxSteps: maximum number of denoising iterations (default 48)
//   - callback: optional per-step callback for streaming (may be nil)
//   - blockIndex: index of this block (passed to callback)
//
// Returns an error if the provider fails.
func DenoiseBlock(
	provider LogitsProvider,
	prefixTokens []int32,
	block *BlockState,
	vocabSize int,
	entropyBound float64,
	maxSteps int,
	callback StepCallback,
	blockIndex int,
) error {
	for block.Step < maxSteps {
		// Adaptive Stopping: if all tokens are unmasked, stop early
		if block.IsFullyUnmasked() {
			break
		}

		// Run forward pass to get logits for all block positions
		logits, err := provider.ComputeBlockLogits(prefixTokens, block.Tokens)
		if err != nil {
			return fmt.Errorf("diffusion step %d: %w", block.Step, err)
		}

		// Validate logits shape
		expectedLen := len(block.Tokens) * vocabSize
		if len(logits) != expectedLen {
			return fmt.Errorf("diffusion step %d: expected %d logits, got %d",
				block.Step, expectedLen, len(logits))
		}

		// For each masked position, compute entropy and potentially unmask
		unmaskedThisStep := 0
		for pos := range block.Tokens {
			if !block.Masked[pos] {
				continue // already unmasked, skip
			}

			// Extract logits for this position
			posLogits := logits[pos*vocabSize : (pos+1)*vocabSize]

			// Compute entropy
			entropy := ComputeEntropy(posLogits)

			// If entropy is below threshold, unmask with argmax token
			if entropy < entropyBound {
				bestToken := int32(ArgMax(posLogits))
				block.Tokens[pos] = bestToken
				block.Masked[pos] = false
				block.NumMasked--
				unmaskedThisStep++
			}
		}

		block.Step++

		// Call the step callback for streaming updates
		if callback != nil {
			if !callback(blockIndex, block.Step, block.Tokens, block.Masked) {
				break // caller requested abort
			}
		}

		// If nothing was unmasked this step and entropy bound is very tight,
		// relax the bound slightly to make progress (exponential schedule)
		if unmaskedThisStep == 0 && block.NumMasked > 0 {
			// Increase entropy bound by 50% to allow more tokens through
			entropyBound *= 1.5
		}
	}

	// Force-unmask any remaining positions with argmax (fallback)
	if block.NumMasked > 0 {
		logits, err := provider.ComputeBlockLogits(prefixTokens, block.Tokens)
		if err != nil {
			return fmt.Errorf("diffusion force-unmask: %w", err)
		}
		for pos := range block.Tokens {
			if block.Masked[pos] {
				posLogits := logits[pos*vocabSize : (pos+1)*vocabSize]
				block.Tokens[pos] = int32(ArgMax(posLogits))
				block.Masked[pos] = false
				block.NumMasked--
			}
		}
	}

	return nil
}

// RunDiffusionDecode runs the full diffusion decode loop, generating
// numBlocks blocks of tokens sequentially. Each block is denoised in
// parallel using the EBD loop.
//
// Parameters:
//   - provider: computes logits
//   - numBlocks: how many blocks to generate
//   - blockSize: tokens per block
//   - maskTokenID: the [MASK] token ID
//   - vocabSize: vocabulary size
//   - entropyBound: entropy threshold (default 0.1)
//   - maxSteps: max denoising steps per block (default 48)
//   - callback: optional per-step callback
//
// Returns the accumulated DiffusionResult.
func RunDiffusionDecode(
	provider LogitsProvider,
	numBlocks int,
	blockSize int,
	maskTokenID int32,
	vocabSize int,
	entropyBound float64,
	maxSteps int,
	callback StepCallback,
) (*DiffusionResult, error) {
	result := &DiffusionResult{
		StepsPerBlock: make([]int, 0, numBlocks),
	}

	var prefixTokens []int32

	for blockIdx := range numBlocks {
		block := NewBlockState(blockSize, maskTokenID)

		err := DenoiseBlock(
			provider,
			prefixTokens,
			block,
			vocabSize,
			entropyBound,
			maxSteps,
			callback,
			blockIdx,
		)
		if err != nil {
			return result, fmt.Errorf("block %d: %w", blockIdx, err)
		}

		// Append finalized block tokens to prefix and result
		prefixTokens = append(prefixTokens, block.Tokens...)
		result.Tokens = append(result.Tokens, block.Tokens...)
		result.StepsPerBlock = append(result.StepsPerBlock, block.Step)
		result.TotalSteps += block.Step
		result.NumBlocks++
	}

	return result, nil
}

// --- Entropy helpers for the decode loop ---

// SoftmaxProbs computes softmax probabilities from logits (numerically stable).
func SoftmaxProbs(logits []float32) []float64 {
	maxVal := float32(-math.MaxFloat32)
	for _, v := range logits {
		if v > maxVal {
			maxVal = v
		}
	}

	probs := make([]float64, len(logits))
	sumExp := float64(0)
	for i, v := range logits {
		probs[i] = math.Exp(float64(v - maxVal))
		sumExp += probs[i]
	}
	for i := range probs {
		probs[i] /= sumExp
	}
	return probs
}
