package diffusiongemma

import (
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model/input"
)

const (
	cacheTypeSWA = iota
	cacheTypeCausal
)

// TextOptions holds transformer backbone configuration, reusing the Gemma4 MoE
// layout with additions for diffusion-specific per-block timestep embedding.
type TextOptions struct {
	hiddenSize             int
	numHeads, numKVHeads   int
	numGlobalKVHeads       int
	headDim, globalHeadDim int
	hiddenLayers           int

	eps               float32
	ropeBase          float32
	ropeLocalBase     float32
	partialRotaryDims int

	slidingWindowPattern []bool

	finalLogitSoftcap float32

	numExperts     int
	numExpertsUsed int
}

func (o *TextOptions) isLocal(layer int) bool {
	if layer < len(o.slidingWindowPattern) {
		return o.slidingWindowPattern[layer]
	}
	return false
}

func (o *TextOptions) ropeForLayer(layer int) (base float32, dims int) {
	if o.isLocal(layer) {
		return o.ropeLocalBase, o.headDim
	}
	return o.ropeBase, o.partialRotaryDims
}

func (o *TextOptions) kvHeadsForLayer(layer int) int {
	if o.isLocal(layer) {
		return o.numKVHeads
	}
	if o.numGlobalKVHeads > 0 {
		return o.numGlobalKVHeads
	}
	return o.numKVHeads
}

func (o *TextOptions) headDimForLayer(layer int) int {
	if o.isLocal(layer) {
		return o.headDim
	}
	return o.globalHeadDim
}

// TextModel is the core transformer backbone for DiffusionGemma.
// It mirrors the Gemma4 text model but supports bidirectional attention
// within diffusion blocks via mask configuration.
type TextModel struct {
	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Layers         []TextLayer   `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	// Diffusion-specific: timestep embedding for conditioning on denoising step
	TimestepEmbedding *nn.Linear `gguf:"diffusion_timestep_embd"`
	NoiseEmbedding    *nn.Linear `gguf:"diffusion_noise_embd"`

	TextOptions
}

func newTextModel(c fs.Config) *TextModel {
	numLayers := int(c.Uint("block_count"))

	globalHeadDim := int(c.Uint("attention.key_length", 256))
	headDim := int(c.Uint("attention.key_length_swa", 256))
	if headDim == 0 {
		headDim = globalHeadDim
	}

	partialRotaryDims := int(c.Uint("rope.dimension_count", 0))
	if partialRotaryDims == 0 {
		partialFactor := c.Float("rope.partial_rotary_factor", 1.0)
		partialRotaryDims = int(float32(globalHeadDim) * partialFactor)
	}

	ropeBase := c.Float("rope.freq_base", 1000000.0)
	ropeLocalBase := c.Float("rope.freq_base_swa", 0)
	if ropeLocalBase == 0 {
		ropeLocalBase = c.Float("rope.local.freq_base", 10000.0)
	}

	numGlobalKVHeads := int(c.Uint("attention.global_head_count_kv", 0))
	slidingPattern := c.Bools("attention.sliding_window_pattern")

	numKVHeads := int(c.Uint("attention.head_count_kv", 0))

	return &TextModel{
		Layers: make([]TextLayer, numLayers),
		TextOptions: TextOptions{
			hiddenSize:           int(c.Uint("embedding_length")),
			numHeads:             int(c.Uint("attention.head_count")),
			numKVHeads:           numKVHeads,
			numGlobalKVHeads:     numGlobalKVHeads,
			headDim:              headDim,
			globalHeadDim:        globalHeadDim,
			hiddenLayers:         numLayers,
			eps:                  c.Float("attention.layer_norm_rms_epsilon", 1e-06),
			ropeBase:             ropeBase,
			ropeLocalBase:        ropeLocalBase,
			partialRotaryDims:    partialRotaryDims,
			slidingWindowPattern: slidingPattern,
			finalLogitSoftcap:    c.Float("final_logit_softcapping", 0.0),
			numExperts:           int(c.Uint("expert_count", 0)),
			numExpertsUsed:       int(c.Uint("expert_used_count", 0)),
		},
	}
}

// Forward runs the transformer backbone on a batch of tokens.
// For diffusion, this processes the entire block at once (not token-by-token).
// The cache/mask configuration handles the hybrid attention pattern:
//   - Bidirectional within the current diffusion block
//   - Causal to previously finalized prefix blocks
func (m *TextModel) Forward(ctx ml.Context, batch input.Batch, cache kvcache.Cache) ml.Tensor {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	hiddenState := m.TokenEmbedding.Forward(ctx, batch.Inputs)
	hiddenState = hiddenState.Scale(ctx, math.Sqrt(float64(m.hiddenSize)))

	for i := range len(m.Layers) {
		layer := m.Layers[i]
		if cache != nil {
			cache.SetLayer(i)
			if wc, ok := cache.(*kvcache.WrapperCache); ok {
				cacheType := cacheTypeSWA
				if !m.isLocal(i) {
					cacheType = cacheTypeCausal
				}
				wc.SetLayerType(cacheType)
			}
		}

		var lastLayerOutputs ml.Tensor
		if i == len(m.Layers)-1 {
			lastLayerOutputs = batch.Outputs
		}

		hiddenState = layer.Forward(ctx, i, hiddenState, positions, lastLayerOutputs, cache, &m.TextOptions)
	}

	return m.OutputNorm.Forward(ctx, hiddenState, m.eps)
}

// --- Self-Attention ---

type TextSelfAttention struct {
	Query     *nn.Linear  `gguf:"attn_q"`
	QueryNorm *nn.RMSNorm `gguf:"attn_q_norm"`
	Key       *nn.Linear  `gguf:"attn_k"`
	KeyNorm   *nn.RMSNorm `gguf:"attn_k_norm"`
	Value     *nn.Linear  `gguf:"attn_v"`
	Output    *nn.Linear  `gguf:"attn_output"`
}

func (sa *TextSelfAttention) Forward(ctx ml.Context, layer int, hiddenState, positions ml.Tensor, cache kvcache.Cache, opts *TextOptions) ml.Tensor {
	batchSize := hiddenState.Dim(1)
	hd := opts.headDimForLayer(layer)
	kvHeads := opts.kvHeadsForLayer(layer)
	ropeBase, ropeDims := opts.ropeForLayer(layer)

	q := sa.Query.Forward(ctx, hiddenState)
	q = q.Reshape(ctx, hd, opts.numHeads, batchSize)
	q = sa.QueryNorm.Forward(ctx, q, opts.eps)

	k := sa.Key.Forward(ctx, hiddenState)
	k = k.Reshape(ctx, hd, kvHeads, batchSize)

	v := sa.Value.Forward(ctx, hiddenState)
	v = v.Reshape(ctx, hd, kvHeads, batchSize)

	k = sa.KeyNorm.Forward(ctx, k, opts.eps)

	q = nn.RoPE(ctx, q, positions, ropeDims, ropeBase, 1.0, rope.WithTypeNeoX())
	k = nn.RoPE(ctx, k, positions, ropeDims, ropeBase, 1.0, rope.WithTypeNeoX())

	attention := nn.Attention(ctx, q, k, v, 1.0, cache)
	attention = attention.Reshape(ctx, hd*opts.numHeads, batchSize)
	return sa.Output.Forward(ctx, attention)
}

// --- MLP ---

type TextMLP struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *TextMLP) Forward(ctx ml.Context, hiddenState ml.Tensor) ml.Tensor {
	hiddenState = mlp.Gate.Forward(ctx, hiddenState).GELU(ctx, mlp.Up.Forward(ctx, hiddenState))
	return mlp.Down.Forward(ctx, hiddenState)
}

// --- MoE Router ---

type TextRouter struct {
	Proj  *nn.Linear `gguf:"ffn_gate_inp"`
	Scale ml.Tensor  `gguf:"ffn_gate_inp.scale"`
}

func (r *TextRouter) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *TextOptions) (routingWeights, selectedExperts ml.Tensor) {
	x := hiddenState.RMSNorm(ctx, nil, opts.eps)
	x = x.Scale(ctx, 1.0/math.Sqrt(float64(opts.hiddenSize)))
	if r.Scale != nil {
		x = x.Mul(ctx, r.Scale)
	}
	expertScores := r.Proj.Forward(ctx, x)
	routingWeights = expertScores.Softmax(ctx)
	selectedExperts = routingWeights.TopK(ctx, opts.numExpertsUsed)
	return routingWeights, selectedExperts
}

// --- MoE Expert Block ---

type TextMoEBlock struct {
	GateUp    *nn.LinearBatch `gguf:"ffn_gate_up_exps"`
	Gate      *nn.LinearBatch `gguf:"ffn_gate_exps"`
	Up        *nn.LinearBatch `gguf:"ffn_up_exps"`
	Down      *nn.LinearBatch `gguf:"ffn_down_exps"`
	DownScale ml.Tensor       `gguf:"ffn_down_exps.scale,alt:ffn_gate_inp.per_expert_scale"`
}

func (moe *TextMoEBlock) Forward(ctx ml.Context, hiddenState, routingWeights, selectedExperts ml.Tensor, opts *TextOptions) ml.Tensor {
	routingWeights = routingWeights.Reshape(ctx, 1, opts.numExperts, hiddenState.Dim(1)).Rows(ctx, selectedExperts)
	routingWeights = routingWeights.Reshape(ctx, opts.numExpertsUsed, hiddenState.Dim(1))
	routingWeights = routingWeights.Div(ctx, routingWeights.SumRows(ctx))
	routingWeights = routingWeights.Reshape(ctx, 1, opts.numExpertsUsed, hiddenState.Dim(1))

	hiddenState = hiddenState.Reshape(ctx, hiddenState.Dim(0), 1, hiddenState.Dim(1))

	var gateOut, upOut ml.Tensor
	if moe.GateUp != nil && moe.GateUp.Weight != nil {
		gateUp := moe.GateUp.Forward(ctx, hiddenState, selectedExperts)
		nFF := gateUp.Dim(0) / 2
		gateOut = gateUp.Slice(ctx, 0, 0, nFF, 1)
		upOut = gateUp.Slice(ctx, 0, nFF, gateUp.Dim(0), 1)
	} else {
		gateOut = moe.Gate.Forward(ctx, hiddenState, selectedExperts)
		upOut = moe.Up.Forward(ctx, hiddenState, selectedExperts)
	}
	hiddenState = gateOut.GELU(ctx, upOut)
	experts := moe.Down.Forward(ctx, hiddenState, selectedExperts)

	if moe.DownScale != nil {
		expertScales := moe.DownScale.Reshape(ctx, opts.numExperts, 1)
		expertScales = expertScales.Repeat(ctx, 1, hiddenState.Dim(2))
		expertScales = expertScales.Reshape(ctx, 1, opts.numExperts, hiddenState.Dim(2)).Rows(ctx, selectedExperts)
		expertScales = expertScales.Reshape(ctx, opts.numExpertsUsed, hiddenState.Dim(2))
		expertScales = expertScales.Reshape(ctx, 1, opts.numExpertsUsed, hiddenState.Dim(2))
		experts = experts.Mul(ctx, expertScales)
	}

	experts = experts.Mul(ctx, routingWeights)

	nextStates := experts.View(ctx, 0, experts.Dim(0), experts.Stride(2), experts.Dim(2))
	for i := 1; i < opts.numExpertsUsed; i++ {
		nextStates = nextStates.Add(ctx, experts.View(ctx, i*experts.Stride(1), experts.Dim(0), experts.Stride(2), experts.Dim(2)))
	}

	return nextStates
}

// --- Transformer Layer ---

type TextLayer struct {
	AttentionNorm     *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention     *TextSelfAttention
	PostAttentionNorm *nn.RMSNorm `gguf:"post_attention_norm,alt:attn_post_norm"`
	MLPNorm           *nn.RMSNorm `gguf:"ffn_norm,alt:ffn_pre_norm"`
	MLP               *TextMLP
	PostMLPNorm       *nn.RMSNorm `gguf:"post_ffw_norm,alt:ffn_post_norm"`

	// MoE (present only when experts are configured)
	Router       *TextRouter
	MoE          *TextMoEBlock
	MoENorm      *nn.RMSNorm `gguf:"pre_ffw_norm_2,alt:ffn_pre_norm_2"`
	PostMoENorm  *nn.RMSNorm `gguf:"post_ffw_norm_2,alt:ffn_post_norm_2"`
	PostMLPNorm1 *nn.RMSNorm `gguf:"post_ffw_norm_1,alt:ffn_post_norm_1"`
}

func (l *TextLayer) Forward(ctx ml.Context, layer int, hiddenState, positions, outputs ml.Tensor, cache kvcache.Cache, opts *TextOptions) ml.Tensor {
	residual := hiddenState

	hiddenState = l.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.SelfAttention.Forward(ctx, layer, hiddenState, positions, cache, opts)
	hiddenState = l.PostAttentionNorm.Forward(ctx, hiddenState, opts.eps)

	if outputs != nil {
		hiddenState = hiddenState.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	// Check for MoE
	hasSplitExperts := l.MoE != nil && l.MoE.Gate != nil && l.MoE.Up != nil && l.MoE.Gate.Weight != nil && l.MoE.Up.Weight != nil
	hasFusedExperts := l.MoE != nil && l.MoE.GateUp != nil && l.MoE.GateUp.Weight != nil
	if l.Router != nil && l.MoE != nil && l.MoE.Down != nil && l.MoE.Down.Weight != nil && (hasSplitExperts || hasFusedExperts) {
		mlpState := l.MLPNorm.Forward(ctx, hiddenState, opts.eps)
		mlpState = l.MLP.Forward(ctx, mlpState)
		mlpState = l.PostMLPNorm1.Forward(ctx, mlpState, opts.eps)

		routingWeights, selectedExperts := l.Router.Forward(ctx, hiddenState, opts)
		moeState := l.MoENorm.Forward(ctx, hiddenState, opts.eps)
		moeState = l.MoE.Forward(ctx, moeState, routingWeights, selectedExperts, opts)
		moeState = l.PostMoENorm.Forward(ctx, moeState, opts.eps)

		combined := mlpState.Add(ctx, moeState)
		combined = l.PostMLPNorm.Forward(ctx, combined, opts.eps)
		hiddenState = combined.Add(ctx, residual)
	} else {
		hiddenState = l.MLPNorm.Forward(ctx, hiddenState, opts.eps)
		hiddenState = l.MLP.Forward(ctx, hiddenState)
		hiddenState = l.PostMLPNorm.Forward(ctx, hiddenState, opts.eps)
		hiddenState = hiddenState.Add(ctx, residual)
	}

	return hiddenState
}
