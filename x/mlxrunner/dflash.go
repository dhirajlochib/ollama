package mlxrunner

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	sampler "github.com/ollama/ollama/x/mlxrunner/sample"
)

type dflashStats struct {
	iterations       int
	drafted          int
	accepted         int
	mismatches       int
	allAccepted      int
	batched          int
	serial           int
	targetDuration   time.Duration
	draftDuration    time.Duration
	validateDuration time.Duration
}

func freeCacheSet(caches []cache.Cache) {
	for _, c := range caches {
		if c != nil {
			c.Free()
		}
	}
}

// runGreedyDFlashDecode implements block-diffusion speculative decoding for
// greedy (temperature=0, neutral penalties) requests. Adapted from the reverted
// #16134 runner but using scheduleSpeculation/commitSpeculation (current main
// snapshot API) instead of cache.BeginSpeculation.
func (r *Runner) runGreedyDFlashDecode(ctx context.Context, request Request, session *cacheSession, targetCaches []cache.Cache, draftCaches []cache.Cache, seed []int32, position *int, started time.Time) error {
	target := r.Model.(base.DFlashTargetModel)
	draft := r.Draft.(base.DFlashDraftModel)
	stats := dflashStats{}
	slog.Info("DFlash greedy decode enabled", "block_size", draft.BlockSize(), "target_layers", draft.TargetLayerIDs())

	targetForward := func(token *mlx.Array) (*mlx.Array, *mlx.Array) {
		hidden, targetHidden := target.ForwardDFlash(&batch.Batch{
			InputIDs:     token,
			SeqOffsets:   []int32{int32(*position)},
			SeqQueryLens: []int32{int32(token.Dim(1))},
		}, targetCaches, draft.TargetLayerIDs())
		*position += token.Dim(1)
		return hidden, targetHidden
	}

	t0 := time.Now()
	hidden, targetHidden := targetForward(mlx.FromValues(seed, 1, len(seed)))
	draft.AppendContext(targetHidden, draftCaches)
	current := sampler.Result{Token: greedyTokenFromLogits(r.lastLogits(hidden))}
	mlx.Pin(current.Arrays()...)
	mlx.Sweep()
	mlx.AsyncEval(current.Arrays()...)
	stats.targetDuration += time.Since(t0)
	defer func() {
		mlx.Unpin(current.Arrays()...)
	}()

	dec := decoder{tokenizer: r.Tokenizer}
	final := CompletionResponse{Done: true, PromptEvalCount: len(request.Tokens), DoneReason: 1}
	now := started
	generated := 0

	for generated < request.Options.NumPredict {
		if err := ctx.Err(); err != nil {
			return err
		}

		if generated == 0 {
			mlx.Eval(current.Arrays()...)
			final.PromptEvalDuration = time.Since(now)
			now = time.Now()
		}

		done, err := r.emitTokens(ctx, request, session, &dec, []sampler.Result{current}, &final, &generated)
		if err != nil {
			return err
		}
		if done || generated >= request.Options.NumPredict {
			break
		}

		draftCount := dflashDraftLimit(draft, request.Options.NumPredict-generated)
		if draftCount <= 0 {
			t0 = time.Now()
			hidden, targetHidden := targetForward(mtpTokenInput(current.Token))
			draft.AppendContext(targetHidden, draftCaches)
			stats.targetDuration += time.Since(t0)
			next := sampler.Result{Token: greedyTokenFromLogits(r.lastLogits(hidden))}
			mlx.Pin(next.Arrays()...)
			old := current
			current = next
			mlx.Unpin(old.Arrays()...)
			mlx.Sweep()
			mlx.AsyncEval(current.Arrays()...)
			continue
		}

		stats.iterations++
		t0 = time.Now()
		draftTokens := r.generateDFlashDrafts(draft, current.Token, draftCaches, draftCount)
		mlx.Pin(draftTokens)
		mlx.Eval(draftTokens)
		stats.draftDuration += time.Since(t0)
		stats.drafted += draftCount

		t0 = time.Now()
		next, accepted, done, err := r.acceptDFlashDrafts(ctx, request, session, &dec, target, draft, targetCaches, draftCaches, position, current, draftTokens, &final, &generated, &stats)
		stats.validateDuration += time.Since(t0)
		mlx.Unpin(draftTokens)
		if err != nil {
			return err
		}
		stats.accepted += accepted
		if accepted == draftCount {
			stats.allAccepted++
		} else {
			stats.mismatches++
		}
		if done || generated >= request.Options.NumPredict {
			break
		}

		mlx.Pin(next.Arrays()...)
		old := current
		current = next
		mlx.Unpin(old.Arrays()...)
		mlx.Sweep()
		mlx.AsyncEval(current.Arrays()...)

		if generated%256 == 0 {
			mlx.ClearCache()
		}
	}

	final.EvalCount = generated
	final.EvalDuration = time.Since(now)
	acceptance := 0.0
	if stats.drafted > 0 {
		acceptance = float64(stats.accepted) / float64(stats.drafted)
	}
	slog.Info("DFlash decode stats",
		"spec_kind", "dflash",
		"mode", "greedy",
		"generated", generated,
		"drafted", stats.drafted,
		"accepted", stats.accepted,
		"acceptance", acceptance,
		"iterations", stats.iterations,
		"batched", stats.batched,
		"serial", stats.serial,
		"mismatches", stats.mismatches,
		"all_accepted", stats.allAccepted,
		"target_duration", stats.targetDuration,
		"draft_duration", stats.draftDuration,
		"validate_duration", stats.validateDuration,
	)
	select {
	case <-ctx.Done():
		return ctx.Err()
	case request.Responses <- final:
		return nil
	}
}

func (r *Runner) runSampleDFlashDecode(ctx context.Context, request Request, session *cacheSession, targetCaches []cache.Cache, draftCaches []cache.Cache, seed []int32, position *int, started time.Time) error {
	target := r.Model.(base.DFlashTargetModel)
	draft := r.Draft.(base.DFlashDraftModel)
	stats := dflashStats{}
	slog.Info("DFlash sample decode enabled", "block_size", draft.BlockSize(), "target_layers", draft.TargetLayerIDs())

	targetForward := func(token *mlx.Array) (*mlx.Array, *mlx.Array) {
		hidden, targetHidden := target.ForwardDFlash(&batch.Batch{
			InputIDs:     token,
			SeqOffsets:   []int32{int32(*position)},
			SeqQueryLens: []int32{int32(token.Dim(1))},
		}, targetCaches, draft.TargetLayerIDs())
		*position += token.Dim(1)
		return hidden, targetHidden
	}

	t0 := time.Now()
	hidden, targetHidden := targetForward(mlx.FromValues(seed, 1, len(seed)))
	draft.AppendContext(targetHidden, draftCaches)
	current := r.Sampler.Sample([]int{pipelineSlot}, r.lastLogits(hidden))
	mlx.Pin(current.Arrays()...)
	mlx.Sweep()
	mlx.AsyncEval(current.Arrays()...)
	stats.targetDuration += time.Since(t0)
	defer func() {
		mlx.Unpin(current.Arrays()...)
	}()

	dec := decoder{tokenizer: r.Tokenizer}
	final := CompletionResponse{Done: true, PromptEvalCount: len(request.Tokens), DoneReason: 1}
	now := started
	generated := 0

	for generated < request.Options.NumPredict {
		if err := ctx.Err(); err != nil {
			return err
		}

		if generated == 0 {
			mlx.Eval(current.Arrays()...)
			final.PromptEvalDuration = time.Since(now)
			now = time.Now()
		}

		done, err := r.emitTokens(ctx, request, session, &dec, []sampler.Result{current}, &final, &generated)
		if err != nil {
			return err
		}
		if done || generated >= request.Options.NumPredict {
			break
		}

		draftCount := dflashDraftLimit(draft, request.Options.NumPredict-generated)
		if draftCount <= 0 {
			t0 = time.Now()
			hidden, targetHidden := targetForward(mtpTokenInput(current.Token))
			draft.AppendContext(targetHidden, draftCaches)
			stats.targetDuration += time.Since(t0)
			next := r.Sampler.Sample([]int{pipelineSlot}, r.lastLogits(hidden))
			mlx.Pin(next.Arrays()...)
			old := current
			current = next
			mlx.Unpin(old.Arrays()...)
			mlx.Sweep()
			mlx.AsyncEval(current.Arrays()...)
			continue
		}

		stats.iterations++
		t0 = time.Now()
		candidates := r.generateDFlashDraftCandidates(draft, current.Token, draftCaches, draftCount)
		mlx.Pin(candidates.Arrays()...)
		mlx.Eval(candidates.Arrays()...)
		stats.draftDuration += time.Since(t0)
		stats.drafted += draftCount

		t0 = time.Now()
		next, accepted, done, err := r.acceptSampleDFlashDrafts(ctx, request, session, &dec, target, draft, targetCaches, draftCaches, position, current, candidates, &final, &generated, &stats)
		stats.validateDuration += time.Since(t0)
		mlx.Unpin(candidates.Arrays()...)
		if err != nil {
			return err
		}
		stats.accepted += accepted
		if accepted == draftCount {
			stats.allAccepted++
		} else {
			stats.mismatches++
		}
		if done || generated >= request.Options.NumPredict {
			break
		}

		mlx.Pin(next.Arrays()...)
		old := current
		current = next
		mlx.Unpin(old.Arrays()...)
		mlx.Sweep()
		mlx.AsyncEval(current.Arrays()...)

		if generated%256 == 0 {
			mlx.ClearCache()
		}
	}

	final.EvalCount = generated
	final.EvalDuration = time.Since(now)
	acceptance := 0.0
	if stats.drafted > 0 {
		acceptance = float64(stats.accepted) / float64(stats.drafted)
	}
	slog.Info("DFlash decode stats",
		"spec_kind", "dflash",
		"mode", "sample",
		"generated", generated,
		"drafted", stats.drafted,
		"accepted", stats.accepted,
		"acceptance", acceptance,
		"iterations", stats.iterations,
		"batched", stats.batched,
		"serial", stats.serial,
		"mismatches", stats.mismatches,
		"all_accepted", stats.allAccepted,
		"target_duration", stats.targetDuration,
		"draft_duration", stats.draftDuration,
		"validate_duration", stats.validateDuration,
	)
	select {
	case <-ctx.Done():
		return ctx.Err()
	case request.Responses <- final:
		return nil
	}
}

func (r *Runner) dflashDraftLogits(draft base.DFlashDraftModel, current *mlx.Array, caches []cache.Cache, draftCount int) *mlx.Array {
	blockLen := draftCount + 1
	values := make([]int32, blockLen)
	values[0] = int32(tokenID(current))
	for i := 1; i < blockLen; i++ {
		values[i] = draft.MaskTokenID()
	}
	block := mlx.FromValues(values, 1, blockLen)
	logits := draft.Draft(block, caches)
	return logits.Slice(mlx.Slice(), mlx.Slice(1, blockLen), mlx.Slice())
}

func (r *Runner) generateDFlashDrafts(draft base.DFlashDraftModel, current *mlx.Array, caches []cache.Cache, draftCount int) *mlx.Array {
	logits := r.dflashDraftLogits(draft, current, caches, draftCount)
	return logits.Argmax(-1, false).AsType(mlx.DTypeInt32)
}

type dflashDraftCandidates struct {
	tokens *mlx.Array
	dist   sampler.Distribution
}

func (c *dflashDraftCandidates) Arrays() []*mlx.Array {
	if c == nil {
		return nil
	}
	return append([]*mlx.Array{c.tokens}, c.dist.Arrays()...)
}

func (r *Runner) generateDFlashDraftCandidates(draft base.DFlashDraftModel, current *mlx.Array, caches []cache.Cache, draftCount int) *dflashDraftCandidates {
	if draftCount <= 0 {
		return nil
	}

	logits := r.dflashDraftLogits(draft, current, caches, draftCount)
	draftTokens := make([]*mlx.Array, 0, draftCount)
	draftDists := make([]sampler.Distribution, 0, draftCount)
	var prefix *mlx.Array

	for i := range draftCount {
		rows := logits.Slice(mlx.Slice(), mlx.Slice(0, i+1), mlx.Slice())
		dist := r.Sampler.Distribution(pipelineSlot, rows, prefix).SliceRows(i, i+1)
		nextToken := mtpTokenVector(r.Sampler.SampleDistribution(pipelineSlot, dist))
		nextInput := mtpTokenInput(nextToken)

		draftTokens = append(draftTokens, nextInput)
		draftDists = append(draftDists, dist)
		if prefix == nil {
			prefix = nextInput
		} else {
			prefix = prefix.Concatenate(1, nextInput)
		}
	}
	if len(draftTokens) == 0 {
		return nil
	}
	return &dflashDraftCandidates{
		tokens: mlx.Concatenate(draftTokens, 1),
		dist:   sampler.ConcatenateDistributions(draftDists),
	}
}

func (r *Runner) acceptDFlashDrafts(ctx context.Context, request Request, session *cacheSession, dec *decoder, target base.DFlashTargetModel, draft base.DFlashDraftModel, targetCaches []cache.Cache, draftCaches []cache.Cache, position *int, current sampler.Result, draftTokens *mlx.Array, final *CompletionResponse, generated *int, stats *dflashStats) (sampler.Result, int, bool, error) {
	stats.batched++
	return r.acceptDFlashDraftsBatched(ctx, request, session, dec, target, draft, targetCaches, draftCaches, position, current, draftTokens, final, generated)
}

func (r *Runner) acceptDFlashDraftsBatched(ctx context.Context, request Request, session *cacheSession, dec *decoder, target base.DFlashTargetModel, draft base.DFlashDraftModel, targetCaches []cache.Cache, draftCaches []cache.Cache, position *int, current sampler.Result, draftTokens *mlx.Array, final *CompletionResponse, generated *int) (sampler.Result, int, bool, error) {
	before := *position
	draftCount := draftTokens.Dim(1)
	verifyInput := mtpTokenInput(current.Token).Concatenate(1, draftTokens)

	scheduleSpeculation(targetCaches, before, verifyInput.Dim(1))
	hiddenSeq, targetHiddenSeq := target.ForwardDFlash(&batch.Batch{
		InputIDs:     verifyInput,
		SeqOffsets:   []int32{int32(before)},
		SeqQueryLens: []int32{int32(verifyInput.Dim(1))},
	}, targetCaches, draft.TargetLayerIDs())

	selectedTokens := r.Model.Unembed(hiddenSeq).Argmax(-1, false).AsType(mlx.DTypeInt32)
	mlx.Eval(draftTokens, selectedTokens)

	draftIDs := draftTokens.Ints()
	selectedIDs := selectedTokens.Ints()
	if len(selectedIDs) < draftCount+1 {
		commitSpeculation(targetCaches, 0, verifyInput.Dim(1), before)
		return sampler.Result{}, 0, false, fmt.Errorf("dflash validation produced %d tokens for %d draft tokens", len(selectedIDs), draftCount)
	}

	accepted := 0
	var next sampler.Result
	done := false
	for i, id := range draftIDs {
		if selectedIDs[i] != id {
			next = sampler.Result{Token: mtpTokenAt(selectedTokens, i)}
			break
		}
		accepted++
		if r.Tokenizer.IsEOS(int32(id)) {
			done = true
			break
		}
	}

	// Commit target caches through the accepted draft tokens only (bonus at
	// accepted+1 is not written as a live cache advance — position tracks accepted).
	// verifyInput length is draftCount+1; we accept `accepted` draft tokens which
	// advances position by accepted (current was already emitted separately).
	commitSpeculation(targetCaches, accepted, verifyInput.Dim(1), before)
	*position = before + accepted

	// Inject target mid-layer features for the verified span into draft caches.
	if targetHiddenSeq != nil && accepted > 0 {
		// Rows 0..accepted are [current, draft_0, ..., draft_{accepted-1}];
		// append all accepted context (skip none — current was not appended at emit).
		span := targetHiddenSeq.Slice(mlx.Slice(), mlx.Slice(0, accepted+1), mlx.Slice())
		draft.AppendContext(span, draftCaches)
	} else if targetHiddenSeq != nil && accepted == 0 {
		// Still need context for the rejected current position's target view.
		span := targetHiddenSeq.Slice(mlx.Slice(), mlx.Slice(0, 1), mlx.Slice())
		draft.AppendContext(span, draftCaches)
	}

	emitted, err := r.emitTokens(ctx, request, session, dec, draftResults(draftIDs[:accepted]), final, generated)
	if err != nil {
		return sampler.Result{}, accepted, emitted || done, err
	}
	if emitted || done {
		return sampler.Result{}, accepted, true, nil
	}
	if next.Token == nil {
		next = sampler.Result{Token: mtpTokenAt(selectedTokens, draftCount)}
	}
	return next, accepted, false, nil
}

func (r *Runner) acceptSampleDFlashDrafts(ctx context.Context, request Request, session *cacheSession, dec *decoder, target base.DFlashTargetModel, draft base.DFlashDraftModel, targetCaches []cache.Cache, draftCaches []cache.Cache, position *int, current sampler.Result, candidates *dflashDraftCandidates, final *CompletionResponse, generated *int, stats *dflashStats) (sampler.Result, int, bool, error) {
	stats.batched++

	before := *position
	draftCount := candidates.tokens.Dim(1)
	verifyInput := mtpTokenInput(current.Token).Concatenate(1, candidates.tokens)

	scheduleSpeculation(targetCaches, before, verifyInput.Dim(1))
	hiddenSeq, targetHiddenSeq := target.ForwardDFlash(&batch.Batch{
		InputIDs:     verifyInput,
		SeqOffsets:   []int32{int32(before)},
		SeqQueryLens: []int32{int32(verifyInput.Dim(1))},
	}, targetCaches, draft.TargetLayerIDs())

	// Target distributions for [current, draft...]; slice off current for comparison to draft dist.
	targetLogits := r.Model.Unembed(hiddenSeq)
	targetDist := r.Sampler.Distribution(pipelineSlot, targetLogits, verifyInput)
	draftDist := candidates.dist
	// Compare draft positions only (skip row 0 = current/anchor).
	acceptedMask := r.mtpSampleAcceptedMask(targetDist.SliceRows(1, draftCount+1), draftDist, candidates.tokens)
	mlx.Eval(candidates.tokens, acceptedMask)

	draftIDs := candidates.tokens.Ints()
	acceptedFlags := acceptedMask.Ints()
	accepted := 0
	for _, ok := range acceptedFlags {
		if ok == 0 {
			break
		}
		accepted++
	}
	if accepted > draftCount {
		commitSpeculation(targetCaches, 0, verifyInput.Dim(1), before)
		return sampler.Result{}, 0, false, fmt.Errorf("dflash sample validation accepted %d tokens for %d draft tokens", accepted, draftCount)
	}

	commitIDs := make([]int32, 0, accepted+1)
	done := false
	for i, id := range draftIDs[:accepted] {
		commitIDs = append(commitIDs, int32(id))
		if r.Tokenizer.IsEOS(int32(id)) {
			done = true
			accepted = i + 1
			commitIDs = commitIDs[:accepted]
			break
		}
	}

	commitSpeculation(targetCaches, accepted, verifyInput.Dim(1), before)
	*position = before + accepted

	if targetHiddenSeq != nil {
		spanEnd := accepted + 1
		if spanEnd > targetHiddenSeq.Dim(1) {
			spanEnd = targetHiddenSeq.Dim(1)
		}
		if spanEnd > 0 {
			span := targetHiddenSeq.Slice(mlx.Slice(), mlx.Slice(0, spanEnd), mlx.Slice())
			draft.AppendContext(span, draftCaches)
		}
	}

	emitted, err := r.emitTokens(ctx, request, session, dec, draftResults(draftIDs[:accepted]), final, generated)
	if err != nil {
		return sampler.Result{}, accepted, emitted || done, err
	}
	if emitted || done {
		r.Sampler.Commit(pipelineSlot, commitIDs)
		return sampler.Result{}, accepted, true, nil
	}

	var nextToken *mlx.Array
	if accepted == draftCount {
		nextToken = r.mtpSampleTokenAt(targetDist, draftCount+1) // bonus after full accept (row accepted+1? row draftCount+0 is last draft; bonus is at draftCount+1)
	} else {
		// Residual at first rejected draft position (row accepted+1 in verify input)
		nextToken = r.mtpSampleResidualToken(targetDist.SliceRows(1, draftCount+1), draftDist, accepted)
	}
	mlx.Eval(nextToken)
	nextID := int32(tokenID(nextToken))
	r.Sampler.Commit(pipelineSlot, append(commitIDs, nextID))
	return sampler.Result{Token: nextToken}, accepted, false, nil
}
