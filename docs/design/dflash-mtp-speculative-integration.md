# DFlash + MTP Speculative Decoding Integration Plan

Status: **Phase 1 landed on this branch** (interfaces, DFlash draft model,
Qwen3.5 target capture, runner gates, depth controller). Full end-to-end
DFlash decode loops and the jessegross/mtp `speculate` engine refactor are
intentionally staged.

## 1. Landscape (what exists in Ollama today)

### 1.1 MLX runner — Gemma4 MTP (live on `main`)

| Area | Path | Role |
|------|------|------|
| Runner loops | `x/mlxrunner/mtp.go` | Greedy + sample MTP decode, draft generation, batched/serial validation |
| Cache rollback | `scheduleSpeculation` / `commitSpeculation` in `mtp.go` | Per-token snapshots + restore for target KV during verify |
| Pipeline gate | `x/mlxrunner/pipeline.go` | `useGreedyMTP` / `useSampleMTP` after prefill |
| Interfaces | `x/mlxrunner/model/base/base.go` | `MTPDraftModel`, `MTPEmbeddingModel`, `MTPDefaultsProvider` |
| Draft weights | `x/models/gemma4/assistant.go` | Gemma4 assistant MTP head |
| Sampler support | `x/mlxrunner/sample/sample.go` | `SpeculativeScores`, draft history without commit |

MTP draft shape (Gemma4 assistant):

1. Target emits last-token hidden state.
2. Draft head fuses `TokenEmbeddings(t) || h_target` and predicts one next token autoregressively, repeating up to N times with the same RoPE/cache position (single-position assistant training).
3. Target verifies the proposed sequence in one batched forward; accept longest matching prefix; emit bonus token from target.

### 1.2 llama-server path — GGUF MTP (live on `main`)

| Area | Path | Role |
|------|------|------|
| Launch args | `llm/llama_server.go` `appendMTPDraftArgs` | `--spec-type draft-mtp`, `--spec-draft-n-max`, optional `--spec-draft-model` |
| Detection | `hasMTPDraft` / `hasLegacyQwenMTPDraft` | `nextn_predict_layers` or legacy `mtp.*` tensors on qwen35/qwen35moe |
| Convert | `convert/convert_qwen3next.go` | Embedded MTP -> nextn tensors; separate MTP draft safetensors |
| Compat | `llama/compat/llama-ollama-compat.cpp` | Rename/skip/duplicate MTP for llama.cpp |

This path is not DFlash; it delegates speculative decode to upstream llama-server.

### 1.3 Manifest / create plumbing (live on `main`)

| Area | Path | Role |
|------|------|------|
| Config | `types/model/config.go` `Draft` | Architecture, tensor prefix, draft config path |
| Create | `x/create/client/create.go` | Reads draft `config.json` architectures (tests already cover `DFlashDraftModel`) |
| Modelfile | `DRAFT` command in parser / `cmd/cmd.go` | Experimental local draft dir for safetensors create |

### 1.4 DFlash — landed then reverted on `main`

| Commit | Summary |
|--------|---------|
| `98e26b8c` | `mlxrunner: add DFlash speculative decoding (#16134)` — full runner + `x/models/dflash` + Qwen3.5 `ForwardDFlash` |
| `358af4af` | Revert — too invasive: pipeline/cache/recurrent/qwen3.5-specific threading |

Reverted design highlights:

- **Draft**: lightweight Qwen3-style transformer (~5-8 layers), bidirectional attention inside the draft block, target hidden states injected as extra KV at every draft layer (`AppendContext`).
- **Block proposal**: one forward over `[anchor_token, MASK, MASK, ...]` of length `block_size`; logits for mask positions become the draft sequence (greedy or Leviathan/Chen sampling).
- **Verify**: target `ForwardDFlash` runs verify tokens and returns concatenated mid-layer activations for the next draft context append.
- **Caches**: draft owns a separate cache set (not target KV); target uses existing speculation snapshots.

### 1.5 Upstream `origin/jessegross/mtp` — unified `speculate` engine (not merged)

Refactors MTP into:

- `drafter` interface (`propose` / `committed` / `finish` / `flush`)
- `speculation` + `speculationSession` on the Runner
- `depthController` — EV-optimal draft depth `argmax_N committed(N)/cost(N)` with probe cadence
- Generalized `DraftModel` (Draft/Unembed/DraftCaches) so Gemma4 assistant and future heads share one loop

This is the right long-term shape for integrating DFlash without a second full decode loop.

### 1.6 DiffusionGemma (`feat/diffusiongemma-architecture`)

Separate branch: block-diffusion as the primary generation model (EBD unmasking loop), not a speculative drafter. Conceptually adjacent to DFlash's block/mask idea but different product path (standalone diffusion LM vs target+draft speculation).

## 2. Research summary

### 2.1 Classic MTP / EAGLE / Medusa

| Method | Drafter | Parallelism | Notes |
|--------|---------|-------------|-------|
| Speculative decoding (Leviathan/Chen) | Small AR LM | Draft AR, verify parallel | Lossless w.r.t. target distribution |
| MTP heads (Gloeckle et al., DeepSeek/Qwen nextn) | Extra heads/layers on target | Usually AR over heads | Often trained jointly; Ollama maps to `draft-mtp` |
| Medusa | Multi heads on final hidden | Tree/parallel heads | Needs tree attention / verification |
| EAGLE / EAGLE-3 | Autoregressive feature-level draft | Draft AR on features | Strong acceptance; still sequential draft cost |
| **DFlash** (Chen et al., arXiv:2602.06036) | **Block diffusion** draft LM | **One** draft forward per block | Bidirectional in-block attention; target-layer KV injection |

### 2.2 DFlash (why it matters for Ollama)

Paper/project: https://arxiv.org/abs/2602.06036 , https://github.com/z-lab/dflash

Key inference loop:

```
for each generation step:
  1. Target forward on accepted/bonus token(s) -> logits + mid-layer hiddens H_L
  2. Draft.AppendContext(H_L)   // inject into draft KV (not target KV)
  3. Build block = [last_token, MASK x (gamma-1)]
  4. Draft.Draft(block) -> logits for all mask positions in one pass
  5. Sample/greedy draft tokens t1..t_{gamma-1}
  6. Target verify [last, t1..t_{gamma-1}] in one batched forward (+ capture H_L for accepted+bonus)
  7. Accept longest prefix; emit bonus; AppendContext for accepted span
```

Differences vs Gemma4 MTP in Ollama:

| | Gemma4 MTP | DFlash |
|--|------------|--------|
| Draft forwards per block | O(gamma) autoregressive | O(1) block diffusion |
| Attention in draft | Causal / single-pos assistant | Bidirectional within block |
| Target coupling | Fuse last embed+hidden into draft input | Inject target hiddens as draft KV |
| Draft caches | Often shares/reuses target history | Own caches; context via AppendContext |
| Typical gamma | 4-16 adaptive | Fixed ~16 (block_size) |
| Target archs (today) | Gemma4 | Qwen3.x (+ future Gemma4 if we add ForwardDFlash) |

## 3. Integration strategy (decision)

Do not fully revive the reverted PR as-is. The revert rationale still stands: threading DFlash-only logic through pipeline + recurrent cache + model forwards was too invasive.

Preferred path: evolve toward a unified speculation engine, then plug MTP and DFlash as drafter backends.

### Phase 0 — Foundation (this branch)

1. Document architecture + plan.
2. Extend `base` with `DFlashTargetModel`, `DFlashDraftModel`, `SpecKind` / `DetectSpecKind`.
3. Land `x/models/dflash` draft package (self-contained; registers `DFlashDraftModel` / `dflash`).
4. Minimal Qwen3.5 target support: `ForwardDFlash` + `TokenEmbeddings`.
5. Runner-side gates + depth controller + DFlash decode loops adapted to current snapshot API (`scheduleSpeculation`/`commitSpeculation`).
6. Pipeline priority: DFlash > MTP > plain when capabilities match.
7. Unit tests for kind detection, config parse, depth EV.

### Phase 1 — Drafter abstraction (follow-up PR)

Port/adapt `origin/jessegross/mtp` `speculate.go` patterns without requiring every draft to implement `DraftCaches` on the shared `DraftModel` interface yet:

1. Introduce internal `drafter` interface in `x/mlxrunner`.
2. Wrap existing MTP loops as `mtpDrafter` (behavior-preserving).
3. Wrap DFlash as `dflashDrafter` (propose = block draft; committed = AppendContext on accepted target hiddens).
4. Move depth selection out of ad-hoc draftLimit into `depthController`.

### Phase 2 — Target coverage

| Target | Work |
|--------|------|
| Qwen3.5 dense | Done (capture path) |
| Qwen3.5 MoE / Laguna | Same forward capture pattern |
| Gemma4 | Optional: add ForwardDFlash only if/when a Gemma4 DFlash draft ships |
| GGUF / llama-server | Out of scope for DFlash; keep draft-mtp for Qwen MTP GGUF |

### Phase 3 — Pipeline / cache hygiene

1. No qwen-specific code in `cache/recurrent` — recurrent playback stays behind target/draft methods.
2. Draft caches owned by the drafter session, not spliced into the target trie except optional snapshot alignment.
3. Prefill: for DFlash, capture mid-layers and AppendContext so draft context matches target at decode start.
4. Keep YaRN/RoPE helpers in `x/models/nn` (already on main; dflash model reuses them).

### Phase 4 — Product / UX

1. `ollama create` with `DRAFT ./path/to/dflash` already reads architecture.
2. Optional env: `OLLAMA_MLX_DFLASH=0` force-disable.
3. API: reuse DraftNumPredict; for DFlash default depth = min(DraftNumPredict, block_size-1).
4. Metrics logs: tag `spec_kind=dflash|mtp`.

### Phase 5 — Validation matrix

1. Unit: config, kind detection, depth EV, accept/reject logic.
2. MLX integration: Qwen3.5 + z-lab DFlash draft; greedy lossless check.
3. Regression: Gemma4 MTP acceptance rates unchanged.
4. Negative: logprobs/top_logprobs disable speculation.

## 4. Fresh start vs extend stub — recommendation

| Option | Pros | Cons |
|--------|------|------|
| A. Fresh unified engine | Clean; one place for depth/stats/rollback | Large PR; high merge risk |
| B. Extend current MTP stubs + additive DFlash (chosen for Phase 0) | Ships incrementally; respects revert lesson | Temporary duplication of decode loops |
| C. Full revive of `98e26b8c` | Fastest if it applied cleanly | Re-introduces invasive cache/pipeline coupling |

Recommendation: B now, migrate to A in Phase 1-2. Do not revive the full reverted PR; cherry-pick only the dflash model + target capture + runner loops, and fold into the jessegross-style engine next.

## 5. Implementation checklist (tracking)

### Phase 0 (this branch)

- [x] Design doc
- [x] `base.DFlashTargetModel` / `DFlashDraftModel` / `DetectSpecKind`
- [x] `x/models/dflash` model + config test
- [x] Qwen3.5 `ForwardDFlash` / `TokenEmbeddings`
- [x] `x/mlxrunner/speculate_depth.go` depth controller
- [x] `x/mlxrunner/speculate.go` kind/gates/helpers
- [x] `x/mlxrunner/dflash.go` greedy/sample decode using existing snapshot helpers
- [x] Pipeline hooks (DFlash before MTP)
- [x] Import `x/models/dflash`
- [x] Tests: kind detection, depth EV

### Phase 1+

- [ ] `drafter` interface + `mtpDrafter` / `dflashDrafter`
- [ ] Prefill capture + AppendContext alignment for DFlash sessions
- [ ] Qwen3.5 MoE / Laguna `ForwardDFlash`
- [ ] Depth controller wired into MTP
- [ ] Create/import docs for DFlash draft models
- [ ] Integration benchmarks vs MTP / plain

## 6. Risks

1. Correctness: sample-mode must use proper speculative rejection sampling.
2. Recurrent / hybrid targets: Qwen3.5 linear attention complicates snapshot/rollback.
3. Acceptance vs speed: depth controller should fall back to 0 when verify dominates.
4. Merge conflicts: coordinate with `jessegross/mtp` by keeping Phase 0 additive.

## 7. References

- DFlash paper: https://arxiv.org/abs/2602.06036
- DFlash code/models: https://github.com/z-lab/dflash
- Ollama commits: `15e6076d` (Gemma4 MTP), `98e26b8c` / `358af4af` (DFlash add/revert), `d0062206` (MTP cache snapshots)
- Branch inspiration: `origin/jessegross/mtp` (`speculate.go`, `speculate_depth.go`)
