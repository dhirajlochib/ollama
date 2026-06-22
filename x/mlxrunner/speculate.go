package mlxrunner

import (
	"os"
	"strconv"
	"strings"

	"github.com/ollama/ollama/x/mlxrunner/model/base"
	sampler "github.com/ollama/ollama/x/mlxrunner/sample"
)

// SpecMode is the active speculative strategy for a request.
type SpecMode int

const (
	SpecModeNone SpecMode = iota
	SpecModeMTPGreedy
	SpecModeMTPSample
	SpecModeDFlashGreedy
	SpecModeDFlashSample
)

func (m SpecMode) String() string {
	switch m {
	case SpecModeMTPGreedy:
		return "mtp_greedy"
	case SpecModeMTPSample:
		return "mtp_sample"
	case SpecModeDFlashGreedy:
		return "dflash_greedy"
	case SpecModeDFlashSample:
		return "dflash_sample"
	default:
		return "none"
	}
}

func (m SpecMode) Enabled() bool { return m != SpecModeNone }

// dflashDisabled reports whether the operator force-disabled DFlash via env.
func dflashDisabled() bool {
	v := strings.TrimSpace(os.Getenv("OLLAMA_MLX_DFLASH"))
	if v == "" {
		return false
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		return false
	}
	return !b
}

// selectSpecMode chooses the speculative strategy for a request.
// Priority: DFlash (when draft+target capable) > MTP > none.
// Logprobs always disable speculation (matches existing MTP gates).
func (r *Runner) selectSpecMode(opts sampler.Options) (SpecMode, string) {
	if opts.Logprobs || opts.TopLogprobs > 0 {
		return SpecModeNone, "logprobs_requested"
	}

	kind := base.DetectSpecKind(r.Model, r.Draft)
	switch kind {
	case base.SpecKindDFlash:
		if dflashDisabled() {
			return SpecModeNone, "dflash_env_disabled"
		}
		if opts.Temperature > 0 || dflashUsesSamplerHistory(opts) {
			return SpecModeDFlashSample, ""
		}
		return SpecModeDFlashGreedy, ""
	case base.SpecKindMTP:
		if r.useGreedyMTP(opts) {
			return SpecModeMTPGreedy, ""
		}
		if r.useSampleMTP(opts) {
			return SpecModeMTPSample, ""
		}
		return SpecModeNone, "mtp_sampler_incompatible"
	default:
		return SpecModeNone, "no_compatible_draft"
	}
}

func dflashUsesSamplerHistory(opts sampler.Options) bool {
	if opts.RepeatLastN == 0 {
		return false
	}
	repeatPenalty := opts.RepeatPenalty
	if repeatPenalty <= 0 {
		repeatPenalty = 1
	}
	return repeatPenalty != 1 || opts.PresencePenalty != 0 || opts.FrequencyPenalty != 0
}

// dflashDraftLimit returns how many tokens (beyond the current/anchor token)
// the drafter should propose this round. Block size includes the anchor, so
// the draft count is block_size-1, optionally capped by remaining predict budget.
func dflashDraftLimit(draft base.DFlashDraftModel, remaining int) int {
	if draft == nil || remaining <= 0 {
		return 0
	}
	return min(draft.BlockSize()-1, remaining)
}
