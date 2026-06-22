package mlxrunner

// depthProbeInterval is the base cadence at which the controller drafts one past its
// selection to refresh the next position up; without it a shallow controller never
// notices a deeper draft becoming worthwhile.
const depthProbeInterval = 32

// depthProbeIntervalMax caps the probe backoff. Probes that keep changing nothing
// double the interval up to this cap.
const depthProbeIntervalMax = 512

// depthController drafts argmax_N EV(N) where EV(N) = committed(N) / cost(N), over
// depths from 0 (plain decode) up to one past the frontier. The depth-0 floor lets
// it stop speculating when no draft pays; the frontier ceiling keeps it from scoring
// a depth on the optimistic inherited rate, so it climbs outward one position at a
// time as acceptance is measured rather than leaping deep.
//
// It holds the depth-selection state learned across requests — the target forward's
// per-depth cost curve, the drafts' per-position acceptance rates, and the probe
// cadence — persisted on the speculation so a fresh request starts at the proven-out
// depth instead of re-ramping from shallow.
//
// Adapted from origin/jessegross/mtp; not yet wired into the live MTP/DFlash loops
// (Phase 0 ships the controller + tests; Phase 1 replaces ad-hoc draftLimit schedules).
type depthController struct {
	cost    *costModel
	acc     *acceptanceModel
	pending bool // the previous round drafted a probe

	// Probe cadence, persisted so a backed-off request need not restart at the base.
	probeInterval int // rounds between probes; backs off while probes change nothing
	probeSince    int // rounds since the cadence was last calibrated
	lastSelected  int // the selection the cadence was calibrated against
}

func newDepthController() *depthController {
	return &depthController{cost: newCostModel(), acc: newAcceptanceModel(), probeInterval: depthProbeInterval}
}

func (c *depthController) acceptance(i int) float64 { return c.acc.acceptance(i) }
func (c *depthController) frontier() int            { return c.acc.frontier() }

// committed returns expected committed tokens at depth N: the current token, which
// always commits, plus the expected number of accepted drafts — each draft position
// contributes the probability its whole prefix was accepted, the running product of
// the per-position acceptance rates summed over positions.
func (c *depthController) committed(n int) float64 {
	total, prod := 1.0, 1.0
	for k := 1; k <= n; k++ {
		prod *= c.acceptance(k)
		total += prod
	}
	return total
}

// next returns the draft depth for the upcoming step: the EV-optimal depth (capped
// at frontier+1), except periodically it probes one past the selection to refresh
// the next position up. The probe stays within the frontier window. The cadence
// doubles toward its cap while probes change nothing and resets on any selection
// change, giving the new selection a full interval to settle.
func (c *depthController) next() int {
	sel := c.selected()
	if sel != c.lastSelected {
		c.probeInterval = depthProbeInterval
		c.probeSince = 0
		c.lastSelected = sel
	} else if c.pending {
		c.probeInterval = min(c.probeInterval*2, depthProbeIntervalMax)
	}
	c.pending = false

	c.probeSince++
	if c.probeSince >= c.probeInterval {
		c.probeSince = 0
		probe := min(sel+1, c.frontier()+1)
		if probe > sel {
			c.pending = true
			return probe
		}
	}
	return sel
}

// selected returns argmax_N EV(N) for N in [0, frontier+1].
func (c *depthController) selected() int {
	bestN, bestEV := 0, 1.0/c.cost.cost(0)
	maxN := c.frontier() + 1
	for n := 1; n <= maxN; n++ {
		ev := c.committed(n) / c.cost.cost(n)
		if ev > bestEV {
			bestEV = ev
			bestN = n
		}
	}
	return bestN
}

// observe records one completed speculation round: draftCount tokens proposed,
// accepted of them taken, and wall duration of the whole target+draft+verify cycle.
func (c *depthController) observe(draftCount, accepted int, dur float64) {
	c.acc.observe(draftCount, accepted)
	c.cost.observe(draftCount, dur)
}

// acceptanceModel tracks per-position acceptance rates (1-indexed draft positions).
// Unseen positions inherit the lowest measured rate so the controller climbs
// conservatively rather than assuming deeper drafts always accept.
type acceptanceModel struct {
	rates []float64 // index 0 unused; rates[i] for draft position i
	hits  []int
	tries []int
}

func newAcceptanceModel() *acceptanceModel {
	return &acceptanceModel{}
}

func (m *acceptanceModel) frontier() int {
	return max(0, len(m.rates)-1)
}

func (m *acceptanceModel) acceptance(i int) float64 {
	if i <= 0 {
		return 1
	}
	if i < len(m.rates) && m.tries[i] > 0 {
		return m.rates[i]
	}
	// Inherit lowest measured rate, or optimistic 0.5 when nothing measured yet.
	lowest := 0.5
	for j := 1; j < len(m.rates); j++ {
		if m.tries[j] > 0 && m.rates[j] < lowest {
			lowest = m.rates[j]
		}
	}
	return lowest
}

func (m *acceptanceModel) observe(draftCount, accepted int) {
	if draftCount <= 0 {
		return
	}
	m.ensure(draftCount)
	for i := 1; i <= draftCount; i++ {
		m.tries[i]++
		if i <= accepted {
			m.hits[i]++
		}
		m.rates[i] = float64(m.hits[i]) / float64(m.tries[i])
	}
}

func (m *acceptanceModel) ensure(n int) {
	for len(m.rates) <= n {
		m.rates = append(m.rates, 0)
		m.hits = append(m.hits, 0)
		m.tries = append(m.tries, 0)
	}
}

// costModel tracks wall cost per draft depth. Depth 0 (plain single-token decode)
// is always available; unseen depths inherit from the nearest measured depth.
type costModel struct {
	sum   []float64
	count []int
}

func newCostModel() *costModel {
	return &costModel{}
}

func (m *costModel) cost(n int) float64 {
	m.ensure(n)
	if m.count[n] > 0 {
		return m.sum[n] / float64(m.count[n])
	}
	// Inherit nearest measured cost; default 1.0 when nothing measured.
	for d := n - 1; d >= 0; d-- {
		if m.count[d] > 0 {
			return m.sum[d] / float64(m.count[d])
		}
	}
	for d := n + 1; d < len(m.count); d++ {
		if m.count[d] > 0 {
			return m.sum[d] / float64(m.count[d])
		}
	}
	return 1.0
}

func (m *costModel) observe(n int, dur float64) {
	if dur <= 0 {
		return
	}
	m.ensure(n)
	m.sum[n] += dur
	m.count[n]++
}

func (m *costModel) ensure(n int) {
	for len(m.sum) <= n {
		m.sum = append(m.sum, 0)
		m.count = append(m.count, 0)
	}
}
