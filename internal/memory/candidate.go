package memory

import (
	"math"
	"regexp"
	"sort"
	"time"

	"github.com/kuandriy/focus-gate/internal/forest"
	"github.com/kuandriy/focus-gate/internal/text"
	"github.com/kuandriy/focus-gate/internal/tfidf"
)

// Candidate is the on-disk form of a tree that has been flagged as worth
// promoting into a long-term memory. Its shape feeds the Stage B prompt
// the LLM receives via `fg: memory promote` (§6 of the plan).
//
// The LLM ultimately decides append / create / discard via a commit
// payload; SuggestedAction is just a hint computed from the merge-match
// search at selection time.
type Candidate struct {
	SchemaVersion         string             `json:"schemaVersion"`
	TempID                string             `json:"tempId"`
	Reason                string             `json:"reason"` // "prune" or "merge"
	SourceTreeID          string             `json:"sourceTreeId"`
	RootAbstraction       string             `json:"rootAbstraction"`
	TopTerms              []string           `json:"topTerms"`
	Refs                  []string           `json:"refs"`
	AgeHours              float64            `json:"ageHours"`
	TotalNodeWeight       float64            `json:"totalNodeWeight"`
	PromptCount           int                `json:"promptCount"`
	PeakScore             float64            `json:"peakScore"`
	NodeContents          []string           `json:"nodeContents"`
	GuideSummaries        []string           `json:"guideSummaries,omitempty"`
	Fingerprint           map[string]float64 `json:"fingerprint"`
	MergeMatches          []MergeMatch       `json:"mergeMatches,omitempty"`
	SuggestedAction       string             `json:"suggestedAction"` // "create" | "append"
	SuggestedTargetID     string             `json:"suggestedTargetId,omitempty"`
	SuggestedTargetSource string             `json:"suggestedTargetSource,omitempty"`
	// CommitRetries tracks how many times the LLM has tried (and failed)
	// to commit this candidate. Incremented by the commit handler on
	// validation errors; once it crosses the configured budget, the
	// candidate is parked rather than discarded — the user can inspect
	// and clear it manually.
	CommitRetries int       `json:"commitRetries,omitempty"`
	CreatedAt     time.Time `json:"createdAt"`
}

// MergeMatch names an existing memory the LLM may want to merge into,
// with its similarity score so merge decisions are transparent.
// Source is populated so the Stage B prompt can group matches by
// origin — a candidate that's similar to memories in two different
// sources is a meaningfully different decision than one that has
// three matches all in the same source.
type MergeMatch struct {
	ID     string  `json:"id"`
	Source string  `json:"source"`
	Title  string  `json:"title"`
	Cosine float64 `json:"cosine"`
}

// DefaultRedactPatterns ships a conservative baseline that catches the
// common credential shapes in transcript paste-ins so they don't leak
// into pending_memories.json even when the user hasn't configured any
// redaction. These are anchored or shape-specific to keep false-
// positive risk low against realistic developer chat.
//
// Callers extend these with their own patterns; both lists feed
// applyRedact verbatim so order doesn't matter.
var DefaultRedactPatterns = []string{
	`AKIA[0-9A-Z]{16}`,                                      // AWS access key id
	`\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+`, // JWT shape
	`-----BEGIN [A-Z ]*PRIVATE KEY-----`,                    // PEM private keys
	`(?i)bearer\s+[A-Za-z0-9._\-+/=]{20,}`,                  // Bearer tokens
	`(?i)(?:api[_-]?key|secret|token|password)["':=\s]+[A-Za-z0-9_\-]{16,}`, // labeled secrets
}

// SelectConfig captures the candidate-selection knobs. Zero-valued fields
// fall back to sensible defaults inside SelectCandidate, so callers can
// wire a partial config at the edges (e.g. from a plan-local config file)
// without populating every field.
type SelectConfig struct {
	MinLeaves          int
	MinPrompts         int
	PromotionThreshold float64
	RescueThreshold    float64
	Cooldown           time.Duration
	MergeSuggestCosine float64
	DedupCosine        float64
	RedactPatterns     []string
	TopTermsCount      int
}

// DefaultSelectConfig mirrors the defaults documented in
// docs/LONG_TERM_MEMORY_PLAN.md §10. Callers tweak fields they care
// about and rely on the rest.
func DefaultSelectConfig() SelectConfig {
	return SelectConfig{
		MinLeaves:          4,
		MinPrompts:         3,
		PromotionThreshold: 1.5,
		RescueThreshold:    1.2,
		Cooldown:           4 * time.Hour,
		MergeSuggestCosine: 0.6,
		DedupCosine:        0.85,
		TopTermsCount:      6,
	}
}

// SelectInputs bundles the state SelectCandidate needs to decide without
// reaching back into Gate. Passing pointers keeps allocations out of the
// hot path: Gate hands over its already-loaded state rather than cloning.
type SelectInputs struct {
	Tree            *forest.Tree
	GuideSummaries  []string             // summaries linked to tree nodes (optional)
	Vocab           VocabSnapshot        // for fingerprint + Vectorize
	DecayRate       float64              // same as Gate.Config.DecayRate; used for scoring
	Reason          string               // "prune" | "merge"
	ExistingEntries []IndexEntry         // manifest entries — used for merge suggestions
	Cooldowns       map[string]time.Time // tree ID → last candidate time
}

// SelectCandidate decides whether a tree qualifies for memory promotion
// and, if so, returns a populated Candidate ready to persist. Returns
// nil when the tree fails any floor check, sits under cooldown, or scores
// below both the promotion threshold and the rescue threshold.
//
// Determinism: all randomness comes from TempID (time-based); everything
// else is a pure function of SelectInputs + SelectConfig.
func SelectCandidate(in SelectInputs, cfg SelectConfig) *Candidate {
	if in.Tree == nil {
		return nil
	}
	if cfg.MinLeaves <= 0 {
		cfg.MinLeaves = 4
	}
	if cfg.MinPrompts <= 0 {
		cfg.MinPrompts = 3
	}
	if cfg.PromotionThreshold <= 0 {
		cfg.PromotionThreshold = 1.5
	}
	if cfg.RescueThreshold <= 0 {
		cfg.RescueThreshold = 1.2
	}
	if cfg.MergeSuggestCosine <= 0 {
		cfg.MergeSuggestCosine = 0.6
	}
	if cfg.TopTermsCount <= 0 {
		cfg.TopTermsCount = 6
	}

	leaves := in.Tree.GetLeaves()
	// Drop root if it appears in the leaf set (single-node tree case).
	leafCount := 0
	for _, l := range leaves {
		if l.ID != in.Tree.RootID {
			leafCount++
		}
	}
	if leafCount < cfg.MinLeaves {
		return nil
	}

	// PromptCount proxy: non-root indexed nodes represent real prompt
	// contributions. Synthetic bubble-up nodes don't count.
	indexedCount := 0
	totalWeight := 0.0
	refSet := map[string]bool{}
	for _, n := range in.Tree.Nodes {
		if n.ID != in.Tree.RootID && n.Indexed {
			indexedCount++
		}
		totalWeight += n.Weight
		for _, r := range n.Refs {
			refSet[r] = true
		}
	}
	if indexedCount < cfg.MinPrompts {
		return nil
	}
	if len(refSet) == 0 && len(in.GuideSummaries) == 0 {
		// Floor requires either real file refs or AI reinforcement.
		return nil
	}

	// Cooldown: a tree that was already promoted recently doesn't get
	// re-queued for promotion until the cool-off window elapses.
	if ts, ok := in.Cooldowns[in.Tree.ID]; ok && cfg.Cooldown > 0 {
		if time.Since(ts) < cfg.Cooldown {
			return nil
		}
	}

	score := candidateScore(in.Tree, totalWeight, len(refSet) > 0, len(in.GuideSummaries) > 0)
	passes := score >= cfg.PromotionThreshold
	rescued := in.Reason == "prune" && in.Tree.PeakScore >= cfg.RescueThreshold
	if !passes && !rescued {
		return nil
	}

	refs := make([]string, 0, len(refSet))
	for r := range refSet {
		refs = append(refs, r)
	}

	contents := collectNodeContents(in.Tree)
	contents = applyRedact(contents, cfg.RedactPatterns)
	guides := applyRedact(in.GuideSummaries, cfg.RedactPatterns)

	// Build a fingerprint from the concatenated tree content so merge
	// lookup uses the same TF-IDF space as Surface.
	joined := in.Tree.Root().Content + " "
	for _, c := range contents {
		joined += c + " "
	}
	for _, g := range guides {
		joined += g + " "
	}
	fp := in.Vocab.Vectorize(joined)

	matches := suggestMergeTargets(fp, in.ExistingEntries, cfg.MergeSuggestCosine)

	action := "create"
	target := ""
	targetSource := ""
	if len(matches) > 0 {
		action = "append"
		target = matches[0].ID
		// At Phase 4 Source on IndexEntry is always populated by the
		// manifest rebuild; older queue entries fall back to the
		// default source name.
		for _, e := range in.ExistingEntries {
			if e.ID == target {
				targetSource = e.Source
				break
			}
		}
		if targetSource == "" {
			targetSource = DefaultSourceName
		}
	}

	now := time.Now().UTC().Truncate(time.Second)
	ageHours := float64(time.Now().UnixMilli()-in.Tree.Created) / 3600000.0
	return &Candidate{
		SchemaVersion:         SchemaVersion,
		TempID:                newTempID(now, in.Tree.ID),
		Reason:                in.Reason,
		SourceTreeID:          in.Tree.ID,
		RootAbstraction:       in.Tree.Root().Content,
		TopTerms:              topTermsFrom(fp, cfg.TopTermsCount),
		Refs:                  refs,
		AgeHours:              ageHours,
		TotalNodeWeight:       totalWeight,
		PromptCount:           indexedCount,
		PeakScore:             in.Tree.PeakScore,
		NodeContents:          contents,
		GuideSummaries:        guides,
		Fingerprint:           fp,
		MergeMatches:          matches,
		SuggestedAction:       action,
		SuggestedTargetID:     target,
		SuggestedTargetSource: targetSource,
		CreatedAt:             now,
	}
}

// candidateScore implements the weighted combo from §5.2 of the plan.
// Kept as an exported-adjacent pure function so tests can exercise it
// without setting up a whole Tree.
func candidateScore(t *forest.Tree, totalWeight float64, hasRefs, hasGuide bool) float64 {
	nodeCount := float64(len(t.Nodes) - 1) // exclude root
	if nodeCount < 1 {
		nodeCount = 1
	}
	ageHours := float64(time.Now().UnixMilli()-t.Created) / 3600000.0
	if ageHours < 0 {
		ageHours = 0
	}

	weightTerm := math.Log2(totalWeight + 1)
	depthTerm := nodeCount / math.Log(ageHours+math.E)

	score := 0.40*weightTerm + 0.30*depthTerm
	if hasRefs {
		score += 0.15
	}
	if hasGuide {
		score += 0.15
	}
	return score
}

// suggestMergeTargets returns the existing memories most similar to the
// candidate's fingerprint, ordered by cosine desc, filtered by
// threshold. Used to pre-populate the "merge candidates" view in the
// pending bundle so the LLM sees its merge options sorted.
func suggestMergeTargets(fp map[string]float64, entries []IndexEntry, threshold float64) []MergeMatch {
	if len(fp) == 0 || len(entries) == 0 {
		return nil
	}
	candVec := tfidf.NewVector(fp)
	var matches []MergeMatch
	for _, e := range entries {
		entryVec := tfidf.NewVector(e.Fingerprint)
		if entryVec == nil {
			continue
		}
		sim := tfidf.CosineSimilarity(candVec, entryVec)
		if sim >= threshold {
			matches = append(matches, MergeMatch{
				ID:     e.ID,
				Source: e.Source,
				Title:  e.Title,
				Cosine: sim,
			})
		}
	}
	// Group by source, then cosine desc within each source. The LLM's
	// merge decision benefits from per-source clusters: "I have three
	// candidates in `personal` and one in `team`" is more actionable
	// than a flat cosine-only sort that interleaves them.
	sort.SliceStable(matches, func(i, j int) bool {
		if matches[i].Source != matches[j].Source {
			return matches[i].Source < matches[j].Source
		}
		return matches[i].Cosine > matches[j].Cosine
	})
	return matches
}

// DedupCandidates collapses near-identical candidates in a single batch
// (e.g. when Prune fires tryMerge which also fires a candidate for the
// same topic). Keeps the first occurrence, drops later duplicates whose
// fingerprint cosine ≥ dedupCosine with any already-kept candidate.
func DedupCandidates(candidates []*Candidate, dedupCosine float64) []*Candidate {
	if dedupCosine <= 0 {
		dedupCosine = 0.85
	}
	var kept []*Candidate
	for _, c := range candidates {
		if c == nil {
			continue
		}
		if isDup(c, kept, dedupCosine) {
			continue
		}
		kept = append(kept, c)
	}
	return kept
}

func isDup(c *Candidate, existing []*Candidate, threshold float64) bool {
	cv := tfidf.NewVector(c.Fingerprint)
	for _, k := range existing {
		if k.SourceTreeID == c.SourceTreeID {
			return true
		}
		kv := tfidf.NewVector(k.Fingerprint)
		if tfidf.CosineSimilarity(cv, kv) >= threshold {
			return true
		}
	}
	return false
}

// collectNodeContents returns the content of every non-root indexed
// leaf. Truncates individual content strings at a generous length so one
// pathological prompt (pasted 8 KB stack trace) doesn't balloon the
// pending file.
func collectNodeContents(t *forest.Tree) []string {
	var out []string
	for _, n := range t.Nodes {
		if n.ID == t.RootID {
			continue
		}
		if !n.Indexed {
			continue
		}
		// Rune-aware truncation so a paste of multi-byte text doesn't
		// produce invalid UTF-8 in the pending bundle.
		out = append(out, text.TruncateRunesWithSuffix(n.Content, 500, "…(truncated)"))
	}
	return out
}

// applyRedact applies every compiled redaction pattern to each string,
// replacing matches with a placeholder so secrets don't enter the
// pending bundle even if they slipped into a prompt.
func applyRedact(strs []string, patterns []string) []string {
	if len(patterns) == 0 || len(strs) == 0 {
		return strs
	}
	compiled := make([]*regexp.Regexp, 0, len(patterns))
	for _, p := range patterns {
		re, err := regexp.Compile(p)
		if err != nil {
			continue
		}
		compiled = append(compiled, re)
	}
	out := make([]string, len(strs))
	for i, s := range strs {
		for _, re := range compiled {
			s = re.ReplaceAllString(s, "«redacted»")
		}
		out[i] = s
	}
	return out
}

// topTermsFrom picks the top-N terms by weight from a fingerprint map.
// Shared with the main memory.RefreshDerived path so the two code paths
// produce identical term lists given the same inputs.
func topTermsFrom(fp map[string]float64, n int) []string {
	terms := make([]string, 0, len(fp))
	for t := range fp {
		terms = append(terms, t)
	}
	// Weight desc, term asc — deterministic tie-break.
	sort.Slice(terms, func(i, j int) bool {
		wi, wj := fp[terms[i]], fp[terms[j]]
		if wi != wj {
			return wi > wj
		}
		return terms[i] < terms[j]
	})
	if n > len(terms) {
		n = len(terms)
	}
	return terms[:n]
}

// newTempID produces a short deterministic-enough ID for a candidate.
// Format: cand_<unix-seconds>_<short-tree-id>. The tree ID tail makes
// later dedup by SourceTreeID cheap.
func newTempID(now time.Time, treeID string) string {
	tail := treeID
	if len(tail) > 8 {
		tail = tail[:8]
	}
	return "cand_" + now.UTC().Format("20060102_150405") + "_" + tail
}
