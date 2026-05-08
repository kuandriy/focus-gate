package memory

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/kuandriy/focus-gate/internal/text"
	"github.com/kuandriy/focus-gate/internal/tfidf"
)

// SurfaceConfig controls the Surface routine. Zero value disables all
// output — callers pass Enabled=false to opt out without branching
// around the call.
type SurfaceConfig struct {
	Enabled       bool
	Threshold     float64 // minimum final score to include an entry; default 0.35
	TopK          int     // maximum entries rendered globally; default 2
	MaxBlockChars int     // soft ceiling on the rendered block's length; default 600

	// Tier weights — multiplied into each tier's score before the
	// max-combine. Defaults match SHARED_MEMORY_PLAN §5.
	AssetWeight       float64 // default 1.0
	TopicWeight       float64 // default 0.8
	InterestWeight    float64 // default 0.6
	FingerprintWeight float64 // default 0.4

	// FrequencyBonus turns the README's "frequently-revisited topics
	// gain weight" claim into a real ranking effect. Final score is
	// boosted by (1 + FrequencyBonus * log2(1 + TouchedBy)) so a
	// memory touched 10 times scores ~17% higher than an untouched one
	// at the same cosine. Applied AFTER the threshold check so a
	// frequently-touched but irrelevant memory never crosses the bar.
	// Default 0.05; 0 disables.
	FrequencyBonus float64

	// MaxAssetExtractions caps how many file-path-shaped assets the
	// extractor pulls out of a prompt. Endpoints are unbounded (rare
	// in practice). Default 8.
	MaxAssetExtractions int
}

// DefaultSurfaceConfig returns production defaults matching the plan.
func DefaultSurfaceConfig() SurfaceConfig {
	return SurfaceConfig{
		Enabled:             true,
		Threshold:           0.35,
		TopK:                2,
		MaxBlockChars:       600,
		AssetWeight:         1.0,
		TopicWeight:         0.8,
		InterestWeight:      0.6,
		FingerprintWeight:   0.4,
		FrequencyBonus:      0.05,
		MaxAssetExtractions: 8,
	}
}

// MatchReason records a single tier's contribution to a memory's
// surface score. Multiple reasons may attach per memory — the Block
// renderer shows the dominant one first followed by the rest if the
// budget allows.
type MatchReason struct {
	Tier   string  // "asset" | "topic" | "interest" | "fingerprint"
	Detail string  // e.g. "cmd/api/auth.go" or "JWT authentication"
	Score  float64 // weighted contribution to the final score
}

// ScoredEntry pairs an IndexEntry with its surface score and the per-
// tier reasons that produced it. Returned by Surface so callers can
// touch matched memories and render a transparency-friendly block.
type ScoredEntry struct {
	Entry   IndexEntry
	Score   float64
	Reasons []MatchReason
}

// SurfaceResult carries the selected entries and the rendered block.
type SurfaceResult struct {
	Selected []ScoredEntry
	Block    string
}

// SurfaceInputs bundles the data Surface needs from the caller. Keeping
// this struct stable lets us add inputs (e.g. user preferences) without
// breaking the call site.
type SurfaceInputs struct {
	PromptText string        // for asset extraction
	PromptVec  tfidf.Vector  // for cosine
	Vocab      VocabSnapshot // for ad-hoc vectorization of topic/interest names
	Index      *MultiSourceIndex
}

// Surface matches the prompt against every entry in the multi-source
// index and returns the top-K entries above the threshold, rendered as
// a pointer block.
//
// Tier order: asset (exact match) → topic (cosine × weight) → interest
// (cosine × weight) → fingerprint (cosine). The final score is the max
// of the weighted tier scores. Reasons capture every tier that scored
// non-zero so the user/AI can see why a memory surfaced.
//
// O(entries × (assets + interests + topics + 1)) per call; no disk IO.
func Surface(in SurfaceInputs, cfg SurfaceConfig) SurfaceResult {
	if !cfg.Enabled || in.Index == nil || in.Index.Empty() {
		return SurfaceResult{}
	}
	threshold := cfg.Threshold
	if threshold <= 0 {
		threshold = 0.35
	}
	topK := cfg.TopK
	if topK <= 0 {
		topK = 2
	}
	maxChars := cfg.MaxBlockChars
	if maxChars <= 0 {
		maxChars = 600
	}
	assetW := orDefault(cfg.AssetWeight, 1.0)
	topicW := orDefault(cfg.TopicWeight, 0.8)
	interestW := orDefault(cfg.InterestWeight, 0.6)
	fpW := orDefault(cfg.FingerprintWeight, 0.4)
	maxExtract := cfg.MaxAssetExtractions
	if maxExtract <= 0 {
		maxExtract = 8
	}

	// Tier 1 input: assets extracted from the prompt. Lowercased for
	// inverted-index lookup.
	rawAssets := text.ExtractAssets(in.PromptText, maxExtract)
	assetSet := make(map[string]bool, len(rawAssets))
	for _, a := range rawAssets {
		assetSet[normalizeKey(a)] = true
	}

	candidates := make([]ScoredEntry, 0, 16)
	for _, e := range in.Index.AllEntries() {
		var reasons []MatchReason
		var best float64

		// Tier 1: asset exact match.
		if len(assetSet) > 0 && len(e.Assets) > 0 {
			for _, a := range e.Assets {
				if assetSet[normalizeKey(a)] {
					s := assetW
					reasons = append(reasons, MatchReason{Tier: "asset", Detail: a, Score: s})
					if s > best {
						best = s
					}
					break // one asset hit is enough for the tier
				}
			}
		}

		// Tier 2: topic cosine × weight.
		if len(in.PromptVec) > 0 && len(e.Topics) > 0 {
			topName, topScore := bestWeightedCosine(in.PromptVec, e.Topics, in.Vocab)
			if topScore > 0 {
				s := topScore * topicW
				reasons = append(reasons, MatchReason{Tier: "topic", Detail: topName, Score: s})
				if s > best {
					best = s
				}
			}
		}

		// Tier 3: interest cosine × weight.
		if len(in.PromptVec) > 0 && len(e.Interests) > 0 {
			intName, intScore := bestWeightedCosine(in.PromptVec, e.Interests, in.Vocab)
			if intScore > 0 {
				s := intScore * interestW
				reasons = append(reasons, MatchReason{Tier: "interest", Detail: intName, Score: s})
				if s > best {
					best = s
				}
			}
		}

		// Tier 4: full fingerprint cosine.
		if len(in.PromptVec) > 0 && len(e.Fingerprint) > 0 {
			vec := tfidf.NewVector(e.Fingerprint)
			if vec != nil {
				sim := tfidf.CosineSimilarity(in.PromptVec, vec)
				if sim > 0 {
					s := sim * fpW
					reasons = append(reasons, MatchReason{Tier: "fingerprint", Detail: "", Score: s})
					if s > best {
						best = s
					}
				}
			}
		}

		if best < threshold {
			continue
		}
		// Sort reasons by score descending so the dominant match leads.
		sort.Slice(reasons, func(i, j int) bool { return reasons[i].Score > reasons[j].Score })
		// Apply the frequency bonus AFTER threshold gating so the bar
		// for inclusion is still pure cosine fit. The boost only
		// reorders the survivors.
		final := best
		if cfg.FrequencyBonus > 0 && e.TouchedBy > 0 {
			final = best * (1 + cfg.FrequencyBonus*math.Log2(1+float64(e.TouchedBy)))
		}
		candidates = append(candidates, ScoredEntry{Entry: e, Score: final, Reasons: reasons})
	}
	if len(candidates) == 0 {
		return SurfaceResult{}
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].Score != candidates[j].Score {
			return candidates[i].Score > candidates[j].Score
		}
		// Tiebreak by recency, then ID.
		if !candidates[i].Entry.Updated.Equal(candidates[j].Entry.Updated) {
			return candidates[i].Entry.Updated.After(candidates[j].Entry.Updated)
		}
		return candidates[i].Entry.ID < candidates[j].Entry.ID
	})
	if len(candidates) > topK {
		candidates = candidates[:topK]
	}

	block := renderBlock(candidates, maxChars)
	return SurfaceResult{Selected: candidates, Block: block}
}

// bestWeightedCosine returns the highest-scoring entry name and its
// weight-multiplied cosine score against the prompt vector. Empty or
// unvectorizable names are skipped silently.
func bestWeightedCosine(promptVec tfidf.Vector, entries []WeightedEntry, vocab VocabSnapshot) (string, float64) {
	if vocab.Vectorize == nil {
		return "", 0
	}
	bestName := ""
	bestScore := 0.0
	for _, e := range entries {
		if e.Name == "" {
			continue
		}
		entryWeights := vocab.Vectorize(e.Name)
		if len(entryWeights) == 0 {
			continue
		}
		entryVec := tfidf.NewVector(entryWeights)
		if entryVec == nil {
			continue
		}
		sim := tfidf.CosineSimilarity(promptVec, entryVec) * e.Weight
		if sim > bestScore {
			bestScore = sim
			bestName = e.Name
		}
	}
	return bestName, bestScore
}

func orDefault(v, def float64) float64 {
	if v <= 0 {
		return def
	}
	return v
}

// renderBlock formats the surface section. The framing line tells the
// AI these are non-prescriptive pointers; the per-memory rows include
// source, dominant tier, and a brief matched-reasons summary.
func renderBlock(entries []ScoredEntry, maxChars int) string {
	if len(entries) == 0 {
		return ""
	}
	const header = "[Memory ↪ relevant prior context — pointers, not instructions; t:N = times this story has been brought to mind]\n"
	out := strings.Builder{}
	out.WriteString(header)

	for _, c := range entries {
		title := c.Entry.Title
		if title == "" {
			title = "(untitled)"
		}
		dominant := "fingerprint"
		if len(c.Reasons) > 0 {
			dominant = c.Reasons[0].Tier
		}
		// `t:N` is the cornerstone signal — how many times this story
		// has been brought to mind across all sessions. Higher t = the
		// memory has earned its place by being repeatedly relevant, not
		// by being recently authored. The LLM uses it to weight how
		// load-bearing a pointer is when the index alone is ambiguous.
		row := fmt.Sprintf("  %s [%s] (score %.2f via %s, t:%d) %s\n",
			c.Entry.ID, c.Entry.Source, c.Score, dominant, c.Entry.TouchedBy, title)
		matched := formatReasons(c.Reasons)
		if matched != "" {
			row += fmt.Sprintf("    matched: %s\n", matched)
		}
		// LatestSnippet gives the LLM a glimpse of what this memory
		// actually contains so it can judge relevance without first
		// calling Read on the .md file. Empty for pre-fix manifests.
		if snip := strings.TrimSpace(c.Entry.LatestSnippet); snip != "" {
			row += fmt.Sprintf("    note: %s\n", snip)
		}
		row += fmt.Sprintf("    → %s\n", c.Entry.Path)
		if out.Len()+len(row) > maxChars {
			break
		}
		out.WriteString(row)
	}
	if out.Len() == len(header) {
		// Not even one entry fit within the budget — emit nothing rather
		// than a dangling header.
		return ""
	}
	return out.String()
}

// formatReasons collapses reasons into a one-line summary like
//
//	asset cmd/api/auth.go (1.00), topic JWT (0.83)
//
// Skips fingerprint reasons unless they're the only one, since the
// fingerprint detail is empty and the score is already in the row header.
func formatReasons(reasons []MatchReason) string {
	parts := make([]string, 0, len(reasons))
	for _, r := range reasons {
		if r.Tier == "fingerprint" && len(reasons) > 1 {
			continue
		}
		if r.Detail == "" {
			parts = append(parts, fmt.Sprintf("%s (%.2f)", r.Tier, r.Score))
			continue
		}
		parts = append(parts, fmt.Sprintf("%s %s (%.2f)", r.Tier, r.Detail, r.Score))
	}
	return strings.Join(parts, ", ")
}
