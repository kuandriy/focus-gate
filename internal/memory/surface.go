package memory

import (
	"fmt"
	"sort"
	"strings"

	"github.com/kuandriy/focus-gate/internal/tfidf"
)

// SurfaceConfig controls the Surface routine. Zero value disables all
// output — callers pass Enabled=false to opt out without branching
// around the call.
type SurfaceConfig struct {
	Enabled       bool
	Threshold     float64 // minimum cosine to include an entry; default 0.35
	TopK          int     // maximum entries rendered; default 2
	MaxBlockChars int     // soft ceiling on the rendered block's length; default 250
}

// DefaultSurfaceConfig returns production defaults matching the plan
// (§10). Callers usually override one or two fields; this keeps the
// config struct small at call sites.
func DefaultSurfaceConfig() SurfaceConfig {
	return SurfaceConfig{
		Enabled:       true,
		Threshold:     0.35,
		TopK:          2,
		MaxBlockChars: 250,
	}
}

// SurfaceResult carries the selected entries and the rendered block.
// Callers inspect Selected to touch each entry's counter, then append
// Block to the context. Separating selection from rendering keeps the
// hook's touch-bookkeeping contract explicit.
type SurfaceResult struct {
	Selected []IndexEntry // entries whose touchedBy counter should bump
	Block    string       // pre-rendered text ready for the context stream
}

// Surface matches the prompt vector against every manifest entry and
// returns the top-K above the threshold, rendered as a pointer block.
//
// This is the hot path run on every prompt, so all work is bounded by
// len(manifest.Entries) × len(promptVector). No disk IO.
//
// Returns an empty SurfaceResult when the feature is disabled, the
// manifest is empty, or no entry clears the threshold — the caller can
// just check len(result.Selected) to decide whether there's anything to
// emit.
func Surface(promptVec tfidf.Vector, manifest *Manifest, cfg SurfaceConfig) SurfaceResult {
	if !cfg.Enabled || manifest == nil || len(manifest.Entries) == 0 || len(promptVec) == 0 {
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
		maxChars = 250
	}

	candidates := make([]scoredEntry, 0, len(manifest.Entries))
	for _, e := range manifest.Entries {
		vec := tfidf.NewVector(e.Fingerprint)
		if vec == nil {
			continue
		}
		sim := tfidf.CosineSimilarity(promptVec, vec)
		if sim >= threshold {
			candidates = append(candidates, scoredEntry{sim, e})
		}
	}
	if len(candidates) == 0 {
		return SurfaceResult{}
	}
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].sim > candidates[j].sim
	})
	if len(candidates) > topK {
		candidates = candidates[:topK]
	}

	selected := make([]IndexEntry, len(candidates))
	for i, c := range candidates {
		selected[i] = c.entry
	}

	block := renderBlock(candidates, maxChars)
	return SurfaceResult{Selected: selected, Block: block}
}

// scoredEntry pairs a cosine score with an IndexEntry for internal
// bookkeeping between Surface and renderBlock.
type scoredEntry struct {
	sim   float64
	entry IndexEntry
}

// renderBlock formats the surface section using the same visual language
// as the forest summary so the AI picks up the convention by pattern.
// Respects the MaxBlockChars budget by dropping entries past the limit
// — we never truncate the header because that would produce malformed
// output.
func renderBlock(entries []scoredEntry, maxChars int) string {
	if len(entries) == 0 {
		return ""
	}
	const header = "[Memory ↪ relevant prior context]\n"
	out := strings.Builder{}
	out.WriteString(header)

	for _, c := range entries {
		title := c.entry.Title
		if title == "" {
			title = "(untitled)"
		}
		line := fmt.Sprintf("  %s [sim %.2f] %s\n    → %s\n",
			c.entry.ID, c.sim, title, c.entry.Path)
		if out.Len()+len(line) > maxChars {
			break
		}
		out.WriteString(line)
	}
	if out.Len() == len(header) {
		// Not even one entry fit within the budget — emit nothing rather
		// than a dangling header. The forest summary already owns the
		// context block's visual weight.
		return ""
	}
	return out.String()
}
