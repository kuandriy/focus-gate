package memory

import (
	"fmt"
	"sort"
	"strings"

	"github.com/kuandriy/focus-gate/internal/tfidf"
)

// PromoteOptions controls the Stage B prompt builder. Zero values are
// fine — every field has a sensible fallback.
type PromoteOptions struct {
	// MaxNodeSnippetLen truncates each NodeContents entry shown in the
	// prompt so a runaway 8 KB transcript paste doesn't blow up the
	// prompt. Default 240 chars.
	MaxNodeSnippetLen int

	// MaxIndexEntries caps how many existing memories are listed in
	// the "EXISTING MEMORIES" section. The LLM picks among them for
	// append; if the catalog is large, we show the cosine-closest
	// memories to the candidate's fingerprint first so the merge-or-
	// create decision is fed the relevant signal, not the whole
	// catalog. Default 30.
	MaxIndexEntries int

	// DefaultSource hint shown to the LLM so it knows where targetSource
	// defaults if omitted. Phase 5 will pass this from config; before
	// then, callers pass DefaultSourceName.
	DefaultSource string
}

// BuildStageBPrompt formats a Candidate + aggregate index into the
// instruction block sent to the LLM via `fg: memory promote <tempId>`.
//
// The prompt includes:
//   - Candidate snapshot (reason, abstraction, refs, terms, score signals)
//   - Per-tree node contents (truncated)
//   - Existing memories: id, title, source, timeMarkers, interests/topics/
//     assets — NO BODIES (per plan §6: "the index, not the bodies")
//   - The decision shape and JSON schema for `fg: memory commit`
//
// Output is a single string ready to write to stdout.
func BuildStageBPrompt(c *Candidate, msi *MultiSourceIndex, opts PromoteOptions) string {
	if c == nil {
		return ""
	}
	maxSnippet := opts.MaxNodeSnippetLen
	if maxSnippet <= 0 {
		maxSnippet = 240
	}
	maxEntries := opts.MaxIndexEntries
	if maxEntries <= 0 {
		maxEntries = 30
	}
	defaultSource := opts.DefaultSource
	if defaultSource == "" {
		defaultSource = DefaultSourceName
	}

	var b strings.Builder
	fmt.Fprintf(&b, "=== Stage B: Memory Promotion (candidate %s) ===\n\n", c.TempID)
	// One framing line, no repetition. The original draft said "you are
	// the student" three times across the preamble; LLMs hit the schema
	// more reliably when the role + invariants are stated once and the
	// concrete examples below carry the rest of the signal.
	b.WriteString("Decide whether this candidate warrants a memory chapter.\n")
	b.WriteString("Stories describe what was true and why — never what someone should do.\n")
	b.WriteString("Old chapters never edit; corrections come as new chapters.\n")
	b.WriteString("When narrating `what`, capture both what stuck AND what was tried-and-abandoned.\n")
	b.WriteString("Past-tense fact includes dead ends — they are how the kept design earned its place.\n")
	b.WriteString("The EXISTING MEMORIES list below is the full aggregate index across every enabled source.\n\n")

	b.WriteString("CANDIDATE SNAPSHOT\n")
	fmt.Fprintf(&b, "  reason:        %s\n", c.Reason)
	fmt.Fprintf(&b, "  source tree:   %s\n", c.SourceTreeID)
	fmt.Fprintf(&b, "  abstraction:   %s\n", oneLine(c.RootAbstraction, 200))
	fmt.Fprintf(&b, "  prompt count:  %d\n", c.PromptCount)
	fmt.Fprintf(&b, "  age (hours):   %.1f\n", c.AgeHours)
	fmt.Fprintf(&b, "  total weight:  %.2f\n", c.TotalNodeWeight)
	fmt.Fprintf(&b, "  peak score:    %.2f\n", c.PeakScore)
	if len(c.TopTerms) > 0 {
		fmt.Fprintf(&b, "  top terms:     %s\n", strings.Join(c.TopTerms, ", "))
	}
	if len(c.Refs) > 0 {
		fmt.Fprintf(&b, "  refs:          %s\n", strings.Join(c.Refs, ", "))
	}
	if len(c.NodeContents) > 0 {
		b.WriteString("  node contents:\n")
		for _, nc := range c.NodeContents {
			fmt.Fprintf(&b, "    - %s\n", oneLine(nc, maxSnippet))
		}
	}
	if len(c.GuideSummaries) > 0 {
		b.WriteString("  AI summaries (assistant reinforcement):\n")
		for _, g := range c.GuideSummaries {
			fmt.Fprintf(&b, "    - %s\n", oneLine(g, maxSnippet))
		}
	}
	b.WriteString("\n")

	// Existing memories — index data only, no bodies. Rank by cosine
	// to the candidate's fingerprint so the LLM's append/create
	// decision sees the most plausible merge targets first instead of
	// "the most-recently-updated 30 entries", which is unrelated to
	// the candidate's topic.
	entries := rankEntriesByRelevance(msi.AllEntries(), c.Fingerprint)
	if len(entries) > maxEntries {
		entries = entries[:maxEntries]
	}
	b.WriteString("EXISTING MEMORIES (titles + indexes only — bodies stay on disk)\n")
	if len(entries) == 0 {
		b.WriteString("  (none)\n\n")
	} else {
		for _, e := range entries {
			fmt.Fprintf(&b, "  %s [%s] %q\n", e.ID, e.Source, e.Title)
			if len(e.TimeMarkers) > 0 {
				fmt.Fprintf(&b, "    timeMarkers: %s\n", strings.Join(e.TimeMarkers, ", "))
			}
			if len(e.Interests) > 0 {
				fmt.Fprintf(&b, "    interests:  %s\n", formatWeightedList(e.Interests))
			}
			if len(e.Topics) > 0 {
				fmt.Fprintf(&b, "    topics:     %s\n", formatWeightedList(e.Topics))
			}
			if len(e.Assets) > 0 {
				fmt.Fprintf(&b, "    assets:     %s\n", strings.Join(e.Assets, ", "))
			}
		}
		b.WriteString("\n")
	}

	if c.SuggestedAction != "" {
		fmt.Fprintf(&b, "HINT (non-binding): selection suggested action=%q", c.SuggestedAction)
		if c.SuggestedTargetID != "" {
			fmt.Fprintf(&b, " targetId=%s", c.SuggestedTargetID)
		}
		if c.SuggestedTargetSource != "" {
			fmt.Fprintf(&b, " targetSource=%s", c.SuggestedTargetSource)
		}
		b.WriteString("\n\n")
	}

	b.WriteString("DECIDE one of:\n")
	b.WriteString("  - append:  the candidate continues an existing story\n")
	b.WriteString("             (set targetId; targetSource defaults to \"" + defaultSource + "\")\n")
	b.WriteString("  - create:  no meaningful overlap — this is a new story\n")
	b.WriteString("  - discard: not memory-worthy (typo storm, single-prompt curiosity)\n\n")

	b.WriteString("REPLY with exactly one line:\n")
	fmt.Fprintf(&b, "  fg: memory commit %s '<json>'\n\n", c.TempID)

	// Few-shot: a worked append, a worked create, and a worked discard.
	// LLMs follow concrete examples more reliably than schema prose, so
	// these sit in front of the formal field reference. Each example is
	// a literal payload — copy-paste-ready.
	b.WriteString("EXAMPLES (literal payloads — match this shape):\n\n")

	b.WriteString("  Example A — append a chapter to an existing story (with abandonment narrative):\n")
	b.WriteString("  {\n")
	b.WriteString("    \"action\": \"append\",\n")
	fmt.Fprintf(&b, "    \"targetId\": \"mem_20260322_a1b2c3\",\n")
	fmt.Fprintf(&b, "    \"targetSource\": %q,\n", defaultSource)
	b.WriteString("    \"chapter\": {\n")
	b.WriteString("      \"title\": \"Refresh token rotation\",\n")
	b.WriteString("      \"timeMarker\": \"2026-04-12\",\n")
	b.WriteString("      \"assets\": [\"middleware/refresh.go\", \"POST /auth/refresh\"],\n")
	b.WriteString("      \"interests\": [{\"name\": \"session lifecycle\", \"weight\": 1.0}],\n")
	b.WriteString("      \"topics\": [{\"name\": \"refresh token rotation\", \"weight\": 0.9}],\n")
	b.WriteString("      \"what\": \"Tried client-side rotation first; abandoned after request-ordering races corrupted the rotation chain. Settled on server-side single-use rotation with a 24-hour refresh window.\",\n")
	b.WriteString("      \"why\": \"Limit blast radius if a refresh token is exfiltrated; server-side state is the only place the chain can be ordered safely.\"\n")
	b.WriteString("    }\n")
	b.WriteString("  }\n\n")

	b.WriteString("  Example B — create a new story from a topic that has no overlap:\n")
	b.WriteString("  {\n")
	b.WriteString("    \"action\": \"create\",\n")
	b.WriteString("    \"newMemory\": {\n")
	b.WriteString("      \"title\": \"Rate limit middleware ordering\",\n")
	b.WriteString("      \"timeMarkers\": [\"2026-04-08..2026-04-10\"],\n")
	b.WriteString("      \"chapter\": {\n")
	b.WriteString("        \"title\": \"Initial design\",\n")
	b.WriteString("        \"timeMarker\": \"2026-04-08..2026-04-10\",\n")
	b.WriteString("        \"assets\": [\"middleware/rate.go\"],\n")
	b.WriteString("        \"topics\": [{\"name\": \"rate limit\", \"weight\": 1.0}],\n")
	b.WriteString("        \"what\": \"Placed rate-limit before auth so unauthenticated floods don't spend auth budget.\",\n")
	b.WriteString("        \"why\": \"Auth is the most expensive step; protect it from cheap probing.\"\n")
	b.WriteString("      }\n")
	b.WriteString("    }\n")
	b.WriteString("  }\n\n")

	b.WriteString("  Example C — discard a one-shot curiosity:\n")
	b.WriteString("  { \"action\": \"discard\" }\n\n")

	b.WriteString("FIELD REFERENCE (only fields not shown above):\n")
	b.WriteString("  - title:      ≤80 runes; required for create, optional for append.\n")
	b.WriteString("  - timeMarker: any short temporal label, ≤60 chars, single line.\n")
	b.WriteString("                Examples: \"2026-04-12\", \"2026-04-10..2026-04-12\", \"sprint-42\", \"v1.2-release\".\n")
	b.WriteString("  - weight:     [0.0, 1.0] — saturated at 1.0 and floored at 0.1 internally.\n")
	b.WriteString("  - what:       past-tense narrative — what was decided AND what was abandoned.\n")
	b.WriteString("  - why:        the rationale behind what stuck (and, when relevant, why dead ends were dropped).\n")
	b.WriteString("  Discard takes no body — \"action\": \"discard\" is the entire payload.\n")
	return b.String()
}

// rankEntriesByRelevance returns entries sorted by cosine similarity
// to the candidate's fingerprint, descending. Entries with no
// fingerprint or zero similarity are kept at the tail in their
// original order so the prompt always shows *something* even when the
// candidate has nothing in common with the catalog. Recency is the
// stable secondary sort.
func rankEntriesByRelevance(entries []IndexEntry, candidateFp map[string]float64) []IndexEntry {
	if len(entries) == 0 {
		return entries
	}
	cv := tfidf.NewVector(candidateFp)
	if cv == nil {
		// No candidate fingerprint — keep input order; the caller's
		// recency-based source ordering is the best we can do.
		return entries
	}
	type ranked struct {
		entry IndexEntry
		score float64
	}
	scored := make([]ranked, len(entries))
	for i, e := range entries {
		ev := tfidf.NewVector(e.Fingerprint)
		if ev == nil {
			scored[i] = ranked{entry: e, score: 0}
			continue
		}
		scored[i] = ranked{entry: e, score: tfidf.CosineSimilarity(cv, ev)}
	}
	sort.SliceStable(scored, func(i, j int) bool {
		if scored[i].score != scored[j].score {
			return scored[i].score > scored[j].score
		}
		return scored[i].entry.Updated.After(scored[j].entry.Updated)
	})
	out := make([]IndexEntry, len(scored))
	for i, r := range scored {
		out[i] = r.entry
	}
	return out
}

func oneLine(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.Join(strings.Fields(s), " ")
	if max > 0 && len([]rune(s)) > max {
		runes := []rune(s)
		s = string(runes[:max]) + "…"
	}
	return s
}

func formatWeightedList(entries []WeightedEntry) string {
	parts := make([]string, len(entries))
	for i, e := range entries {
		parts[i] = fmt.Sprintf("%s@%.2f", e.Name, e.Weight)
	}
	return strings.Join(parts, ", ")
}
