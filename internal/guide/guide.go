package guide

import (
	"fmt"
	"strings"
	"time"

	"github.com/kuandriy/focus-gate/internal/forest"
)

// Entry represents a single AI response summary linked to an intent node.
type Entry struct {
	Summary   string   `json:"summary"`
	IntentID  string   `json:"intentId"`
	Refs      []string `json:"refs,omitempty"`
	Timestamp int64    `json:"timestamp"`

	// Reinforced is set after this entry has been used by Gate.ReinforceFromGuide
	// to Touch the matching tree root. Prevents double-reinforcement across restarts.
	Reinforced bool `json:"reinforced,omitempty"`
}

// SchemaVersion is the on-disk schema version for guide.json.
const SchemaVersion = "1"

// Guide is a ring buffer of AI response summaries linked to intent nodes.
//
// It serves two roles in the feedback loop:
//  1. Context output — Render() formats recent summaries for the AI's next prompt.
//  2. Bidirectional reinforcement — unreinforced entries are processed by
//     Gate.ReinforceFromGuide(), which matches each summary against the forest
//     and touches the closest tree root. This way both user prompts and AI
//     responses contribute to topic weight, keeping actively-discussed trees
//     alive longer.
//
// Entries are capped at MaxSize. Oldest entries are evicted on overflow.
type Guide struct {
	Schema  string  `json:"schemaVersion"`
	Entries []Entry `json:"entries"`
	MaxSize int     `json:"maxSize"`
}

// SetSchemaVersion implements persist.SchemaVersioner.
func (g *Guide) SetSchemaVersion(v string) { g.Schema = v }

// New creates a guide with the given capacity.
func New(maxSize int) *Guide {
	return &Guide{
		Schema:  SchemaVersion,
		MaxSize: maxSize,
	}
}

// Add appends a response summary. If capacity is exceeded, the oldest entry is dropped.
// Near-duplicate summaries are skipped: Claude Code re-fires the hook on every
// prompt and reads the same "last assistant message" from the transcript, so
// without dedup the same summary would accumulate in consecutive slots. An
// entry is considered a duplicate if its normalized form matches any of the
// last dedupWindow entries.
func (g *Guide) Add(summary string, intentID string, refs []string) {
	if summary == "" {
		return
	}
	if g.isDuplicate(summary) {
		return
	}
	g.Entries = append(g.Entries, Entry{
		Summary:   summary,
		IntentID:  intentID,
		Refs:      refs,
		Timestamp: time.Now().UnixMilli(),
	})
	if len(g.Entries) > g.MaxSize {
		g.Entries = g.Entries[len(g.Entries)-g.MaxSize:]
	}
}

// dedupWindow is how many trailing entries Add checks for duplicates. Larger
// values catch more echoes at the cost of rejecting legitimate repetition
// (e.g. "fix the test" appearing twice across an hours-long session).
const dedupWindow = 3

// isDuplicate returns true if summary's normalized form matches any of the
// most recent dedupWindow entries. Normalization lowercases and collapses
// whitespace so trivial rewordings register as duplicates.
func (g *Guide) isDuplicate(summary string) bool {
	target := normalizeSummary(summary)
	if target == "" {
		return false
	}
	start := len(g.Entries) - dedupWindow
	if start < 0 {
		start = 0
	}
	for i := start; i < len(g.Entries); i++ {
		if normalizeSummary(g.Entries[i].Summary) == target {
			return true
		}
	}
	return false
}

func normalizeSummary(s string) string {
	return strings.Join(strings.Fields(strings.ToLower(s)), " ")
}

// UnreinforcedEntries returns pointers to entries not yet processed for
// forest reinforcement. Gate.ReinforceFromGuide uses this to avoid
// double-touching trees on repeated loads.
func (g *Guide) UnreinforcedEntries() []*Entry {
	var entries []*Entry
	for i := range g.Entries {
		if !g.Entries[i].Reinforced {
			entries = append(entries, &g.Entries[i])
		}
	}
	return entries
}

// Render formats guide entries whose intentID still exists in the forest.
// Dead links (pruned intent nodes) are excluded.
func (g *Guide) Render(f *forest.Forest) string {
	if len(g.Entries) == 0 {
		return ""
	}

	// Build a set of valid intent node IDs
	valid := make(map[string]bool)
	for _, tree := range f.Trees {
		for id := range tree.Nodes {
			valid[id] = true
		}
	}

	var b strings.Builder
	hasContent := false

	for _, e := range g.Entries {
		// Include if intentID is still valid or if intentID is empty (legacy)
		if e.IntentID != "" && !valid[e.IntentID] {
			continue
		}
		if !hasContent {
			b.WriteString("Guide:\n")
			hasContent = true
		}
		fmt.Fprintf(&b, "  - %s\n", e.Summary)
	}

	return b.String()
}
