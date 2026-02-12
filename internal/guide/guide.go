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
	Entries []Entry `json:"entries"`
	MaxSize int     `json:"maxSize"`
}

// New creates a guide with the given capacity.
func New(maxSize int) *Guide {
	return &Guide{
		MaxSize: maxSize,
	}
}

// Add appends a response summary. If capacity is exceeded, the oldest entry is dropped.
func (g *Guide) Add(summary string, intentID string, refs []string) {
	if summary == "" {
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
