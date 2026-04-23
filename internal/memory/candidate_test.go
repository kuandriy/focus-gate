package memory

import (
	"strings"
	"testing"
	"time"

	"github.com/kuandriy/focus-gate/internal/forest"
)

// buildTestTree fabricates a Tree that meets the default selection
// floors. Used by several tests as the "happy path" fixture.
func buildTestTree(leafCount int, indexed bool) *forest.Tree {
	t := forest.NewTree("auth jwt session", "")
	root := t.Root()
	root.Indexed = false // roots are never indexed in practice
	for i := 0; i < leafCount; i++ {
		c := t.AddChild(root.ID, "leaf content "+string(rune('a'+i)), "")
		c.Indexed = indexed
		c.Refs = []string{"cmd/api/auth.go"}
		c.Weight = 1.2
	}
	t.PeakScore = 1.5 // above both promotion + rescue thresholds
	return t
}

func TestSelectCandidate_FloorMinLeaves(t *testing.T) {
	tree := buildTestTree(2, true) // below default MinLeaves=4
	got := SelectCandidate(SelectInputs{
		Tree:   tree,
		Vocab:  fakeVocab("v1"),
		Reason: "prune",
	}, DefaultSelectConfig())
	if got != nil {
		t.Errorf("expected nil for tree below minLeaves, got %+v", got)
	}
}

func TestSelectCandidate_FloorMinPrompts(t *testing.T) {
	tree := buildTestTree(5, false) // enough leaves, but none indexed
	got := SelectCandidate(SelectInputs{
		Tree:   tree,
		Vocab:  fakeVocab("v1"),
		Reason: "prune",
	}, DefaultSelectConfig())
	if got != nil {
		t.Errorf("expected nil when no indexed nodes, got %+v", got)
	}
}

func TestSelectCandidate_FloorRequiresRefsOrGuide(t *testing.T) {
	tree := buildTestTree(5, true)
	// Strip refs.
	for _, n := range tree.Nodes {
		n.Refs = nil
	}
	got := SelectCandidate(SelectInputs{
		Tree:   tree,
		Vocab:  fakeVocab("v1"),
		Reason: "prune",
	}, DefaultSelectConfig())
	if got != nil {
		t.Errorf("expected nil when tree has no refs AND no guide summaries")
	}

	// Add a guide entry; expect a candidate now.
	got = SelectCandidate(SelectInputs{
		Tree:           tree,
		GuideSummaries: []string{"added JWT auth"},
		Vocab:          fakeVocab("v1"),
		Reason:         "prune",
	}, DefaultSelectConfig())
	if got == nil {
		t.Error("expected candidate when guide provides reinforcement")
	}
}

func TestSelectCandidate_CooldownSuppresses(t *testing.T) {
	tree := buildTestTree(5, true)
	cooldowns := map[string]time.Time{
		tree.ID: time.Now().Add(-5 * time.Minute),
	}
	cfg := DefaultSelectConfig()
	cfg.Cooldown = 1 * time.Hour

	got := SelectCandidate(SelectInputs{
		Tree:      tree,
		Vocab:     fakeVocab("v1"),
		Reason:    "prune",
		Cooldowns: cooldowns,
	}, cfg)
	if got != nil {
		t.Error("expected cooldown to suppress candidate")
	}

	// Make the last promotion older than the cooldown window.
	cooldowns[tree.ID] = time.Now().Add(-90 * time.Minute)
	got = SelectCandidate(SelectInputs{
		Tree:      tree,
		Vocab:     fakeVocab("v1"),
		Reason:    "prune",
		Cooldowns: cooldowns,
	}, cfg)
	if got == nil {
		t.Error("cooldown expired — expected candidate")
	}
}

func TestSelectCandidate_RescuePathAllowsBelowScoreThreshold(t *testing.T) {
	tree := buildTestTree(4, true)
	tree.PeakScore = 1.3 // above RescueThreshold(1.2), but actual score is low
	// Zero out weights so candidateScore is too low to pass PromotionThreshold.
	for _, n := range tree.Nodes {
		n.Weight = 0
	}

	cfg := DefaultSelectConfig()
	cfg.PromotionThreshold = 10.0 // unattainable
	cfg.RescueThreshold = 1.0

	got := SelectCandidate(SelectInputs{
		Tree:   tree,
		Vocab:  fakeVocab("v1"),
		Reason: "prune",
	}, cfg)
	if got == nil {
		t.Error("expected rescue-path candidate for cooling-but-once-hot tree")
	}

	// Same tree on the "merge" path should not be rescued — rescue only
	// applies to prune, not merge, because merge keeps the content.
	got = SelectCandidate(SelectInputs{
		Tree:   tree,
		Vocab:  fakeVocab("v1"),
		Reason: "merge",
	}, cfg)
	if got != nil {
		t.Error("rescue path should not apply to merge")
	}
}

func TestSelectCandidate_SuggestsMergeTarget(t *testing.T) {
	tree := buildTestTree(5, true)
	// Replace the filler leaf content so the candidate fingerprint lines
	// up with the existing memory's keyword space — otherwise the cosine
	// stays below the default merge-suggest threshold due to "leaf
	// content X" noise dominating the vector.
	idx := 0
	for _, n := range tree.Nodes {
		if n.ID == tree.RootID {
			continue
		}
		terms := []string{
			"configure jwt auth session",
			"refresh jwt auth session token",
			"fix session expiry auth jwt",
			"rotate jwt session auth",
			"add auth jwt session endpoint",
		}
		n.Content = terms[idx%len(terms)]
		idx++
	}
	existing := []IndexEntry{
		{
			ID:          "mem_existing",
			Title:       "Auth & session model",
			Fingerprint: map[string]float64{"auth": 1.0, "jwt": 1.0, "session": 1.0},
		},
	}
	cfg := DefaultSelectConfig()
	cfg.MergeSuggestCosine = 0.3 // lenient for the unit-test fakeVocab

	got := SelectCandidate(SelectInputs{
		Tree:            tree,
		GuideSummaries:  []string{"added JWT auth session token"},
		Vocab:           fakeVocab("v1"),
		Reason:          "prune",
		ExistingEntries: existing,
	}, cfg)
	if got == nil {
		t.Fatal("expected candidate")
	}
	if got.SuggestedAction != "merge" {
		t.Errorf("suggestedAction = %q, want merge", got.SuggestedAction)
	}
	if got.SuggestedTargetID != "mem_existing" {
		t.Errorf("suggestedTargetId = %q, want mem_existing", got.SuggestedTargetID)
	}
}

func TestDedupCandidates_CollapsesSameTree(t *testing.T) {
	a := &Candidate{SourceTreeID: "t1", TempID: "a", Fingerprint: map[string]float64{"x": 1.0}}
	b := &Candidate{SourceTreeID: "t1", TempID: "b", Fingerprint: map[string]float64{"x": 1.0}}
	c := &Candidate{SourceTreeID: "t2", TempID: "c", Fingerprint: map[string]float64{"y": 1.0}}
	out := DedupCandidates([]*Candidate{a, b, c}, 0.5)
	if len(out) != 2 {
		t.Errorf("expected 2 unique candidates, got %d", len(out))
	}
}

func TestApplyRedact(t *testing.T) {
	in := []string{"api key is sk-abcd1234 leaking", "nothing sensitive here"}
	out := applyRedact(in, []string{`sk-[a-z0-9]+`})
	if strings.Contains(out[0], "sk-abcd") {
		t.Errorf("expected secret redacted, got %q", out[0])
	}
	if out[1] != "nothing sensitive here" {
		t.Errorf("innocent string should pass through, got %q", out[1])
	}
}
