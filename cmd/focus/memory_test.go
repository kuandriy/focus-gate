package main

import (
	"testing"

	"github.com/kuandriy/focus-gate/internal/forest"
	"github.com/kuandriy/focus-gate/internal/guide"
)

// TestGuideSummariesForTree pins the per-tree guide scoping that
// attachMemoryCollector relies on. Each summary must be routed to the
// tree whose nodes contain its IntentID — otherwise every candidate's
// fingerprint absorbs every guide entry and the hasGuideEntries floor
// degenerates to "guide non-empty".
func TestGuideSummariesForTree(t *testing.T) {
	treeA := forest.NewTree("auth jwt", "nodeA")
	treeB := forest.NewTree("database migrations", "nodeB")

	gd := guide.New(10)
	gd.Entries = []guide.Entry{
		{Summary: "matches A", IntentID: treeA.RootID},
		{Summary: "matches B", IntentID: treeB.RootID},
		{Summary: "anonymous", IntentID: ""},
		{Summary: "dangling reference", IntentID: "ghost-id"},
	}

	gotA := guideSummariesForTree(gd, treeA)
	if len(gotA) != 1 || gotA[0] != "matches A" {
		t.Errorf("treeA summaries = %v, want [\"matches A\"]", gotA)
	}

	gotB := guideSummariesForTree(gd, treeB)
	if len(gotB) != 1 || gotB[0] != "matches B" {
		t.Errorf("treeB summaries = %v, want [\"matches B\"]", gotB)
	}

	if got := guideSummariesForTree(nil, treeA); got != nil {
		t.Errorf("nil guide should produce nil, got %v", got)
	}
	if got := guideSummariesForTree(gd, nil); got != nil {
		t.Errorf("nil tree should produce nil, got %v", got)
	}
}
