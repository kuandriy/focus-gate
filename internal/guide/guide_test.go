package guide

import (
	"strings"
	"testing"

	"github.com/kuandriy/focus-gate/internal/forest"
)

func TestGuideAdd(t *testing.T) {
	g := New(5)
	g.Add("implemented auth", "node1", nil)
	g.Add("fixed migration", "node2", []string{"db/migration.sql"})

	if len(g.Entries) != 2 {
		t.Errorf("entries = %d, want 2", len(g.Entries))
	}
	if g.Entries[1].Summary != "fixed migration" {
		t.Errorf("second entry summary = %q", g.Entries[1].Summary)
	}
	if len(g.Entries[1].Refs) != 1 {
		t.Errorf("refs = %v, want 1 ref", g.Entries[1].Refs)
	}
}

func TestGuideAddOverflow(t *testing.T) {
	g := New(3)
	g.Add("one", "n1", nil)
	g.Add("two", "n2", nil)
	g.Add("three", "n3", nil)
	g.Add("four", "n4", nil)

	if len(g.Entries) != 3 {
		t.Errorf("entries = %d, want 3 (max)", len(g.Entries))
	}
	// Oldest ("one") should be evicted
	if g.Entries[0].Summary != "two" {
		t.Errorf("first entry = %q, want 'two' (oldest evicted)", g.Entries[0].Summary)
	}
}

func TestGuideAddEmpty(t *testing.T) {
	g := New(5)
	g.Add("", "node1", nil)
	if len(g.Entries) != 0 {
		t.Error("empty summary should not be added")
	}
}

func TestGuideRenderValidLinks(t *testing.T) {
	g := New(5)
	g.Add("implemented auth", "node1", nil)
	g.Add("fixed database", "node2", nil)

	// Create a forest with node1 but not node2
	f := forest.NewForest()
	tree := forest.NewTree("auth topic", "")
	// Manually set the root node ID to "node1" for testing
	root := tree.Root()
	root.ID = "node1"
	tree.Nodes["node1"] = root
	delete(tree.Nodes, tree.RootID)
	tree.RootID = "node1"
	f.AddTree(tree)

	rendered := g.Render(f)

	if !strings.Contains(rendered, "implemented auth") {
		t.Error("should include entry with valid intentID")
	}
	if strings.Contains(rendered, "fixed database") {
		t.Error("should exclude entry with invalid (pruned) intentID")
	}
}

func TestGuideRenderEmpty(t *testing.T) {
	g := New(5)
	f := forest.NewForest()
	if g.Render(f) != "" {
		t.Error("empty guide should render empty string")
	}
}

func TestGuideRenderFormat(t *testing.T) {
	g := New(5)
	g.Add("did something", "", nil) // empty intentID = always shown

	f := forest.NewForest()
	rendered := g.Render(f)

	if !strings.HasPrefix(rendered, "Guide:\n") {
		t.Errorf("should start with 'Guide:\\n', got %q", rendered)
	}
	if !strings.Contains(rendered, "  - did something") {
		t.Error("should contain formatted entry")
	}
}
