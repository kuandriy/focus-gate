package main

import (
	"os"
	"testing"

	"github.com/kuandriy/focus-gate/internal/forest"
)

func TestParseSlashCommand(t *testing.T) {
	tests := []struct {
		name    string
		raw     string
		wantOk  bool
		wantSub string
		wantArg string
	}{
		{"empty", "", false, "", ""},
		{"normal prompt", "add JWT authentication", false, "", ""},
		{"just /focus", "/focus", true, "help", ""},
		{"focus with space", "  /focus  ", true, "help", ""},
		{"help", "/focus help", true, "help", ""},
		{"status", "/focus status", true, "status", ""},
		{"inspect", "/focus inspect", true, "inspect", ""},
		{"tree no arg", "/focus tree", true, "tree", ""},
		{"tree with index", "/focus tree 2", true, "tree", "2"},
		{"tree with id", "/focus tree abc123", true, "tree", "abc123"},
		{"terms", "/focus terms", true, "terms", ""},
		{"terms with count", "/focus terms 50", true, "terms", "50"},
		{"markov", "/focus markov", true, "markov", ""},
		{"score with prompt", "/focus score add auth to api", true, "score", "add auth to api"},
		{"health", "/focus health", true, "health", ""},
		{"case insensitive", "/FOCUS STATUS", true, "status", ""},
		{"mixed case", "/Focus Tree 0", true, "tree", "0"},
		{"unknown sub", "/focus foobar", true, "foobar", ""},
		{"not slash focus", "/focusgate status", false, "", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd, ok := parseSlashCommand(tt.raw)
			if ok != tt.wantOk {
				t.Fatalf("parseSlashCommand(%q): ok = %v, want %v", tt.raw, ok, tt.wantOk)
			}
			if !ok {
				return
			}
			if cmd.sub != tt.wantSub {
				t.Errorf("sub = %q, want %q", cmd.sub, tt.wantSub)
			}
			if cmd.arg != tt.wantArg {
				t.Errorf("arg = %q, want %q", cmd.arg, tt.wantArg)
			}
		})
	}
}

func TestFindTree(t *testing.T) {
	f := setupTestForest()

	// By index
	tree := findTree(f, "0")
	if tree == nil {
		t.Fatal("findTree by index 0 returned nil")
	}
	if tree != f.Trees[0] {
		t.Error("findTree(0) returned wrong tree")
	}

	// By index out of range
	tree = findTree(f, "99")
	if tree != nil {
		t.Error("findTree(99) should return nil")
	}

	// By partial ID
	tree = findTree(f, f.Trees[0].ID[:4])
	if tree == nil {
		t.Fatal("findTree by partial ID returned nil")
	}
	if tree != f.Trees[0] {
		t.Error("findTree partial ID returned wrong tree")
	}

	// No match
	tree = findTree(f, "zzzzzzzzz")
	if tree != nil {
		t.Error("findTree should return nil for nonexistent ID")
	}
}

func TestMemoryBar(t *testing.T) {
	bar0 := memoryBar(0)
	if bar0 != "[░░░░░░░░░░░░]" {
		t.Errorf("memoryBar(0) = %q", bar0)
	}

	bar50 := memoryBar(50)
	if bar50 != "[██████░░░░░░]" {
		t.Errorf("memoryBar(50) = %q", bar50)
	}

	bar100 := memoryBar(100)
	if bar100 != "[████████████]" {
		t.Errorf("memoryBar(100) = %q", bar100)
	}

	// Over 100% should cap at full
	bar150 := memoryBar(150)
	if bar150 != "[████████████]" {
		t.Errorf("memoryBar(150) = %q, want full bar", bar150)
	}
}

func TestFormatAge(t *testing.T) {
	tests := []struct {
		hours float64
		want  string
	}{
		{0.1, "6m"},
		{0.5, "30m"},
		{2.5, "2.5h"},
		{23.9, "23.9h"},
		{48.0, "2.0d"},
	}
	for _, tt := range tests {
		got := formatAge(tt.hours)
		if got != tt.want {
			t.Errorf("formatAge(%v) = %q, want %q", tt.hours, got, tt.want)
		}
	}
}

func TestSlashHelp(t *testing.T) {
	// Just verify it doesn't panic and returns nil error.
	f, cleanup := tmpFile(t)
	defer cleanup()
	err := slashHelp(f)
	if err != nil {
		t.Fatalf("slashHelp: %v", err)
	}
}

// --- helpers ---

func setupTestForest() *forest.Forest {
	f := forest.NewForest()
	t1 := forest.NewTree("authentication and JWT tokens", "p0")
	t1.AddChild(t1.RootID, "add refresh token rotation", "p1")
	t2 := forest.NewTree("database migration schema", "p2")
	f.Trees = append(f.Trees, t1, t2)
	return f
}

func tmpFile(t *testing.T) (*os.File, func()) {
	t.Helper()
	f, err := os.CreateTemp("", "focus-test-*")
	if err != nil {
		t.Fatal(err)
	}
	return f, func() {
		f.Close()
		os.Remove(f.Name())
	}
}
