package gate

import (
	"fmt"
	"strings"
	"testing"

	"github.com/kuandriy/focus-gate/internal/forest"
	"github.com/kuandriy/focus-gate/internal/tfidf"
)

func newTestGate() *Gate {
	return New(forest.NewForest(), tfidf.NewEngine(), DefaultConfig())
}

func TestNewPromptCreatesTree(t *testing.T) {
	g := newTestGate()
	ctx := g.ProcessPrompt("add JWT authentication to the API", "p1")

	if len(g.Forest.Trees) != 1 {
		t.Fatalf("expected 1 tree, got %d", len(g.Forest.Trees))
	}
	if ctx == "" {
		t.Error("context should not be empty")
	}
	if !strings.Contains(ctx, "[Focus") {
		t.Error("context should contain [Focus header")
	}
	if !strings.Contains(ctx, "[/Focus]") {
		t.Error("context should contain [/Focus] footer")
	}
}

func TestDissimilarPromptsCreateSeparateTrees(t *testing.T) {
	g := newTestGate()
	g.ProcessPrompt("add JWT authentication to the API", "p1")
	g.ProcessPrompt("fix the database migration schema error", "p2")

	if len(g.Forest.Trees) != 2 {
		t.Errorf("expected 2 trees for dissimilar prompts, got %d", len(g.Forest.Trees))
	}
}

func TestSimilarPromptExtends(t *testing.T) {
	g := newTestGate()
	g.ProcessPrompt("add JWT authentication to the API", "p1")
	g.ProcessPrompt("fix JWT authentication token expiry", "p2")

	if len(g.Forest.Trees) != 1 {
		t.Errorf("expected 1 tree for similar prompts, got %d", len(g.Forest.Trees))
	}
	tree := g.Forest.Trees[0]
	if tree.NodeCount() < 3 {
		t.Errorf("expected >= 3 nodes (root + 2 leaves), got %d", tree.NodeCount())
	}
}

func TestRootPreservation(t *testing.T) {
	g := newTestGate()

	// First prompt creates a single-node tree
	g.ProcessPrompt("add JWT authentication to the API", "p1")
	tree := g.Forest.Trees[0]
	originalContent := tree.Root().Content

	// Second similar prompt should trigger root preservation
	g.ProcessPrompt("fix JWT authentication token expiry", "p2")

	// The original content should exist as a child leaf
	found := false
	for _, node := range tree.Nodes {
		if node.ID != tree.RootID && node.Content == originalContent {
			found = true
			break
		}
	}
	if !found {
		t.Error("original root content should be preserved as a child after first branch")
	}

	// Root should now be an abstraction (pipe-separated terms)
	root := tree.Root()
	if !strings.Contains(root.Content, "|") && tree.NodeCount() > 2 {
		t.Errorf("root should be abstracted after bubble-up, got %q", root.Content)
	}
}

func TestBubbleUpGeneratesAbstraction(t *testing.T) {
	g := newTestGate()

	f := g.Forest
	tree := forest.NewTree("placeholder", "")
	root := tree.Root()
	tree.AddChild(root.ID, "add JWT authentication token", "")
	tree.AddChild(root.ID, "fix JWT token expiry bug", "")
	tree.AddChild(root.ID, "refresh JWT token rotation", "")
	f.AddTree(tree)

	g.bubbleUp(tree, tree.RootID)

	// Root content should be pipe-separated top terms
	rootContent := root.Content
	if !strings.Contains(rootContent, "|") {
		t.Errorf("bubble-up should create pipe-separated abstraction, got %q", rootContent)
	}
	if !strings.Contains(rootContent, "jwt") && !strings.Contains(rootContent, "token") {
		t.Errorf("bubble-up should include common terms like 'jwt' or 'token', got %q", rootContent)
	}
}

func TestContextFormat(t *testing.T) {
	g := newTestGate()
	g.ProcessPrompt("add authentication to the app", "p1")

	ctx := g.GenerateContext()

	if !strings.HasPrefix(ctx, "[Focus |") {
		t.Errorf("context should start with [Focus |, got %q", ctx[:20])
	}
	if !strings.HasSuffix(ctx, "[/Focus]\n") {
		t.Errorf("context should end with [/Focus], got %q", ctx[len(ctx)-20:])
	}
	if !strings.Contains(ctx, "prompts") {
		t.Error("context should contain prompt count")
	}
	if !strings.Contains(ctx, "mem") {
		t.Error("context should contain memory usage")
	}
	if !strings.Contains(ctx, "trees") {
		t.Error("context should contain tree count")
	}
}

func TestPruningTriggered(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MemorySize = 5
	g := New(forest.NewForest(), tfidf.NewEngine(), cfg)

	// Generate enough unique prompts to exceed memory
	prompts := []string{
		"authentication JWT token security",
		"database migration schema postgres",
		"frontend react component styling",
		"deployment docker kubernetes cluster",
		"testing unit integration coverage",
		"logging monitoring alerting metrics",
	}
	for i, p := range prompts {
		g.ProcessPrompt(p, fmt.Sprintf("p%d", i))
	}

	if g.Forest.NodeCount() > cfg.MemorySize {
		t.Errorf("after pruning: NodeCount = %d, want <= %d", g.Forest.NodeCount(), cfg.MemorySize)
	}
}

func TestEmptyPromptNoOp(t *testing.T) {
	g := newTestGate()
	ctx := g.ProcessPrompt("", "p1")
	if ctx != "" {
		t.Errorf("empty prompt should return empty context, got %q", ctx)
	}
	if len(g.Forest.Trees) != 0 {
		t.Error("empty prompt should not create trees")
	}
}

func TestStopWordsOnlyNoOp(t *testing.T) {
	g := newTestGate()
	ctx := g.ProcessPrompt("the and or but in on at to for", "p1")
	if ctx != "" {
		t.Errorf("stop-words-only prompt should return empty context, got %q", ctx)
	}
}

// Ensure fmt is used
var _ = fmt.Sprintf
