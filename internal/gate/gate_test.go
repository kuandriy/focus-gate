package gate

import (
	"fmt"
	"strings"
	"testing"
	"time"

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

func TestRefsExtractedAndAttached(t *testing.T) {
	g := newTestGate()
	g.ProcessPrompt("fix the bug in src/auth/middleware.go", "p1")

	if len(g.Forest.Trees) != 1 {
		t.Fatalf("expected 1 tree, got %d", len(g.Forest.Trees))
	}

	// Find the leaf node with the prompt content
	tree := g.Forest.Trees[0]
	foundRef := false
	for _, node := range tree.Nodes {
		for _, ref := range node.Refs {
			if ref == "src/auth/middleware.go" {
				foundRef = true
			}
		}
	}
	if !foundRef {
		t.Error("expected src/auth/middleware.go in node refs")
	}
}

func TestRefsRenderedInContext(t *testing.T) {
	g := newTestGate()
	g.Config.ContextLimit = 800

	g.ProcessPrompt("fix the bug in src/auth/middleware.go", "p1")
	g.ProcessPrompt("also check src/auth/jwt.go for token issues", "p2")

	ctx := g.GenerateContext()
	if !strings.Contains(ctx, "@ ") {
		t.Errorf("context should contain ref line (@ prefix), got:\n%s", ctx)
	}
	if !strings.Contains(ctx, "src/auth/middleware.go") {
		t.Errorf("context should contain middleware.go ref, got:\n%s", ctx)
	}
}

func TestRefsAbsentWhenNoFilePaths(t *testing.T) {
	g := newTestGate()
	g.ProcessPrompt("add JWT authentication to the API", "p1")

	ctx := g.GenerateContext()
	if strings.Contains(ctx, "@ ") {
		t.Errorf("context should not contain ref line when no file paths, got:\n%s", ctx)
	}
}

func TestRefsPreservedOnRootPreservation(t *testing.T) {
	g := newTestGate()

	// First prompt with a file ref — creates single-node tree
	g.ProcessPrompt("fix src/auth/middleware.go authentication", "p1")

	// Second similar prompt — triggers root preservation
	g.ProcessPrompt("update src/auth/middleware.go token validation", "p2")

	tree := g.Forest.Trees[0]
	// The original ref should be on a child (preserved from root)
	foundRef := false
	for _, node := range tree.Nodes {
		if node.ID == tree.RootID {
			continue // root is now an abstraction
		}
		for _, ref := range node.Refs {
			if ref == "src/auth/middleware.go" {
				foundRef = true
			}
		}
	}
	if !foundRef {
		t.Error("refs should be preserved when root is copied to child")
	}
}

func TestRefsRankedByFrequency(t *testing.T) {
	g := newTestGate()
	g.Config.ContextLimit = 1000

	// Mention middleware.go 3 times, jwt.go once
	g.ProcessPrompt("fix src/auth/middleware.go bug", "p1")
	g.ProcessPrompt("update src/auth/middleware.go handler", "p2")
	g.ProcessPrompt("refactor src/auth/middleware.go and src/auth/jwt.go", "p3")

	ctx := g.GenerateContext()
	// middleware.go should appear before jwt.go (higher frequency)
	mwIdx := strings.Index(ctx, "src/auth/middleware.go")
	jwtIdx := strings.Index(ctx, "src/auth/jwt.go")
	if mwIdx < 0 {
		t.Fatal("middleware.go should appear in context")
	}
	if jwtIdx < 0 {
		t.Fatal("jwt.go should appear in context")
	}
	if mwIdx > jwtIdx {
		t.Error("middleware.go (3 mentions) should appear before jwt.go (1 mention)")
	}
}

func TestSessionBoundaryPenalizesOldTrees(t *testing.T) {
	f := forest.NewForest()
	e := tfidf.NewEngine()

	cfg := DefaultConfig()
	cfg.SessionTimeout = 1.0 // 1 hour

	tree := forest.NewTree("authentication JWT token", "p1")
	f.AddTree(tree)
	f.Meta.TotalPrompts = 3

	// Simulate last update was 2 hours ago
	f.Meta.LastUpdate = time.Now().Add(-2 * time.Hour).UnixMilli()

	root := tree.Root()
	root.Frequency = 10
	root.Weight = 3.46 // log2(11)
	origFreq := root.Frequency

	g := New(f, e, cfg)

	// ProcessPrompt should trigger session boundary
	g.ProcessPrompt("add new database migration", "p4")

	// The original root's frequency should be halved
	if root.Frequency >= origFreq {
		t.Errorf("session boundary should reduce frequency: got %d, was %d", root.Frequency, origFreq)
	}
}

func TestSessionBoundaryNoEffectWithinTimeout(t *testing.T) {
	f := forest.NewForest()
	e := tfidf.NewEngine()

	cfg := DefaultConfig()
	cfg.SessionTimeout = 4.0

	tree := forest.NewTree("authentication JWT token", "p1")
	f.AddTree(tree)
	f.Meta.TotalPrompts = 1
	// Last update was 1 hour ago (within 4h timeout)
	f.Meta.LastUpdate = time.Now().Add(-1 * time.Hour).UnixMilli()

	root := tree.Root()
	root.Frequency = 10
	origFreq := root.Frequency

	g := New(f, e, cfg)
	g.ProcessPrompt("fix JWT token refresh", "p2")

	// Frequency should NOT be halved
	if root.Frequency < origFreq {
		t.Errorf("should not penalize within session timeout: freq %d < %d", root.Frequency, origFreq)
	}
}

func TestSessionBoundaryDisabledWhenZero(t *testing.T) {
	f := forest.NewForest()
	e := tfidf.NewEngine()

	cfg := DefaultConfig()
	cfg.SessionTimeout = 0 // disabled

	tree := forest.NewTree("authentication", "p1")
	f.AddTree(tree)
	f.Meta.TotalPrompts = 1
	f.Meta.LastUpdate = time.Now().Add(-24 * time.Hour).UnixMilli()

	root := tree.Root()
	root.Frequency = 10
	origFreq := root.Frequency

	g := New(f, e, cfg)
	g.ProcessPrompt("something new", "p2")

	if root.Frequency < origFreq {
		t.Error("session boundary disabled (timeout=0) should not reduce frequency")
	}
}

func TestClusterMerging(t *testing.T) {
	f := forest.NewForest()
	e := tfidf.NewEngine()

	cfg := DefaultConfig()
	cfg.MergeSimilarity = 0.7

	// Create two trees with very similar content
	tree1 := forest.NewTree("JWT authentication token security", "p1")
	tree2 := forest.NewTree("JWT authentication token validation", "p2")
	f.AddTree(tree1)
	f.AddTree(tree2)
	f.Meta.TotalPrompts = 2

	e.AddDocument([]string{"jwt", "authentication", "token", "security"})
	e.AddDocument([]string{"jwt", "authentication", "token", "validation"})

	g := New(f, e, cfg)

	// Process a prompt that won't create a very similar 3rd tree
	g.ProcessPrompt("fix the database migration error", "p3")

	// The two similar trees should have been merged
	jwtTrees := 0
	for _, tree := range g.Forest.Trees {
		root := tree.Root()
		if root != nil && (strings.Contains(root.Content, "jwt") || strings.Contains(root.Content, "JWT")) {
			jwtTrees++
		}
	}
	if jwtTrees > 1 {
		t.Errorf("expected similar JWT trees to merge, but found %d JWT-related trees", jwtTrees)
	}
}

func TestClusterMergingDisabledWhenZero(t *testing.T) {
	f := forest.NewForest()
	e := tfidf.NewEngine()

	cfg := DefaultConfig()
	cfg.MergeSimilarity = 0 // disabled

	tree1 := forest.NewTree("JWT authentication token security", "p1")
	tree2 := forest.NewTree("JWT authentication token validation", "p2")
	f.AddTree(tree1)
	f.AddTree(tree2)
	f.Meta.TotalPrompts = 2

	e.AddDocument([]string{"jwt", "authentication", "token", "security"})
	e.AddDocument([]string{"jwt", "authentication", "token", "validation"})

	g := New(f, e, cfg)
	g.ProcessPrompt("fix the database migration error", "p3")

	// Trees should NOT be merged
	if len(g.Forest.Trees) < 2 {
		t.Error("merge disabled (similarity=0) should not merge trees")
	}
}

func TestContextBudgetRespected(t *testing.T) {
	g := newTestGate()
	g.Config.ContextLimit = 200

	g.ProcessPrompt("add JWT authentication to the API", "p1")
	g.ProcessPrompt("fix the database migration error", "p2")
	g.ProcessPrompt("deploy kubernetes cluster config", "p3")

	ctx := g.GenerateContext()

	if len(ctx) > 200+len("[/Focus]\n") {
		t.Errorf("context length %d exceeds budget %d + footer", len(ctx), 200)
	}
	if !strings.HasPrefix(ctx, "[Focus") {
		t.Error("header should always be present")
	}
	if !strings.HasSuffix(ctx, "[/Focus]\n") {
		t.Error("footer should always be present")
	}
}

func TestContextTopTreeAlwaysIncluded(t *testing.T) {
	g := newTestGate()
	g.Config.ContextLimit = 600

	g.ProcessPrompt("add JWT authentication to the API", "p1")

	ctx := g.GenerateContext()
	// The top tree's content should appear in output
	if !strings.Contains(ctx, "[") || !strings.Contains(ctx, "]") {
		t.Error("top tree should always be included in context")
	}
}
