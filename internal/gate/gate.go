package gate

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/kuandriy/focus-gate/internal/forest"
	"github.com/kuandriy/focus-gate/internal/guide"
	"github.com/kuandriy/focus-gate/internal/text"
	"github.com/kuandriy/focus-gate/internal/tfidf"
)

// Config holds gate classification parameters.
type Config struct {
	ExtendThreshold float64 `json:"extend"`
	BranchThreshold float64 `json:"branch"`
	BubbleUpTerms   int     `json:"bubbleUpTerms"`
	MaxRefsPerNode  int     `json:"maxRefsPerNode"`
	MemorySize      int     `json:"memorySize"`
	DecayRate       float64 `json:"decayRate"`
	ContextLimit    int     `json:"contextLimit"`
	SessionTimeout  float64 `json:"sessionTimeout"`  // hours; 0 = disabled
	MergeSimilarity float64 `json:"mergeSimilarity"` // threshold for cluster merging; 0 = disabled
	// ProjectDir is the working directory used to validate extracted file
	// refs. When non-empty, refs not present on disk relative to this dir
	// are dropped before being stored on a node. Empty disables validation.
	ProjectDir string `json:"-"`
	// TypoTolerance controls whether novel tokens get canonicalised against
	// existing TF-IDF vocabulary to absorb misspellings. Disabled zero value
	// preserves the original tokenizer behaviour exactly.
	TypoTolerance text.CanonicalizeOpts `json:"typoTolerance"`
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{
		ExtendThreshold: 0.55,
		BranchThreshold: 0.25,
		BubbleUpTerms:   6,
		MaxRefsPerNode:  5,
		MemorySize:      100,
		DecayRate:       0.05,
		ContextLimit:    600,
		SessionTimeout:  4.0,
		MergeSimilarity: 0.6,
	}
}

// Action describes how a prompt was classified.
type Action int

const (
	ActionNew      Action = iota // Unrelated — start a new topic tree
	ActionBranch                 // Broadly related — add under root
	ActionExtend                 // Closely related — add near matching leaf
	ActionContinue               // Terse/unknown-vocabulary prompt — attach to last active tree
	ActionSkip                   // Terse prompt with no tree to continue — drop silently
)

func (a Action) String() string {
	switch a {
	case ActionNew:
		return "new"
	case ActionBranch:
		return "branch"
	case ActionExtend:
		return "extend"
	case ActionContinue:
		return "continue"
	case ActionSkip:
		return "skip"
	}
	return "unknown"
}

// Classification holds the result of classifying a prompt against the forest.
type Classification struct {
	Action  Action
	TreeIdx int
	LeafID  string // For extend: the matching leaf
	Score   float64
}

// Gate is the Focus Gate classifier. It classifies prompts, mutates the forest,
// and generates context output.
type Gate struct {
	Forest *forest.Forest
	Engine *tfidf.Engine
	Config Config

	// vecCache stores pre-computed TF-IDF vectors keyed by node ID. classify()
	// would otherwise re-tokenize and re-vectorize every node on every prompt.
	// Entries are lazily populated on first access and invalidated when a node's
	// content changes (bubbleUp). The cache is transient — not persisted — because
	// IDF weights shift as documents are added or removed between sessions.
	vecCache map[string]tfidf.Vector

	// lastCls is the classification produced by the most recent ProcessPrompt
	// call. GenerateContext reads it via haveLastCls to annotate the injected
	// header with the action taken ("extend"/"branch"/"new"/"continue"),
	// giving the user an at-a-glance audit trail. Read-only paths
	// (handleStatus, /focus status) leave haveLastCls=false and the tag is
	// omitted from the header.
	lastCls     Classification
	haveLastCls bool

	// lastPromptVec caches the TF-IDF vector of the most recent prompt so
	// hook callers can reuse it for downstream features (e.g. long-term
	// memory surfacing) without re-tokenising. Nil before the first
	// ProcessPrompt or when the prompt produced no usable signal.
	lastPromptVec tfidf.Vector

	// OnTreeAtRisk is an optional callback invoked just before a tree's
	// content is at risk of being lost — right before Forest.Prune runs
	// (reason="prune", called once per tree) and right before tryMerge
	// absorbs the smaller tree (reason="merge", called with the smaller
	// tree). The callback is expected to inspect the tree's state and
	// persist any downstream artefacts (e.g. long-term memory
	// candidates). Kept as a callback rather than a direct dependency so
	// the Gate package stays free of the memory package's imports.
	OnTreeAtRisk func(tree *forest.Tree, reason string)
}

// LastPromptVector returns the TF-IDF vector of the most recent prompt
// that ProcessPrompt classified. Returns nil if no prompt has been
// processed yet or the most recent prompt was terse/empty.
func (g *Gate) LastPromptVector() tfidf.Vector { return g.lastPromptVec }

// New creates a Gate from existing forest and engine state.
func New(f *forest.Forest, e *tfidf.Engine, cfg Config) *Gate {
	return &Gate{Forest: f, Engine: e, Config: cfg, vecCache: make(map[string]tfidf.Vector)}
}

// recordClassification appends a log entry to the forest's ring buffer so
// /focus last can surface recent classifications. A trimmed prompt snippet is
// stored rather than the full text — the buffer is meant for quick triage, not
// audit. Skips are included because they represent real hook firings the user
// may want to understand ("why didn't that terse prompt get recorded?").
func (g *Gate) recordClassification(cls Classification, prompt string) {
	snippet := prompt
	if len(snippet) > 80 {
		snippet = snippet[:80] + "..."
	}
	treeID := ""
	if cls.TreeIdx >= 0 && cls.TreeIdx < len(g.Forest.Trees) {
		// For ActionNew the tree isn't in the forest yet; leave TreeID empty
		// and let the viewer resolve the latest tree from ordering.
		if cls.Action != ActionNew {
			treeID = g.Forest.Trees[cls.TreeIdx].ID
		}
	}
	g.Forest.RecordClassification(forest.ClassificationLog{
		Action:    cls.Action.String(),
		Score:     cls.Score,
		TreeID:    treeID,
		Prompt:    snippet,
		Timestamp: time.Now().UnixMilli(),
	})
}

// tokenize runs the canonicalising tokenizer so incoming prompts, node
// content, bubble-up abstractions, and guide summaries all map to the same
// vocabulary. When TypoTolerance is disabled this collapses to plain
// text.Tokenize, so the behaviour is a pure superset of the previous path.
func (g *Gate) tokenize(s string) []string {
	return text.TokenizeWithCorpus(s, g.Engine.DocFreq, g.Config.TypoTolerance)
}

// updatePeaks walks every tree and bumps its all-time PeakScore. Idempotent
// and cheap — ObservePeak only writes when the new score beats the stored
// peak. Called before any mutation that could absorb or delete a tree
// (prune, tryMerge) so the OnTreeAtRisk observer sees the latest figure.
// GenerateContext also calls ObservePeak on its own scoring pass; duplicating
// the work there is harmless and keeps peak tracking on the simple
// read-only path too.
func (g *Gate) updatePeaks() {
	if len(g.Forest.Trees) == 0 {
		return
	}
	now := time.Now().UnixMilli()
	for _, t := range g.Forest.Trees {
		root := t.Root()
		if root == nil {
			continue
		}
		t.ObservePeak(root.Score(now, g.Config.DecayRate))
	}
}

// nodeVec returns the TF-IDF vector for a node, caching the result.
// Reduces classify() cost from O(nodes × tokenize) to O(nodes × dot_product)
// after initial computation. Cache entries are invalidated in bubbleUp.
//
// Uses the canonicalising tokenizer (via g.tokenize) so that a node whose
// content contains a typo is still vectorised in the same space as an
// incoming correctly-spelled prompt — without this, the cosine comparison
// would under-score related nodes whenever the underlying text drifted.
func (g *Gate) nodeVec(nodeID string, content string) tfidf.Vector {
	if v, ok := g.vecCache[nodeID]; ok {
		return v
	}
	v := g.Engine.VectorizeTokens(g.tokenize(content))
	g.vecCache[nodeID] = v
	return v
}

// ProcessPrompt classifies a prompt, applies it to the forest, and returns context.
func (g *Gate) ProcessPrompt(prompt string, source string) string {
	tokens := g.tokenize(prompt)

	// Session boundary detection: if the gap since last prompt exceeds the
	// timeout, penalize all existing tree scores so stale context doesn't
	// dominate. We halve every node's frequency (which drives weight), making
	// old trees easier to prune and new prompts more likely to create fresh trees.
	g.applySessionBoundary()

	vec := g.Engine.VectorizeTokens(tokens)
	g.lastPromptVec = vec

	// Decide the action. Terse prompts (short and/or missing IDF signal) are
	// routed to ActionContinue/ActionSkip before calling the pure classifier,
	// so classify() only ever sees inputs that warrant real scoring.
	var cls Classification
	if isTerse(tokens, vec) {
		if len(g.Forest.Trees) == 0 {
			cls = Classification{Action: ActionSkip}
		} else {
			cls = Classification{
				Action:  ActionContinue,
				TreeIdx: lastActiveTreeIndex(g.Forest),
			}
		}
	} else {
		cls = g.classify(vec)
	}

	// Record the classification before any early return so /focus last reflects
	// every real hook firing, including skips.
	g.recordClassification(cls, prompt)
	g.lastCls = cls
	g.haveLastCls = true

	if cls.Action == ActionSkip {
		return ""
	}

	g.apply(cls, prompt, source, tokens)

	g.Forest.Meta.TotalPrompts++
	g.Forest.Meta.LastUpdate = time.Now().UnixMilli()

	// Continuation leaves carry terse/novel tokens that would distort IDF
	// (inflating DF for throwaway terms like "fix"/"yes"). Keep them out of
	// the corpus so the next identical terse prompt still vectorizes to nil
	// and falls through to continuation, rather than matching the previous
	// continuation node via a weakly-weighted term.
	if cls.Action != ActionContinue {
		g.Engine.AddDocument(tokens)
		// Reset vector cache — AddDocument shifts IDF globally, so all
		// previously cached vectors are stale.
		g.vecCache = make(map[string]tfidf.Vector)
	}

	// Prune if needed
	if g.Forest.NodeCount() > g.Config.MemorySize {
		// Refresh all-time peak scores *before* firing the at-risk
		// observer so long-term memory's rescue path (which compares a
		// tree's PeakScore against RescueThreshold) sees an up-to-date
		// figure. Without this, GenerateContext is the only site that
		// calls ObservePeak — and on a fresh tree that got hot and is
		// about to be pruned in the same cycle, the peak would still be
		// 0 when SelectCandidate reads it, so rescue would never trigger.
		g.updatePeaks()
		// Give the at-risk observer a chance to persist anything worth
		// preserving (long-term memory candidates) before we mutate.
		// Fires once per tree regardless of which leaves actually get
		// removed — SelectCandidate filters by score so trees that don't
		// qualify produce nothing.
		if g.OnTreeAtRisk != nil {
			for _, t := range g.Forest.Trees {
				g.OnTreeAtRisk(t, "prune")
			}
		}
		removed := g.Forest.Prune(g.Config.MemorySize, g.Config.DecayRate)
		for _, content := range removed {
			// Tokenise via canonicaliser so pruning decrements the same
			// DocFreq keys that AddDocument inserted at indexing time.
			g.Engine.RemoveDocument(g.tokenize(content))
		}
	}

	// Cluster merging: after classification and pruning, check for trees that
	// have drifted toward the same topic and merge the smaller into the larger.
	// Skip after a continuation — no new semantic signal to justify merging.
	if cls.Action != ActionContinue {
		// Same rationale as the prune path: keep PeakScore current so
		// the tree's history is visible to OnTreeAtRisk("merge") before
		// its identity is absorbed.
		g.updatePeaks()
		g.tryMerge()
	}

	return g.GenerateContext()
}

// classify compares the prompt vector against all tree roots and leaves.
func (g *Gate) classify(vec tfidf.Vector) Classification {
	cls, _ := g.classifyDetailed(vec, false)
	return cls
}

// classifyDetailed compares the prompt vector against all tree roots and leaves.
// When detailed is true, per-tree scoring data is returned for dry-run diagnostics.
//
// This function is deliberately pure: it maps (vec, forest) onto an
// {extend, branch, new} decision. The ActionContinue / ActionSkip paths
// (terse prompts, empty forests) live one level up in ProcessPrompt, so
// DryRun and any future read-only caller get consistent scoring numbers.
func (g *Gate) classifyDetailed(vec tfidf.Vector, detailed bool) (Classification, []TreeScore) {
	if len(g.Forest.Trees) == 0 || vec == nil {
		return Classification{Action: ActionNew, Score: 0}, nil
	}

	best := Classification{Action: ActionNew, Score: 0}
	var scores []TreeScore

	for i, tree := range g.Forest.Trees {
		root := tree.Root()
		if root == nil {
			continue
		}

		// Compare against root
		rootVec := g.nodeVec(root.ID, root.Content)
		rootCosine := tfidf.CosineSimilarity(vec, rootVec)
		if rootCosine > best.Score {
			best.Score = rootCosine
			best.TreeIdx = i
			best.LeafID = ""
		}

		var ts TreeScore
		if detailed {
			ts = TreeScore{
				TreeIdx:     i,
				TreeID:      tree.ID,
				RootID:      root.ID,
				RootContent: root.Content,
				RootCosine:  rootCosine,
			}
		}

		// Compare against each leaf
		for _, leaf := range tree.GetLeaves() {
			leafVec := g.nodeVec(leaf.ID, leaf.Content)
			leafCosine := tfidf.CosineSimilarity(vec, leafVec)
			if leafCosine > best.Score {
				best.Score = leafCosine
				best.TreeIdx = i
				best.LeafID = leaf.ID
			}
			if detailed {
				ts.LeafScores = append(ts.LeafScores, LeafScore{
					LeafID:  leaf.ID,
					Content: leaf.Content,
					Cosine:  leafCosine,
				})
			}
		}

		if detailed {
			scores = append(scores, ts)
		}
	}

	if best.Score >= g.Config.ExtendThreshold {
		best.Action = ActionExtend
	} else if best.Score >= g.Config.BranchThreshold {
		best.Action = ActionBranch
	} else {
		best.Action = ActionNew
	}

	return best, scores
}

// apply mutates the forest based on the classification.
func (g *Gate) apply(cls Classification, content string, source string, tokens []string) {
	refs := text.FilterExistingPaths(
		g.Config.ProjectDir,
		text.ExtractFilePaths(content, g.Config.MaxRefsPerNode),
	)

	switch cls.Action {
	case ActionSkip:
		// Terse prompt and the forest is empty — nothing to attach to.
		return

	case ActionContinue:
		// Terse/unknown-vocabulary prompt. Attach as a leaf under the last
		// active tree's root so follow-up AI context stays grounded in what
		// the user was just working on. The node is intentionally not indexed
		// in TF-IDF (it has no meaningful terms) and bubble-up is skipped
		// (it would contribute nothing to the parent abstraction).
		tree := g.Forest.Trees[cls.TreeIdx]
		g.preserveRoot(tree)
		child := tree.AddChild(tree.RootID, content, source)
		if child != nil {
			child.Indexed = false
			child.Refs = refs
		}

	case ActionNew:
		tree := forest.NewTree(content, source)
		tree.Root().Indexed = true // real user prompt — register in TF-IDF
		tree.Root().Refs = refs
		g.Forest.AddTree(tree)

	case ActionBranch:
		tree := g.Forest.Trees[cls.TreeIdx]
		g.preserveRoot(tree)
		child := tree.AddChild(tree.RootID, content, source)
		if child != nil {
			child.Indexed = true
			child.Refs = refs
		}
		g.bubbleUp(tree, tree.RootID)

	case ActionExtend:
		tree := g.Forest.Trees[cls.TreeIdx]
		leaf := tree.Nodes[cls.LeafID]
		if leaf == nil {
			// Fallback to branch
			g.preserveRoot(tree)
			child := tree.AddChild(tree.RootID, content, source)
			if child != nil {
				child.Indexed = true
				child.Refs = refs
			}
		} else {
			parentID := leaf.ParentID
			if parentID == "" {
				// Leaf is root — preserve and add as sibling
				g.preserveRoot(tree)
				parentID = tree.RootID
			}
			child := tree.AddChild(parentID, content, source)
			if child != nil {
				child.Indexed = true
				child.Refs = refs
			}
		}
		g.bubbleUp(tree, tree.RootID)
	}
}

// terseTokenThreshold is the maximum post-tokenization length for which a
// prompt with no IDF signal is treated as a continuation rather than a new
// tree. Prompts like "fix", "yes", "run it" tokenize to 1-2 tokens; longer
// prompts with nil vec (e.g. a fresh 5-word topic whose vocabulary the corpus
// has not seen yet) are genuine new topics and should fall through to the
// normal classifier.
const terseTokenThreshold = 2

// isTerse returns true when the prompt lacks enough signal to warrant a new
// tree: either empty tokens (pure stop-words / blank text) or a very short
// prompt that produces no IDF vector. Longer prompts with nil vec signal a
// genuinely new topic — they are classified normally and land as ActionNew
// via the standard path.
func isTerse(tokens []string, vec tfidf.Vector) bool {
	if len(tokens) == 0 {
		return true
	}
	return vec == nil && len(tokens) <= terseTokenThreshold
}

// lastActiveTreeIndex returns the index of the tree with the most recent
// LastAccessed timestamp. Ties break by slice order (earliest wins). Assumes
// len(Forest.Trees) > 0; the caller must check.
func lastActiveTreeIndex(f *forest.Forest) int {
	best := 0
	bestTs := f.Trees[0].LastAccessed
	for i, tree := range f.Trees {
		if tree.LastAccessed > bestTs {
			bestTs = tree.LastAccessed
			best = i
		}
	}
	return best
}

// preserveRoot handles the root preservation edge case: when a single-node tree
// gets its first branch, the root content must be copied to a child before
// bubble-up overwrites it with an abstraction.
func (g *Gate) preserveRoot(tree *forest.Tree) {
	root := tree.Root()
	if root == nil || !root.IsLeaf() {
		return
	}
	// Root is a leaf (single-node tree). Preserve its content as a child.
	child := tree.AddChild(root.ID, root.Content, "")
	if child != nil {
		child.Frequency = root.Frequency
		child.Weight = root.Weight
		child.Created = root.Created
		child.LastAccessed = root.LastAccessed
		// Inherit the index flag — the child now owns the original prompt content.
		child.Indexed = root.Indexed
		child.Refs = root.Refs
	}
}

// bubbleUp regenerates parent node content bottom-up from children.
func (g *Gate) bubbleUp(tree *forest.Tree, nodeID string) {
	node := tree.Nodes[nodeID]
	if node == nil {
		return
	}

	// Recurse children first (post-order)
	for _, childID := range node.ChildIDs {
		g.bubbleUp(tree, childID)
	}

	// Only abstract non-leaf nodes
	if node.IsLeaf() {
		return
	}

	// Clear indexed flag — bubbleUp replaces content with a synthetic abstraction
	// that was never added to the TF-IDF corpus.
	node.Indexed = false

	// Collect per-child unique tokens and count how many children contain each
	// term (presence count). This favors terms with cross-child breadth over
	// terms concentrated in a single child.
	presence := make(map[string]int)
	for _, childID := range node.ChildIDs {
		child := tree.Nodes[childID]
		if child == nil {
			continue
		}
		tokens := g.tokenize(child.Content)
		seen := make(map[string]bool, len(tokens))
		for _, t := range tokens {
			if !seen[t] {
				presence[t]++
				seen[t] = true
			}
		}
	}

	// Score each term by presence × IDF. This preserves the cross-child breadth
	// signal (terms in more children score higher) while penalizing corpus-common
	// terms like "add" or "fix" that survive stop-word filtering.
	type termScore struct {
		term  string
		score float64
	}
	sorted := make([]termScore, 0, len(presence))
	for t, count := range presence {
		idf := g.Engine.IDF(t)
		if idf == 0 {
			// Term not in corpus yet — use presence count alone as fallback.
			// This can happen when bubbleUp runs before AddDocument for
			// the newest prompt's tokens.
			idf = 1.0
		}
		sorted = append(sorted, termScore{t, float64(count) * idf})
	}
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].score != sorted[j].score {
			return sorted[i].score > sorted[j].score
		}
		return sorted[i].term < sorted[j].term
	})

	n := g.Config.BubbleUpTerms
	if n > len(sorted) {
		n = len(sorted)
	}
	terms := make([]string, n)
	for i := 0; i < n; i++ {
		terms[i] = sorted[i].term
	}

	node.Content = strings.Join(terms, " | ")

	// Invalidate cached vector — content just changed.
	delete(g.vecCache, nodeID)
}

// GenerateContext formats the forest state as a compact context block using
// budget-based prioritized rendering. Each phase checks remaining character
// budget before writing, ensuring the most important information always fits.
//
// Phase order:
//  1. Header [Focus | ...] — always included
//  2. Top tree (highest score + leaves) — always fits
//  3. Additional trees — until budget exhausted
//  4. File refs — for included trees, if budget permits
//  5. Footer [/Focus]
func (g *Gate) GenerateContext() string {
	if len(g.Forest.Trees) == 0 {
		return ""
	}

	budget := g.Config.ContextLimit
	if budget <= 0 {
		budget = 600
	}

	// Sort trees by root score descending
	type scoredTree struct {
		tree  *forest.Tree
		score float64
	}
	scored := make([]scoredTree, len(g.Forest.Trees))
	now := g.Forest.Trees[0].LastAccessed
	for i, t := range g.Forest.Trees {
		s := t.Root().Score(now, g.Config.DecayRate)
		scored[i] = scoredTree{t, s}
		// Track all-time peak — used by long-term memory candidate
		// selection to rescue cooling-but-once-hot trees from prune.
		t.ObservePeak(s)
	}
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	// Pre-compute header and footer. When a classification has just been
	// produced (ProcessPrompt path), tag the header with "<action> <score>"
	// so the user gets a one-glance audit trail of how the last prompt landed.
	// Read-only paths (handleStatus, /focus status) leave haveLastCls=false
	// and the tag is omitted from the header.
	actionTag := ""
	if g.haveLastCls {
		switch g.lastCls.Action {
		case ActionExtend, ActionBranch:
			actionTag = fmt.Sprintf(" | %s %.2f", g.lastCls.Action, g.lastCls.Score)
		case ActionNew, ActionContinue, ActionSkip:
			actionTag = fmt.Sprintf(" | %s", g.lastCls.Action)
		}
	}
	header := fmt.Sprintf("[Focus | %d prompts | %d/%d mem | %d trees%s]\n",
		g.Forest.Meta.TotalPrompts,
		g.Forest.NodeCount(),
		g.Config.MemorySize,
		len(g.Forest.Trees),
		actionTag)
	footer := "[/Focus]\n"

	remaining := budget - len(header) - len(footer)
	if remaining <= 0 {
		return header + footer
	}

	var parts []string
	var includedTrees []*forest.Tree

	// Phase 2: Top tree — attempt to include the #1 tree first; skipped only if
	// the budget is too tight to fit even a single tree block.
	if len(scored) > 0 {
		block := g.renderTree(scored[0])
		if len(block) <= remaining {
			parts = append(parts, block)
			remaining -= len(block)
			includedTrees = append(includedTrees, scored[0].tree)
		}
	}

	// Phase 3: Additional trees — up to 4 more, budget permitting
	limit := 5
	if limit > len(scored) {
		limit = len(scored)
	}
	for _, st := range scored[1:limit] {
		block := g.renderTree(st)
		if len(block) > remaining {
			break
		}
		parts = append(parts, block)
		remaining -= len(block)
		includedTrees = append(includedTrees, st.tree)
	}

	// Phase 4: File refs — add refs for included trees if budget permits
	for _, tree := range includedTrees {
		refLine := g.collectTreeRefs(tree, 3)
		if refLine == "" {
			continue
		}
		if len(refLine) > remaining {
			break
		}
		parts = append(parts, refLine)
		remaining -= len(refLine)
	}

	var b strings.Builder
	b.WriteString(header)
	for _, p := range parts {
		b.WriteString(p)
	}
	b.WriteString(footer)
	return b.String()
}

// renderTree formats a single tree block for context output (without refs).
func (g *Gate) renderTree(st struct {
	tree  *forest.Tree
	score float64
}) string {
	var b strings.Builder
	fmt.Fprintf(&b, "  [%.2f] %s\n", st.score, st.tree.Root().Content)

	leaves := st.tree.GetLeaves()
	sort.Slice(leaves, func(i, j int) bool {
		return leaves[i].LastAccessed > leaves[j].LastAccessed
	})
	leafLimit := 3
	if leafLimit > len(leaves) {
		leafLimit = len(leaves)
	}
	for _, leaf := range leaves[:leafLimit] {
		if leaf.ID == st.tree.RootID {
			continue
		}
		content := leaf.Content
		if len(content) > 80 {
			content = content[:80] + "..."
		}
		fmt.Fprintf(&b, "    - %s\n", content)
	}

	return b.String()
}

// collectTreeRefs aggregates file path refs from all nodes in a tree,
// counts by frequency, and returns a formatted line with the top N refs.
// Returns "" if no refs exist.
func (g *Gate) collectTreeRefs(tree *forest.Tree, maxRefs int) string {
	freq := make(map[string]int)
	for _, node := range tree.Nodes {
		for _, ref := range node.Refs {
			freq[ref]++
		}
	}
	if len(freq) == 0 {
		return ""
	}

	type refCount struct {
		path  string
		count int
	}
	ranked := make([]refCount, 0, len(freq))
	for path, count := range freq {
		ranked = append(ranked, refCount{path, count})
	}
	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].count != ranked[j].count {
			return ranked[i].count > ranked[j].count
		}
		return ranked[i].path < ranked[j].path
	})

	if maxRefs > len(ranked) {
		maxRefs = len(ranked)
	}
	paths := make([]string, maxRefs)
	for i := 0; i < maxRefs; i++ {
		paths[i] = ranked[i].path
	}
	return fmt.Sprintf("    @ %s\n", strings.Join(paths, ", "))
}

// applySessionBoundary detects gaps between prompts that exceed SessionTimeout
// and penalizes existing tree scores to prevent stale context from dominating.
// When triggered, all node frequencies are halved (reducing weight via log2),
// making old trees easier to prune and new prompts more likely to create fresh trees.
func (g *Gate) applySessionBoundary() {
	if g.Config.SessionTimeout <= 0 || len(g.Forest.Trees) == 0 {
		return
	}

	now := time.Now().UnixMilli()

	// LastUpdate is zero when state was loaded from disk but never written by
	// this version (e.g. first run after upgrading). Treat it as "now" to avoid
	// a false boundary trigger on the first prompt after an upgrade.
	if g.Forest.Meta.LastUpdate == 0 {
		g.Forest.Meta.LastUpdate = now
		return
	}

	gapHours := float64(now-g.Forest.Meta.LastUpdate) / 3600000.0

	if gapHours < g.Config.SessionTimeout {
		return
	}

	// Halve all node frequencies — reduces weight without destroying state
	for _, tree := range g.Forest.Trees {
		for _, node := range tree.Nodes {
			node.Frequency = (node.Frequency + 1) / 2
			if node.Frequency < 1 {
				node.Frequency = 1
			}
			node.Weight = math.Log2(float64(node.Frequency) + 1)
		}
	}
}

// tryMerge checks all tree pairs for root cosine similarity above MergeSimilarity
// and merges the smaller tree into the larger. All non-root nodes from the small
// tree are re-parented as direct children of the large tree's root (flattened).
// The small root itself is skipped — it is a bubbleUp abstraction, not a real
// prompt, and the large tree will regenerate its own abstraction after bubbleUp.
// Only one merge per ProcessPrompt call to avoid cascading merges that could
// destabilize the forest.
func (g *Gate) tryMerge() {
	if g.Config.MergeSimilarity <= 0 || len(g.Forest.Trees) < 2 {
		return
	}

	bestSim := 0.0
	bestI, bestJ := -1, -1

	for i := 0; i < len(g.Forest.Trees); i++ {
		r1 := g.Forest.Trees[i].Root()
		if r1 == nil {
			continue
		}
		v1 := g.nodeVec(r1.ID, r1.Content)

		for j := i + 1; j < len(g.Forest.Trees); j++ {
			r2 := g.Forest.Trees[j].Root()
			if r2 == nil {
				continue
			}
			v2 := g.nodeVec(r2.ID, r2.Content)
			sim := tfidf.CosineSimilarity(v1, v2)
			if sim > bestSim {
				bestSim = sim
				bestI = i
				bestJ = j
			}
		}
	}

	if bestSim < g.Config.MergeSimilarity || bestI < 0 {
		return
	}

	// Merge smaller into larger
	large, small := g.Forest.Trees[bestI], g.Forest.Trees[bestJ]
	if large.NodeCount() < small.NodeCount() {
		large, small = small, large
		bestI, bestJ = bestJ, bestI
	}

	// Fire the at-risk observer on the smaller tree before its identity
	// is absorbed. This is the prime rescue case for long-term memory —
	// a tree that was coherent enough to warrant its own bubble-up is
	// about to lose that coherence, so its state should be snapshotted
	// while it still stands.
	if g.OnTreeAtRisk != nil {
		g.OnTreeAtRisk(small, "merge")
	}

	// Move all non-root nodes from small under large's root (flat re-parent).
	// Using Nodes map directly covers leaves AND interior nodes at any depth,
	// preventing content loss in trees deeper than root→leaf.
	for _, node := range small.Nodes {
		if node.ID == small.RootID {
			continue // root is a computed abstraction; large rebuilds its own
		}
		child := large.AddChild(large.RootID, node.Content, "")
		if child != nil {
			child.Indexed = node.Indexed
			child.Frequency = node.Frequency
			child.Weight = node.Weight
			child.Created = node.Created
			child.LastAccessed = node.LastAccessed
			child.Refs = node.Refs
		}
	}

	// Re-run bubble-up on the large tree's root
	g.bubbleUp(large, large.RootID)

	// Remove the small tree from the forest
	g.Forest.Trees = append(g.Forest.Trees[:bestJ], g.Forest.Trees[bestJ+1:]...)
}

// ReinforceFromGuide processes unreinforced guide entries against the forest.
// When an AI responds about a topic, that response is evidence the topic is
// actively being worked on. We find the best-matching tree by cosine similarity
// and Touch its root, increasing its weight and recency (making it stickier
// and harder to prune).
//
// Only Touch is applied — no new nodes or content changes. AI responses confirm
// existing topics rather than defining new ones.
//
// Returns the number of entries reinforced, for diagnostic logging.
func (g *Gate) ReinforceFromGuide(gd *guide.Guide) int {
	unreinforced := gd.UnreinforcedEntries()
	if len(unreinforced) == 0 {
		return 0
	}

	reinforced := 0

	for _, entry := range unreinforced {
		tokens := g.tokenize(entry.Summary)
		if len(tokens) == 0 {
			entry.Reinforced = true
			continue
		}

		responseVec := g.Engine.VectorizeTokens(tokens)

		// Find the best-matching tree root by pure cosine similarity.
		bestScore := 0.0
		bestTreeIdx := -1

		for i, tree := range g.Forest.Trees {
			root := tree.Root()
			if root == nil {
				continue
			}
			rootVec := g.nodeVec(root.ID, root.Content)
			score := tfidf.CosineSimilarity(responseVec, rootVec)
			if score > bestScore {
				bestScore = score
				bestTreeIdx = i
			}
		}

		// Only reinforce above the branch threshold — generic responses
		// (e.g. "Sure, here's the code:") shouldn't boost any tree.
		if bestTreeIdx >= 0 && bestScore >= g.Config.BranchThreshold {
			root := g.Forest.Trees[bestTreeIdx].Root()
			if root != nil {
				root.Touch()
				reinforced++
			}
		}

		entry.Reinforced = true
	}

	return reinforced
}
