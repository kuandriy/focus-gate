package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"time"

	"github.com/kuandriy/focus-gate/internal/forest"
	"github.com/kuandriy/focus-gate/internal/gate"
	"github.com/kuandriy/focus-gate/internal/guide"
	"github.com/kuandriy/focus-gate/internal/text"
	"github.com/kuandriy/focus-gate/internal/tfidf"
)

// ---------------------------------------------------------------------------
// CLI flag helpers
// ---------------------------------------------------------------------------

// hasFlag returns true if the given flag appears anywhere in args.
func hasFlag(args []string, flag string) bool {
	for _, a := range args {
		if a == flag {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// handleInspect — full state dump
// ---------------------------------------------------------------------------

// handleInspect loads all persisted state and prints a comprehensive view of
// every data structure: forest trees with full node hierarchy, TF-IDF corpus
// statistics, and guide entries with reinforcement state. This lets the user
// verify at a glance whether the system is tracking intent correctly after a
// series of prompts.
func handleInspect(p paths, cfg config, asJSON bool) error {
	f := loadForest(p.intentFile)
	e := loadEngine(p.engineFile)
	g := loadGuide(p.guideFile, cfg.GuideSize)

	if asJSON {
		return inspectJSON(f, e, g, cfg)
	}
	return inspectText(os.Stdout, f, e, g, cfg)
}

// ---------------------------------------------------------------------------
// handleDryRun — classify without mutation
// ---------------------------------------------------------------------------

// handleDryRun runs the full classification pipeline on a prompt without
// modifying any persisted state, showing exactly how the classifier would
// score each tree. Useful for understanding why a prompt was classified a
// certain way or testing threshold tuning.
func handleDryRun(p paths, cfg config, prompt string, asJSON bool) error {
	f := loadForest(p.intentFile)
	e := loadEngine(p.engineFile)

	// Clean the prompt the same way the hook path does.
	prompt = text.CleanPrompt(prompt)
	if prompt == "" {
		return fmt.Errorf("prompt is empty after cleaning")
	}

	gt := gate.New(f, e, toGateConfig(cfg))
	result := gt.DryRun(prompt)

	if asJSON {
		return dryRunJSON(result)
	}
	return dryRunText(result, cfg)
}

// ---------------------------------------------------------------------------
// Text formatters
// ---------------------------------------------------------------------------

func inspectText(w io.Writer, f *forest.Forest, e *tfidf.Engine, g *guide.Guide, cfg config) error {
	now := time.Now().UnixMilli()

	fmt.Fprintln(w, "=== Focus Gate Inspect ===")
	fmt.Fprintln(w)

	// --- Config ---
	// Print every knob, not the curated 7 — `--inspect` is the
	// debugging surface, so a user tuning thresholds needs to see
	// what's actually in effect (typoTolerance, memory weights,
	// session boundary, merge similarity).
	fmt.Fprintln(w, "--- Config ---")
	fmt.Fprintf(w, "  memorySize:        %d\n", cfg.MemorySize)
	fmt.Fprintf(w, "  decayRate:         %.3f\n", cfg.DecayRate)
	fmt.Fprintf(w, "  similarity.extend: %.3f\n", cfg.Similarity.Extend)
	fmt.Fprintf(w, "  similarity.branch: %.3f\n", cfg.Similarity.Branch)
	fmt.Fprintf(w, "  contextLimit:      %d\n", cfg.ContextLimit)
	fmt.Fprintf(w, "  bubbleUpTerms:     %d\n", cfg.BubbleUpTerms)
	fmt.Fprintf(w, "  maxRefsPerNode:    %d\n", cfg.MaxRefsPerNode)
	fmt.Fprintf(w, "  guideSize:         %d\n", cfg.GuideSize)
	fmt.Fprintf(w, "  sessionTimeout:    %.2fh\n", cfg.SessionTimeout)
	fmt.Fprintf(w, "  mergeSimilarity:   %.3f\n", cfg.MergeSimilarity)
	fmt.Fprintf(w, "  terseTokenThreshold: %d\n", cfg.TerseTokenThreshold)
	fmt.Fprintf(w, "  sublinearTF:       %v\n", cfg.SublinearTF)
	fmt.Fprintf(w, "  typoTolerance.enabled:           %v\n", cfg.TypoTolerance.Enabled)
	fmt.Fprintf(w, "  typoTolerance.maxDistance:       %d\n", cfg.TypoTolerance.MaxDistance)
	fmt.Fprintf(w, "  typoTolerance.minWordLen:        %d\n", cfg.TypoTolerance.MinWordLen)
	fmt.Fprintf(w, "  typoTolerance.minEstablishedDF:  %d\n", cfg.TypoTolerance.MinEstablishedDF)
	fmt.Fprintf(w, "  memory.enabled:                  %v\n", cfg.Memory.Enabled)
	fmt.Fprintf(w, "  memory.dir:                      %s\n", cfg.Memory.Dir)
	fmt.Fprintf(w, "  memory.surfaceThreshold:         %.3f\n", cfg.Memory.SurfaceThreshold)
	fmt.Fprintf(w, "  memory.topK:                     %d\n", cfg.Memory.TopK)
	fmt.Fprintf(w, "  memory.maxBlockChars:            %d\n", cfg.Memory.MaxBlockChars)
	fmt.Fprintf(w, "  memory.minLeaves:                %d\n", cfg.Memory.MinLeaves)
	fmt.Fprintf(w, "  memory.minPrompts:               %d\n", cfg.Memory.MinPrompts)
	fmt.Fprintf(w, "  memory.promotionThreshold:       %.3f\n", cfg.Memory.PromotionThreshold)
	fmt.Fprintf(w, "  memory.rescueThreshold:          %.3f\n", cfg.Memory.RescueThreshold)
	fmt.Fprintf(w, "  memory.promotionCooldown:        %s\n", cfg.Memory.PromotionCooldown)
	fmt.Fprintf(w, "  memory.pendingMaxAge:            %s\n", cfg.Memory.PendingMaxAge)
	fmt.Fprintf(w, "  memory.mergeSuggestCosine:       %.3f\n", cfg.Memory.MergeSuggestCosine)
	fmt.Fprintf(w, "  memory.dedupCosine:              %.3f\n", cfg.Memory.DedupCosine)
	fmt.Fprintf(w, "  memory.autoNudge:                %v\n", cfg.Memory.AutoNudge)
	fmt.Fprintf(w, "  memory.commitRetries:            %d\n", cfg.Memory.CommitRetries)
	fmt.Fprintf(w, "  memory.weights.asset:            %.2f\n", cfg.Memory.Weights.Asset)
	fmt.Fprintf(w, "  memory.weights.topic:            %.2f\n", cfg.Memory.Weights.Topic)
	fmt.Fprintf(w, "  memory.weights.interest:         %.2f\n", cfg.Memory.Weights.Interest)
	fmt.Fprintf(w, "  memory.weights.fingerprint:      %.2f\n", cfg.Memory.Weights.Fingerprint)
	fmt.Fprintf(w, "  memory.frequencyBonus:           %.3f\n", cfg.Memory.FrequencyBonus)
	fmt.Fprintf(w, "  memory.redactPatterns:           %d configured\n", len(cfg.Memory.RedactPatterns))
	fmt.Fprintln(w)

	// --- Forest ---
	fmt.Fprintf(w, "--- Forest: %d trees, %d/%d nodes, %d prompts ---\n",
		len(f.Trees), f.NodeCount(), cfg.MemorySize, f.Meta.TotalPrompts)
	fmt.Fprintf(w, "  created:    %s\n", msToTime(f.Meta.Created))
	fmt.Fprintf(w, "  lastUpdate: %s\n", msToTime(f.Meta.LastUpdate))
	fmt.Fprintln(w)

	for i, tree := range f.Trees {
		root := tree.Root()
		if root == nil {
			continue
		}
		rootScore := root.Score(now, cfg.DecayRate)
		fmt.Fprintf(w, "  Tree #%d [id=%s] score=%.3f\n", i, tree.ID, rootScore)
		fmt.Fprintf(w, "    %d nodes, %d leaves, created %s\n",
			tree.NodeCount(), len(tree.GetLeaves()), msToTime(tree.Created))
		writeNodeTree(w, tree, tree.RootID, "    ", now, cfg.DecayRate, true)
		fmt.Fprintln(w)
	}

	// --- TF-IDF ---
	fmt.Fprintf(w, "--- TF-IDF: %d docs, %d unique terms ---\n", e.TotalDocs, len(e.DocFreq))
	top := topTermsByDF(e, 20)
	if len(top) > 0 {
		fmt.Fprintf(w, "  Top %d by document frequency:\n", len(top))
		for _, t := range top {
			fmt.Fprintf(w, "    %-20s df=%d\n", t.term, t.df)
		}
	}
	fmt.Fprintln(w)

	// --- Guide ---
	fmt.Fprintf(w, "--- Guide: %d/%d entries ---\n", len(g.Entries), g.MaxSize)
	for i, entry := range g.Entries {
		status := "pending"
		if entry.Reinforced {
			status = "reinforced"
		}
		summary := entry.Summary
		if len(summary) > 80 {
			summary = truncate(summary, 83)
		}
		treeName := resolveNodeTree(f, entry.IntentID)
		if treeName != "" {
			treeName = " (" + treeName + ")"
		}
		fmt.Fprintf(w, "  #%d [%-10s] -> %s%s  %s\n",
			i, status, entry.IntentID, treeName, msToTime(entry.Timestamp))
		fmt.Fprintf(w, "     %q\n", summary)
	}
	fmt.Fprintln(w)

	return nil
}

func dryRunText(result gate.DryRunResult, cfg config) error {
	w := os.Stdout

	fmt.Fprintln(w, "=== Focus Gate Dry Run ===")
	fmt.Fprintln(w)
	fmt.Fprintf(w, "Prompt: %q\n", result.Prompt)
	fmt.Fprintf(w, "Tokens: %v\n", result.Tokens)
	fmt.Fprintln(w)

	// Vector terms with weights
	if len(result.Vector) > 0 {
		fmt.Fprintf(w, "TF-IDF Vector (%d terms):\n", len(result.Vector))
		for _, v := range result.Vector {
			fmt.Fprintf(w, "  %-20s %.4f\n", v.Term, v.Weight)
		}
		fmt.Fprintln(w)
	}

	fmt.Fprintf(w, "Thresholds: extend >= %.3f, branch >= %.3f\n",
		cfg.Similarity.Extend, cfg.Similarity.Branch)
	fmt.Fprintln(w)

	// Per-tree scoring
	if len(result.TreeScores) > 0 {
		fmt.Fprintln(w, "Per-tree scoring:")
		for _, ts := range result.TreeScores {
			rootContent := ts.RootContent
			if len(rootContent) > 50 {
				rootContent = truncate(rootContent, 53)
			}
			fmt.Fprintf(w, "  Tree #%d %q\n", ts.TreeIdx, rootContent)
			fmt.Fprintf(w, "    Root %-14s  cosine=%.4f\n",
				ts.RootID, ts.RootCosine)

			for _, ls := range ts.LeafScores {
				leafContent := ls.Content
				if len(leafContent) > 50 {
					leafContent = truncate(leafContent, 53)
				}
				marker := ""
				if ls.LeafID == result.BestLeaf && result.BestTree == ts.TreeIdx {
					marker = "  <- BEST"
				}
				fmt.Fprintf(w, "    Leaf %-14s  cosine=%.4f  %q%s\n",
					ls.LeafID, ls.Cosine, leafContent, marker)
			}
			fmt.Fprintln(w)
		}
	} else {
		fmt.Fprintln(w, "  (no trees — forest is empty)")
		fmt.Fprintln(w)
	}

	// Final result
	fmt.Fprintf(w, "Result: %s (score=%.4f)\n", result.BestAction, result.BestScore)
	switch result.BestAction {
	case "new":
		fmt.Fprintln(w, "  Would create a new topic tree with this prompt.")
	case "branch":
		fmt.Fprintf(w, "  Would add as new subtopic under root of Tree #%d.\n", result.BestTree)
	case "extend":
		fmt.Fprintf(w, "  Would add as sibling near leaf %s in Tree #%d.\n", result.BestLeaf, result.BestTree)

	}

	if result.NearMiss {
		fmt.Fprintln(w)
		fmt.Fprintf(w, "Near-miss: top score %.4f fell short of branch threshold %.3f.\n",
			result.BestScore, result.NearMissThreshold)
		fmt.Fprintln(w, "  A typo, missing stem, or rare-term mismatch may have prevented a match")
		fmt.Fprintln(w, "  you were expecting. Check /focus tree N to see the tree's vector terms.")
		return nil
	}

	return nil
}

// ---------------------------------------------------------------------------
// JSON formatters
// ---------------------------------------------------------------------------

// JSON wrapper types give a clean, fully-typed JSON output. Computed fields
// like node scores and transition probabilities are included so consumers
// don't need to re-derive them.

type jsonInspect struct {
	Config config     `json:"config"`
	Forest jsonForest `json:"forest"`
	TFIDF  jsonTFIDF  `json:"tfidf"`
	Guide  jsonGuide  `json:"guide"`
}

type jsonForest struct {
	TotalPrompts int        `json:"totalPrompts"`
	NodeCount    int        `json:"nodeCount"`
	MemorySize   int        `json:"memorySize"`
	TreeCount    int        `json:"treeCount"`
	Created      int64      `json:"created"`
	LastUpdate   int64      `json:"lastUpdate"`
	Trees        []jsonTree `json:"trees"`
}

type jsonTree struct {
	ID           string   `json:"id"`
	RootID       string   `json:"rootId"`
	NodeCount    int      `json:"nodeCount"`
	LeafCount    int      `json:"leafCount"`
	RootScore    float64  `json:"rootScore"`
	Created      int64    `json:"created"`
	LastAccessed int64    `json:"lastAccessed"`
	Root         jsonNode `json:"root"`
}

type jsonNode struct {
	ID           string     `json:"id"`
	Content      string     `json:"content"`
	Depth        int        `json:"depth"`
	Weight       float64    `json:"weight"`
	Frequency    int        `json:"frequency"`
	Indexed      bool       `json:"indexed"`
	Score        float64    `json:"score"`
	Created      int64      `json:"created"`
	LastAccessed int64      `json:"lastAccessed"`
	Children     []jsonNode `json:"children,omitempty"`
}

type jsonTFIDF struct {
	TotalDocs   int        `json:"totalDocs"`
	UniqueTerms int        `json:"uniqueTerms"`
	TopTerms    []jsonTerm `json:"topTerms"`
}

type jsonTerm struct {
	Term string `json:"term"`
	DF   int    `json:"df"`
}

type jsonGuide struct {
	Count   int              `json:"count"`
	MaxSize int              `json:"maxSize"`
	Entries []jsonGuideEntry `json:"entries"`
}

type jsonGuideEntry struct {
	Summary    string `json:"summary"`
	IntentID   string `json:"intentId"`
	Reinforced bool   `json:"reinforced"`
	Timestamp  int64  `json:"timestamp"`
}

func inspectJSON(f *forest.Forest, e *tfidf.Engine, g *guide.Guide, cfg config) error {
	now := time.Now().UnixMilli()

	// Build forest tree structures
	trees := make([]jsonTree, 0, len(f.Trees))
	for _, tree := range f.Trees {
		root := tree.Root()
		if root == nil {
			continue
		}
		trees = append(trees, jsonTree{
			ID:           tree.ID,
			RootID:       tree.RootID,
			NodeCount:    tree.NodeCount(),
			LeafCount:    len(tree.GetLeaves()),
			RootScore:    root.Score(now, cfg.DecayRate),
			Created:      tree.Created,
			LastAccessed: tree.LastAccessed,
			Root:         buildNodeJSON(tree, tree.RootID, now, cfg.DecayRate),
		})
	}

	// Build top TF-IDF terms
	top := topTermsByDF(e, 30)
	jsonTerms := make([]jsonTerm, len(top))
	for i, t := range top {
		jsonTerms[i] = jsonTerm{Term: t.term, DF: t.df}
	}

	// Build guide entries
	guideEntries := make([]jsonGuideEntry, len(g.Entries))
	for i, entry := range g.Entries {
		guideEntries[i] = jsonGuideEntry{
			Summary:    entry.Summary,
			IntentID:   entry.IntentID,
			Reinforced: entry.Reinforced,
			Timestamp:  entry.Timestamp,
		}
	}

	result := jsonInspect{
		Config: cfg,
		Forest: jsonForest{
			TotalPrompts: f.Meta.TotalPrompts,
			NodeCount:    f.NodeCount(),
			MemorySize:   cfg.MemorySize,
			TreeCount:    len(f.Trees),
			Created:      f.Meta.Created,
			LastUpdate:   f.Meta.LastUpdate,
			Trees:        trees,
		},
		TFIDF: jsonTFIDF{
			TotalDocs:   e.TotalDocs,
			UniqueTerms: len(e.DocFreq),
			TopTerms:    jsonTerms,
		},
		Guide: jsonGuide{
			Count:   len(g.Entries),
			MaxSize: g.MaxSize,
			Entries: guideEntries,
		},
	}

	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal inspect: %w", err)
	}
	fmt.Fprintln(os.Stdout, string(data))
	return nil
}

func dryRunJSON(result gate.DryRunResult) error {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal dry-run: %w", err)
	}
	fmt.Fprintln(os.Stdout, string(data))
	return nil
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// writeNodeTree recursively prints a tree's node hierarchy with box-drawing
// connectors. isRoot controls whether the node metadata is printed (children
// are always printed by their parent's iteration).
func writeNodeTree(w io.Writer, tree *forest.Tree, nodeID string, prefix string, now int64, decayRate float64, isRoot bool) {
	node := tree.Nodes[nodeID]
	if node == nil {
		return
	}

	score := node.Score(now, decayRate)
	idx := "-"
	if node.Indexed {
		idx = "Y"
	}
	content := node.Content
	if len(content) > 70 {
		content = truncate(content, 73)
	}

	if isRoot {
		fmt.Fprintf(w, "%s[root] %s  d=%d w=%.2f f=%d idx=%s s=%.3f\n",
			prefix, node.ID, node.Depth, node.Weight, node.Frequency, idx, score)
		fmt.Fprintf(w, "%s%q\n", prefix, content)
	}

	for i, childID := range node.ChildIDs {
		child := tree.Nodes[childID]
		if child == nil {
			continue
		}
		last := i == len(node.ChildIDs)-1

		connector := "├── "
		extension := "│   "
		if last {
			connector = "└── "
			extension = "    "
		}

		cScore := child.Score(now, decayRate)
		cIdx := "-"
		if child.Indexed {
			cIdx = "Y"
		}
		cContent := child.Content
		if len(cContent) > 70 {
			cContent = truncate(cContent, 73)
		}

		fmt.Fprintf(w, "%s%s%s  d=%d w=%.2f f=%d idx=%s s=%.3f\n",
			prefix, connector, child.ID, child.Depth, child.Weight, child.Frequency, cIdx, cScore)
		fmt.Fprintf(w, "%s%s%q\n", prefix, extension, cContent)

		// Recurse into grandchildren with updated prefix.
		writeNodeTree(w, tree, childID, prefix+extension, now, decayRate, false)
	}
}

// buildNodeJSON recursively builds a JSON-friendly node hierarchy.
func buildNodeJSON(tree *forest.Tree, nodeID string, now int64, decayRate float64) jsonNode {
	node := tree.Nodes[nodeID]
	if node == nil {
		return jsonNode{}
	}

	jn := jsonNode{
		ID:           node.ID,
		Content:      node.Content,
		Depth:        node.Depth,
		Weight:       node.Weight,
		Frequency:    node.Frequency,
		Indexed:      node.Indexed,
		Score:        node.Score(now, decayRate),
		Created:      node.Created,
		LastAccessed: node.LastAccessed,
	}

	for _, childID := range node.ChildIDs {
		jn.Children = append(jn.Children, buildNodeJSON(tree, childID, now, decayRate))
	}

	return jn
}

// termDF is a helper for sorting terms by document frequency.
type termDF struct {
	term string
	df   int
}

// topTermsByDF returns the top n terms from the TF-IDF engine sorted by DF descending.
func topTermsByDF(e *tfidf.Engine, n int) []termDF {
	terms := make([]termDF, 0, len(e.DocFreq))
	for t, df := range e.DocFreq {
		terms = append(terms, termDF{t, df})
	}
	sort.Slice(terms, func(i, j int) bool {
		if terms[i].df != terms[j].df {
			return terms[i].df > terms[j].df
		}
		return terms[i].term < terms[j].term
	})
	if n > len(terms) {
		n = len(terms)
	}
	return terms[:n]
}

// msToTime formats a Unix-millisecond timestamp as a human-readable string.
// Returns "(none)" for zero timestamps.
func msToTime(ms int64) string {
	if ms == 0 {
		return "(none)"
	}
	return time.UnixMilli(ms).Format("2006-01-02 15:04:05")
}

// treeNameByID returns the truncated root content for a tree ID, or "".
func treeNameByID(f *forest.Forest, treeID string) string {
	for _, tree := range f.Trees {
		if tree.ID == treeID {
			root := tree.Root()
			if root != nil {
				name := root.Content
				if len(name) > 40 {
					name = truncate(name, 43)
				}
				return name
			}
		}
	}
	return ""
}

// resolveNodeTree finds which tree contains a given node ID and returns
// the tree's root content as a short label.
func resolveNodeTree(f *forest.Forest, nodeID string) string {
	if nodeID == "" {
		return ""
	}
	for _, tree := range f.Trees {
		if _, ok := tree.Nodes[nodeID]; ok {
			root := tree.Root()
			if root != nil {
				name := root.Content
				if len(name) > 30 {
					name = truncate(name, 33)
				}
				return name
			}
		}
	}
	return ""
}
