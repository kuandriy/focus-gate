package main

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/kuandriy/focus-gate/internal/forest"
	"github.com/kuandriy/focus-gate/internal/gate"
	"github.com/kuandriy/focus-gate/internal/guide"
	"github.com/kuandriy/focus-gate/internal/markov"
	"github.com/kuandriy/focus-gate/internal/persist"
	"github.com/kuandriy/focus-gate/internal/text"
	"github.com/kuandriy/focus-gate/internal/tfidf"
)

// slashCommand holds the parsed /focus subcommand and its argument (if any).
type slashCommand struct {
	sub string // "inspect", "status", "tree", "terms", "markov", "score", "health", "help"
	arg string // e.g. tree index/ID, or prompt text for score
}

// parseSlashCommand checks whether the raw (uncleaned) prompt starts with
// "/focus" and extracts the subcommand and optional argument.
// Returns (cmd, true) if matched, (zero, false) otherwise.
func parseSlashCommand(raw string) (slashCommand, bool) {
	trimmed := strings.TrimSpace(raw)
	lower := strings.ToLower(trimmed)

	if !strings.HasPrefix(lower, "/focus") {
		return slashCommand{}, false
	}

	// Ensure "/focus" is the complete word — not a prefix of "/focusgate" etc.
	if len(lower) > len("/focus") && lower[len("/focus")] != ' ' {
		return slashCommand{}, false
	}

	// Strip the "/focus" prefix and parse the rest.
	rest := strings.TrimSpace(trimmed[len("/focus"):])
	if rest == "" {
		return slashCommand{sub: "help"}, true
	}

	// Split into subcommand and argument.
	parts := strings.SplitN(rest, " ", 2)
	sub := strings.ToLower(parts[0])
	arg := ""
	if len(parts) > 1 {
		arg = strings.TrimSpace(parts[1])
	}

	return slashCommand{sub: sub, arg: arg}, true
}

// handleSlashCommand dispatches a parsed slash command to the appropriate handler.
// All commands load state read-only — no mutations, no saves.
func handleSlashCommand(cmd slashCommand, p paths, cfg config) error {
	// Load all state (read-only)
	f := forest.NewForest()
	logLoadErr("intent", persist.Load(p.intentFile, f))

	e := tfidf.NewEngine()
	logLoadErr("engine", persist.Load(p.engineFile, e))

	g := guide.New(cfg.GuideSize)
	logLoadErr("guide", persist.Load(p.guideFile, g))

	c := markov.New()
	logLoadErr("markov", persist.Load(p.markovFile, c))

	w := os.Stdout

	switch cmd.sub {
	case "inspect":
		return inspectText(f, e, g, c, cfg)

	case "status":
		return slashStatus(w, f, e, g, c, cfg)

	case "tree":
		return slashTree(w, f, e, cfg, cmd.arg)

	case "terms":
		return slashTerms(w, e, cmd.arg)

	case "markov":
		return slashMarkov(w, f, c)

	case "score":
		return slashScore(w, f, e, c, cfg, cmd.arg)

	case "health":
		return slashHealth(w, f, e, c, cfg)

	case "help":
		return slashHelp(w)

	default:
		fmt.Fprintf(w, "[Focus] Unknown command: /focus %s\n", cmd.sub)
		return slashHelp(w)
	}
}

// ---------------------------------------------------------------------------
// /focus status — compact summary (same as --status but in chat)
// ---------------------------------------------------------------------------
func slashStatus(w *os.File, f *forest.Forest, e *tfidf.Engine, g *guide.Guide, c *markov.Chain, cfg config) error {
	gateCfg := toGateConfig(cfg)
	gt := gate.NewWithChain(f, e, c, gateCfg)
	ctx := gt.GenerateContext()
	if ctx != "" {
		fmt.Fprint(w, ctx)
	} else {
		fmt.Fprintf(w, "[Focus | %d prompts | %d/%d mem | %d trees]\n[/Focus]\n",
			f.Meta.TotalPrompts, f.NodeCount(), cfg.MemorySize, len(f.Trees))
	}

	guideCtx := g.Render(f)
	if guideCtx != "" {
		fmt.Fprint(w, guideCtx)
	}
	return nil
}

// ---------------------------------------------------------------------------
// /focus tree <index|id> — single tree deep-dive
// ---------------------------------------------------------------------------
func slashTree(w *os.File, f *forest.Forest, e *tfidf.Engine, cfg config, arg string) error {
	if len(f.Trees) == 0 {
		fmt.Fprintln(w, "[Focus] No trees in forest.")
		return nil
	}

	if arg == "" {
		// List all trees briefly so user can pick one.
		fmt.Fprintln(w, "[Focus] Trees:")
		now := time.Now().UnixMilli()
		for i, tree := range f.Trees {
			root := tree.Root()
			if root == nil {
				continue
			}
			score := root.Score(now, cfg.DecayRate)
			content := root.Content
			if len(content) > 60 {
				content = content[:60] + "..."
			}
			fmt.Fprintf(w, "  #%d [%s] score=%.3f  %d nodes  %q\n",
				i, tree.ID[:8], score, tree.NodeCount(), content)
		}
		fmt.Fprintln(w, "\nUsage: /focus tree <number>")
		return nil
	}

	// Find tree by index or partial ID.
	tree := findTree(f, arg)
	if tree == nil {
		fmt.Fprintf(w, "[Focus] Tree not found: %s\n", arg)
		return nil
	}

	root := tree.Root()
	if root == nil {
		fmt.Fprintln(w, "[Focus] Tree has no root.")
		return nil
	}

	now := time.Now().UnixMilli()

	// Tree header
	fmt.Fprintf(w, "=== Tree: %s ===\n", tree.ID)
	fmt.Fprintf(w, "  Nodes: %d, Leaves: %d\n", tree.NodeCount(), len(tree.GetLeaves()))
	fmt.Fprintf(w, "  Created:  %s\n", msToTime(tree.Created))
	fmt.Fprintf(w, "  Accessed: %s\n", msToTime(tree.LastAccessed))
	fmt.Fprintf(w, "  Root score: %.3f\n", root.Score(now, cfg.DecayRate))
	fmt.Fprintln(w)

	// Full node hierarchy
	writeNodeTree(w, tree, tree.RootID, "  ", now, cfg.DecayRate, true)
	fmt.Fprintln(w)

	// Show the TF-IDF terms that define the root vector
	fmt.Fprintln(w, "  Root vector terms:")
	rootVec := e.Vectorize(root.Content)
	if len(rootVec) == 0 {
		fmt.Fprintln(w, "    (empty — root content not vectorized)")
	} else {
		for _, t := range rootVec {
			fmt.Fprintf(w, "    %-20s %.4f\n", t.Word, t.Weight)
		}
	}
	fmt.Fprintln(w)

	// Show leaf vectors
	leaves := tree.GetLeaves()
	if len(leaves) > 0 {
		fmt.Fprintln(w, "  Leaf vectors:")
		for _, leaf := range leaves {
			if leaf.ID == tree.RootID {
				continue
			}
			content := leaf.Content
			if len(content) > 50 {
				content = content[:50] + "..."
			}
			leafScore := leaf.Score(now, cfg.DecayRate)
			fmt.Fprintf(w, "    %s (score=%.3f) %q\n", leaf.ID[:8], leafScore, content)
			leafVec := e.Vectorize(leaf.Content)
			for _, t := range leafVec {
				fmt.Fprintf(w, "      %-18s %.4f\n", t.Word, t.Weight)
			}
		}
	}

	// Pruning candidates (lowest-scoring leaves)
	if len(leaves) > 1 {
		sort.Slice(leaves, func(i, j int) bool {
			return leaves[i].Score(now, cfg.DecayRate) < leaves[j].Score(now, cfg.DecayRate)
		})
		fmt.Fprintln(w)
		fmt.Fprintln(w, "  Pruning candidates (lowest score first):")
		limit := 3
		if limit > len(leaves) {
			limit = len(leaves)
		}
		for _, leaf := range leaves[:limit] {
			if leaf.ID == tree.RootID {
				continue
			}
			content := leaf.Content
			if len(content) > 50 {
				content = content[:50] + "..."
			}
			fmt.Fprintf(w, "    [PRUNE?] %s  score=%.3f  %q\n",
				leaf.ID[:8], leaf.Score(now, cfg.DecayRate), content)
		}
	}

	return nil
}

// ---------------------------------------------------------------------------
// /focus terms [N] — TF-IDF vocabulary with IDF values
// ---------------------------------------------------------------------------
func slashTerms(w *os.File, e *tfidf.Engine, arg string) error {
	n := 30
	if arg != "" {
		if parsed, err := strconv.Atoi(arg); err == nil && parsed > 0 {
			n = parsed
		}
	}

	top := topTermsByDF(e, n)
	if len(top) == 0 {
		fmt.Fprintln(w, "[Focus] No terms in TF-IDF engine.")
		return nil
	}

	fmt.Fprintf(w, "=== TF-IDF Vocabulary: %d docs, %d unique terms ===\n",
		e.TotalDocs, len(e.DocFreq))
	fmt.Fprintf(w, "  Showing top %d by document frequency:\n\n", len(top))
	fmt.Fprintf(w, "  %-22s %4s  %8s\n", "TERM", "DF", "IDF")
	fmt.Fprintf(w, "  %-22s %4s  %8s\n", "----", "--", "---")
	for _, t := range top {
		idf := e.IDF(t.term)
		fmt.Fprintf(w, "  %-22s %4d  %8.4f\n", t.term, t.df, idf)
	}
	return nil
}

// ---------------------------------------------------------------------------
// /focus markov — transition matrix
// ---------------------------------------------------------------------------
func slashMarkov(w *os.File, f *forest.Forest, c *markov.Chain) error {
	if len(c.Counts) == 0 {
		fmt.Fprintln(w, "[Focus] No Markov transitions recorded yet.")
		return nil
	}

	fmt.Fprintln(w, "=== Markov Transition Matrix ===")

	if c.LastTopic != "" {
		name := treeNameByID(f, c.LastTopic)
		fmt.Fprintf(w, "  Last topic: %s", c.LastTopic[:8])
		if name != "" {
			fmt.Fprintf(w, " (%s)", name)
		}
		fmt.Fprintln(w)
	}
	fmt.Fprintln(w)

	// Sort sources for deterministic output.
	froms := make([]string, 0, len(c.Counts))
	for from := range c.Counts {
		froms = append(froms, from)
	}
	sort.Strings(froms)

	for _, from := range froms {
		row := c.Counts[from]
		total := c.Totals[from]
		name := treeNameByID(f, from)
		label := from[:8]
		if name != "" {
			label += " (" + name + ")"
		}
		fmt.Fprintf(w, "  %s ->\n", label)

		type dest struct {
			to    string
			count int
		}
		dests := make([]dest, 0, len(row))
		for to, count := range row {
			dests = append(dests, dest{to, count})
		}
		sort.Slice(dests, func(i, j int) bool { return dests[i].count > dests[j].count })

		for _, d := range dests {
			prob := float64(d.count) / float64(total) * 100
			dName := treeNameByID(f, d.to)
			dLabel := d.to[:8]
			if dName != "" {
				dLabel += " (" + dName + ")"
			}
			fmt.Fprintf(w, "    %s: %d/%d (%.0f%%)\n", dLabel, d.count, total, prob)
		}
	}

	return nil
}

// ---------------------------------------------------------------------------
// /focus score "prompt" — dry-run scoring from chat
// ---------------------------------------------------------------------------
func slashScore(w *os.File, f *forest.Forest, e *tfidf.Engine, c *markov.Chain, cfg config, arg string) error {
	if arg == "" {
		fmt.Fprintln(w, "[Focus] Usage: /focus score <prompt text>")
		return nil
	}

	prompt := text.CleanPrompt(arg)
	if prompt == "" {
		fmt.Fprintln(w, "[Focus] Prompt is empty after cleaning.")
		return nil
	}

	gt := gate.NewWithChain(f, e, c, toGateConfig(cfg))
	result := gt.DryRun(prompt)

	fmt.Fprintln(w, "=== Score ===")
	fmt.Fprintf(w, "  Prompt: %q\n", result.Prompt)
	fmt.Fprintf(w, "  Tokens: %v\n\n", result.Tokens)

	if len(result.Vector) > 0 {
		fmt.Fprintf(w, "  TF-IDF Vector (%d terms):\n", len(result.Vector))
		for _, v := range result.Vector {
			fmt.Fprintf(w, "    %-20s %.4f\n", v.Term, v.Weight)
		}
		fmt.Fprintln(w)
	}

	fmt.Fprintf(w, "  Thresholds: extend >= %.3f, branch >= %.3f\n\n", cfg.Similarity.Extend, cfg.Similarity.Branch)

	for _, ts := range result.TreeScores {
		rootContent := ts.RootContent
		if len(rootContent) > 50 {
			rootContent = rootContent[:50] + "..."
		}
		fmt.Fprintf(w, "  Tree #%d %q  [boost=%.3f]\n", ts.TreeIdx, rootContent, ts.BoostFactor)
		fmt.Fprintf(w, "    Root %-14s  cosine=%.4f  boosted=%.4f\n",
			ts.RootID[:8], ts.RootCosine, ts.RootBoosted)

		for _, ls := range ts.LeafScores {
			leafContent := ls.Content
			if len(leafContent) > 50 {
				leafContent = leafContent[:50] + "..."
			}
			marker := ""
			if ls.LeafID == result.BestLeaf && result.BestTree == ts.TreeIdx {
				marker = "  <- BEST"
			}
			fmt.Fprintf(w, "    Leaf %-14s  cosine=%.4f  boosted=%.4f  %q%s\n",
				ls.LeafID[:8], ls.Cosine, ls.Boosted, leafContent, marker)
		}
		fmt.Fprintln(w)
	}

	fmt.Fprintf(w, "  Result: %s (score=%.4f)\n", result.BestAction, result.BestScore)

	return nil
}

// ---------------------------------------------------------------------------
// /focus health — system diagnostics
// ---------------------------------------------------------------------------
func slashHealth(w *os.File, f *forest.Forest, e *tfidf.Engine, c *markov.Chain, cfg config) error {
	now := time.Now().UnixMilli()

	fmt.Fprintln(w, "=== Focus Health ===")
	fmt.Fprintln(w)

	// --- Memory pressure ---
	nodeCount := f.NodeCount()
	pct := 0
	if cfg.MemorySize > 0 {
		pct = nodeCount * 100 / cfg.MemorySize
	}
	bar := memoryBar(pct)
	fmt.Fprintf(w, "  Memory:  %d/%d nodes (%d%%) %s\n", nodeCount, cfg.MemorySize, pct, bar)

	// --- Tree balance ---
	if len(f.Trees) > 0 {
		minN, maxN, totalN := math.MaxInt, 0, 0
		minD, maxD := math.MaxInt, 0
		for _, tree := range f.Trees {
			nc := tree.NodeCount()
			if nc < minN {
				minN = nc
			}
			if nc > maxN {
				maxN = nc
			}
			totalN += nc

			for _, node := range tree.Nodes {
				if node.Depth < minD {
					minD = node.Depth
				}
				if node.Depth > maxD {
					maxD = node.Depth
				}
			}
		}
		avg := float64(totalN) / float64(len(f.Trees))
		fmt.Fprintf(w, "  Trees:   %d (nodes per tree: min=%d avg=%.1f max=%d, max depth=%d)\n",
			len(f.Trees), minN, avg, maxN, maxD)
	} else {
		fmt.Fprintln(w, "  Trees:   0")
	}
	fmt.Fprintf(w, "  Prompts: %d\n", f.Meta.TotalPrompts)
	fmt.Fprintln(w)

	// --- Term diversity ---
	noiseCount := 0
	for _, df := range e.DocFreq {
		if df == 1 {
			noiseCount++
		}
	}
	fmt.Fprintf(w, "  TF-IDF:  %d docs, %d unique terms\n", e.TotalDocs, len(e.DocFreq))
	if len(e.DocFreq) > 0 {
		fmt.Fprintf(w, "           %d terms with df=1 (noise: %.0f%%)\n",
			noiseCount, float64(noiseCount)/float64(len(e.DocFreq))*100)
	}
	fmt.Fprintln(w)

	// --- Staleness ---
	if len(f.Trees) > 0 {
		fmt.Fprintln(w, "  Tree activity:")
		type treeAge struct {
			idx   int
			name  string
			score float64
			age   string
			tag   string
		}
		var entries []treeAge
		for i, tree := range f.Trees {
			root := tree.Root()
			if root == nil {
				continue
			}
			score := root.Score(now, cfg.DecayRate)
			ageHours := float64(now-tree.LastAccessed) / 3600000.0
			name := root.Content
			if len(name) > 40 {
				name = name[:40] + "..."
			}

			tag := "[HOT]"
			if ageHours > 24 {
				tag = "[COLD]"
			} else if ageHours > 4 {
				tag = "[WARM]"
			}

			entries = append(entries, treeAge{
				idx:   i,
				name:  name,
				score: score,
				age:   formatAge(ageHours),
				tag:   tag,
			})
		}

		// Sort by score descending
		sort.Slice(entries, func(i, j int) bool { return entries[i].score > entries[j].score })
		for _, e := range entries {
			fmt.Fprintf(w, "    #%d %-6s score=%.3f  age=%s  %q\n",
				e.idx, e.tag, e.score, e.age, e.name)
		}
		fmt.Fprintln(w)
	}

	// --- Pruning forecast ---
	if nodeCount > 0 && len(f.Trees) > 0 {
		fmt.Fprintln(w, "  Pruning forecast (lowest-scoring leaves):")
		type candidate struct {
			treeIdx int
			nodeID  string
			content string
			score   float64
		}
		var candidates []candidate
		for i, tree := range f.Trees {
			for _, leaf := range tree.GetLeaves() {
				if leaf.ID == tree.RootID {
					continue
				}
				content := leaf.Content
				if len(content) > 40 {
					content = content[:40] + "..."
				}
				candidates = append(candidates, candidate{
					treeIdx: i,
					nodeID:  leaf.ID,
					content: content,
					score:   leaf.Score(now, cfg.DecayRate),
				})
			}
		}
		sort.Slice(candidates, func(i, j int) bool { return candidates[i].score < candidates[j].score })
		limit := 5
		if limit > len(candidates) {
			limit = len(candidates)
		}
		if limit == 0 {
			fmt.Fprintln(w, "    (no non-root leaves to prune)")
		}
		for _, c := range candidates[:limit] {
			fmt.Fprintf(w, "    [PRUNE?] tree#%d %s  score=%.4f  %q\n",
				c.treeIdx, c.nodeID[:8], c.score, c.content)
		}
		slotsLeft := cfg.MemorySize - nodeCount
		if slotsLeft > 0 {
			fmt.Fprintf(w, "\n    %d slots remaining before pruning triggers.\n", slotsLeft)
		} else {
			fmt.Fprintf(w, "\n    Memory full — pruning will trigger on next prompt.\n")
		}
	}

	// --- Markov ---
	fmt.Fprintln(w)
	fmt.Fprintf(w, "  Markov:  %d topics tracked", len(c.Counts))
	if c.LastTopic != "" {
		name := treeNameByID(f, c.LastTopic)
		if name != "" {
			fmt.Fprintf(w, ", last=%s (%s)", c.LastTopic[:8], name)
		} else {
			fmt.Fprintf(w, ", last=%s", c.LastTopic[:8])
		}
	}
	fmt.Fprintln(w)

	return nil
}

// ---------------------------------------------------------------------------
// /focus help — list available commands
// ---------------------------------------------------------------------------
func slashHelp(w *os.File) error {
	fmt.Fprintln(w, "=== Focus Gate Commands ===")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "  /focus status          Compact context summary")
	fmt.Fprintln(w, "  /focus inspect         Full state dump (forest, TF-IDF, guide, Markov)")
	fmt.Fprintln(w, "  /focus tree [N]        Deep-dive into tree #N (list trees if N omitted)")
	fmt.Fprintln(w, "  /focus terms [N]       TF-IDF vocabulary with IDF values (default: top 30)")
	fmt.Fprintln(w, "  /focus markov          Transition matrix")
	fmt.Fprintln(w, "  /focus score \"prompt\"  Dry-run classification scoring")
	fmt.Fprintln(w, "  /focus health          System diagnostics and pruning forecast")
	fmt.Fprintln(w, "  /focus help            This help message")
	return nil
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// findTree locates a tree by numeric index or partial ID match.
func findTree(f *forest.Forest, arg string) *forest.Tree {
	// Try as numeric index first.
	if idx, err := strconv.Atoi(arg); err == nil && idx >= 0 && idx < len(f.Trees) {
		return f.Trees[idx]
	}
	// Try as partial ID match.
	lower := strings.ToLower(arg)
	for _, tree := range f.Trees {
		if strings.HasPrefix(strings.ToLower(tree.ID), lower) {
			return tree
		}
	}
	return nil
}

// memoryBar returns a visual bar like [████████░░░░] for a percentage.
func memoryBar(pct int) string {
	const width = 12
	filled := pct * width / 100
	if filled > width {
		filled = width
	}
	return "[" + strings.Repeat("█", filled) + strings.Repeat("░", width-filled) + "]"
}

// formatAge formats hours into a human-readable age string.
func formatAge(hours float64) string {
	if hours < 1 {
		return fmt.Sprintf("%.0fm", hours*60)
	}
	if hours < 24 {
		return fmt.Sprintf("%.1fh", hours)
	}
	return fmt.Sprintf("%.1fd", hours/24)
}
