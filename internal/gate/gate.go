package gate

import (
	"fmt"
	"sort"
	"strings"

	"github.com/kuandriy/focus-gate/internal/forest"
	"github.com/kuandriy/focus-gate/internal/markov"
	"github.com/kuandriy/focus-gate/internal/text"
	"github.com/kuandriy/focus-gate/internal/tfidf"
)

// Config holds gate classification parameters.
type Config struct {
	ExtendThreshold   float64 `json:"extend"`
	BranchThreshold   float64 `json:"branch"`
	BubbleUpTerms     int     `json:"bubbleUpTerms"`
	MaxSourcesPerNode int     `json:"maxSourcesPerNode"`
	MemorySize        int     `json:"memorySize"`
	DecayRate         float64 `json:"decayRate"`
	ContextLimit      int     `json:"contextLimit"`
	TransitionBoost   float64 `json:"transitionBoost"`
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{
		ExtendThreshold:   0.55,
		BranchThreshold:   0.25,
		BubbleUpTerms:     6,
		MaxSourcesPerNode: 20,
		MemorySize:        100,
		DecayRate:         0.05,
		ContextLimit:      600,
		TransitionBoost:   0.2,
	}
}

// Action describes how a prompt was classified.
type Action int

const (
	ActionNew    Action = iota // Unrelated — start a new topic tree
	ActionBranch               // Broadly related — add under root
	ActionExtend               // Closely related — add near matching leaf
)

func (a Action) String() string {
	switch a {
	case ActionNew:
		return "new"
	case ActionBranch:
		return "branch"
	case ActionExtend:
		return "extend"
	}
	return "unknown"
}

// Classification holds the result of classifying a prompt against the forest.
type Classification struct {
	Action  Action
	TreeIdx int
	LeafID  string  // For extend: the matching leaf
	Score   float64
}

// Gate is the Focus Gate classifier. It classifies prompts, mutates the forest,
// and generates context output.
type Gate struct {
	Forest *forest.Forest
	Engine *tfidf.Engine
	Chain  *markov.Chain
	Config Config
}

// New creates a Gate from existing forest and engine state.
func New(f *forest.Forest, e *tfidf.Engine, cfg Config) *Gate {
	return &Gate{Forest: f, Engine: e, Chain: markov.New(), Config: cfg}
}

// NewWithChain creates a Gate with an existing Markov chain.
func NewWithChain(f *forest.Forest, e *tfidf.Engine, c *markov.Chain, cfg Config) *Gate {
	return &Gate{Forest: f, Engine: e, Chain: c, Config: cfg}
}

// ProcessPrompt classifies a prompt, applies it to the forest, and returns context.
func (g *Gate) ProcessPrompt(prompt string, source string) string {
	tokens := text.Tokenize(prompt)
	if len(tokens) == 0 {
		return ""
	}

	vec := g.Engine.VectorizeTokens(tokens)

	cls := g.classify(vec)
	g.apply(cls, prompt, source, tokens)

	// Determine the tree ID that this prompt was classified into
	currentTreeID := ""
	if len(g.Forest.Trees) > 0 {
		if cls.Action == ActionNew {
			// New tree was just appended
			currentTreeID = g.Forest.Trees[len(g.Forest.Trees)-1].ID
		} else {
			currentTreeID = g.Forest.Trees[cls.TreeIdx].ID
		}
	}

	// Record Markov transition
	g.Chain.Record(g.Chain.LastTopic, currentTreeID)
	g.Chain.LastTopic = currentTreeID

	g.Forest.Meta.TotalPrompts++
	g.Forest.Meta.LastUpdate = g.Forest.Trees[len(g.Forest.Trees)-1].LastAccessed

	// Add the new prompt to the TF-IDF corpus
	g.Engine.AddDocument(tokens)

	// Prune if needed — track which trees existed before pruning
	if g.Forest.NodeCount() > g.Config.MemorySize {
		treeIDs := make(map[string]bool, len(g.Forest.Trees))
		for _, t := range g.Forest.Trees {
			treeIDs[t.ID] = true
		}

		removed := g.Forest.Prune(g.Config.MemorySize, g.Config.DecayRate)
		for _, content := range removed {
			g.Engine.RemoveDocument(text.Tokenize(content))
		}

		// Sync Markov chain: prune topics for trees that were removed
		for id := range treeIDs {
			found := false
			for _, t := range g.Forest.Trees {
				if t.ID == id {
					found = true
					break
				}
			}
			if !found {
				g.Chain.PruneTopic(id)
			}
		}
	}

	return g.GenerateContext()
}

// classify compares the prompt vector against all roots and leaves.
// Adds a Markov transition boost per tree when transition data exists.
func (g *Gate) classify(vec tfidf.Vector) Classification {
	if len(g.Forest.Trees) == 0 || vec == nil {
		return Classification{Action: ActionNew, Score: 0}
	}

	best := Classification{Action: ActionNew, Score: 0}
	alpha := g.Config.TransitionBoost

	for i, tree := range g.Forest.Trees {
		root := tree.Root()
		if root == nil {
			continue
		}

		// Markov boost: P(this_tree | last_topic)
		boost := 0.0
		if alpha > 0 && g.Chain.LastTopic != "" {
			boost = alpha * g.Chain.Probability(g.Chain.LastTopic, tree.ID)
		}

		// Compare against root
		rootVec := g.Engine.Vectorize(root.Content)
		rootSim := tfidf.CosineSimilarity(vec, rootVec) + boost
		if rootSim > best.Score {
			best.Score = rootSim
			best.TreeIdx = i
			best.LeafID = ""
		}

		// Compare against each leaf
		for _, leaf := range tree.GetLeaves() {
			leafVec := g.Engine.Vectorize(leaf.Content)
			leafSim := tfidf.CosineSimilarity(vec, leafVec) + boost
			if leafSim > best.Score {
				best.Score = leafSim
				best.TreeIdx = i
				best.LeafID = leaf.ID
			}
		}
	}

	if best.Score >= g.Config.ExtendThreshold {
		best.Action = ActionExtend
	} else if best.Score >= g.Config.BranchThreshold {
		best.Action = ActionBranch
	} else {
		best.Action = ActionNew
	}

	return best
}

// apply mutates the forest based on the classification.
func (g *Gate) apply(cls Classification, content string, source string, tokens []string) {
	switch cls.Action {
	case ActionNew:
		tree := forest.NewTree(content, source)
		g.Forest.AddTree(tree)

	case ActionBranch:
		tree := g.Forest.Trees[cls.TreeIdx]
		g.preserveRoot(tree)
		tree.AddChild(tree.RootID, content, source)
		g.bubbleUp(tree, tree.RootID)

	case ActionExtend:
		tree := g.Forest.Trees[cls.TreeIdx]
		leaf := tree.Nodes[cls.LeafID]
		if leaf == nil {
			// Fallback to branch
			g.preserveRoot(tree)
			tree.AddChild(tree.RootID, content, source)
		} else {
			parentID := leaf.ParentID
			if parentID == "" {
				// Leaf is root — preserve and add as sibling
				g.preserveRoot(tree)
				parentID = tree.RootID
			}
			tree.AddChild(parentID, content, source)
		}
		g.bubbleUp(tree, tree.RootID)
	}
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
		child.Sources = append(child.Sources, root.Sources...)
		child.Frequency = root.Frequency
		child.Weight = root.Weight
		child.Created = root.Created
		child.LastAccessed = root.LastAccessed
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

	// Collect all children content, tokenize, count frequencies
	freq := make(map[string]int)
	for _, childID := range node.ChildIDs {
		child := tree.Nodes[childID]
		if child == nil {
			continue
		}
		tokens := text.Tokenize(child.Content)
		for _, t := range tokens {
			freq[t]++
		}
	}

	// Extract top N terms by frequency
	type termCount struct {
		term  string
		count int
	}
	sorted := make([]termCount, 0, len(freq))
	for t, c := range freq {
		sorted = append(sorted, termCount{t, c})
	}
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].count != sorted[j].count {
			return sorted[i].count > sorted[j].count
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
}

// GenerateContext formats the forest state as a compact context block.
func (g *Gate) GenerateContext() string {
	if len(g.Forest.Trees) == 0 {
		return ""
	}

	var b strings.Builder

	// Header
	fmt.Fprintf(&b, "[Focus | %d prompts | %d/%d mem | %d trees]\n",
		g.Forest.Meta.TotalPrompts,
		g.Forest.NodeCount(),
		g.Config.MemorySize,
		len(g.Forest.Trees))

	// Sort trees by root score descending, with Markov transition boost
	type scoredTree struct {
		tree  *forest.Tree
		score float64
	}
	scored := make([]scoredTree, len(g.Forest.Trees))
	now := g.Forest.Trees[0].LastAccessed
	alpha := g.Config.TransitionBoost
	for i, t := range g.Forest.Trees {
		decayScore := t.Root().Score(now, g.Config.DecayRate)
		// Boost by transition probability from current topic
		if alpha > 0 && g.Chain.LastTopic != "" {
			tp := g.Chain.Probability(g.Chain.LastTopic, t.ID)
			decayScore *= (1 + alpha*tp)
		}
		scored[i] = scoredTree{t, decayScore}
	}
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	// Limit to top 5 trees
	limit := 5
	if limit > len(scored) {
		limit = len(scored)
	}

	for _, st := range scored[:limit] {
		fmt.Fprintf(&b, "  [%.2f] %s\n", st.score, st.tree.Root().Content)

		// Show up to 3 recent leaves
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
				continue // Don't re-show root
			}
			content := leaf.Content
			if len(content) > 80 {
				content = content[:80] + "..."
			}
			fmt.Fprintf(&b, "    - %s\n", content)
		}
	}

	// Prediction line: show likely next topics if transition data exists
	if g.Chain.LastTopic != "" {
		top := g.Chain.TopTransitions(g.Chain.LastTopic, 3)
		if len(top) > 0 && top[0].Probability >= 0.3 {
			b.WriteString("  -> next:")
			for i, t := range top {
				// Find tree name for this topic ID
				name := t.TopicID[:8] // fallback: truncated ID
				for _, tree := range g.Forest.Trees {
					if tree.ID == t.TopicID {
						root := tree.Root()
						if root != nil {
							name = root.Content
							if len(name) > 30 {
								name = name[:30]
							}
						}
						break
					}
				}
				if i > 0 {
					b.WriteString(",")
				}
				fmt.Fprintf(&b, " %s (%.0f%%)", name, t.Probability*100)
			}
			b.WriteString("\n")
		}
	}

	result := b.String()

	// Enforce context limit
	if g.Config.ContextLimit > 0 && len(result) > g.Config.ContextLimit {
		result = result[:g.Config.ContextLimit]
		// Trim to last complete line
		if idx := strings.LastIndex(result, "\n"); idx > 0 {
			result = result[:idx+1]
		}
	}

	return result + "[/Focus]\n"
}
