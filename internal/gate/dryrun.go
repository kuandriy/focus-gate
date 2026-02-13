package gate

import (
	"github.com/kuandriy/focus-gate/internal/text"
	"github.com/kuandriy/focus-gate/internal/tfidf"
)

// VectorTerm is a single term-weight pair for display in dry-run output.
// It mirrors tfidf.Term but with JSON-friendly field names.
type VectorTerm struct {
	Term   string  `json:"term"`
	Weight float64 `json:"weight"`
}

// LeafScore holds per-leaf cosine similarity details. Cosine is the raw
// dot-product score; Boosted is after applying the multiplicative Markov factor.
type LeafScore struct {
	LeafID  string  `json:"leafId"`
	Content string  `json:"content"`
	Cosine  float64 `json:"cosine"`
	Boosted float64 `json:"boosted"`
}

// TreeScore holds per-tree classification scoring details. For each tree we
// compute the raw cosine similarity between the prompt vector and the root
// vector, then multiply by the Markov transition boost factor. Leaf scores
// follow the same formula. The classifier picks the single highest boosted
// score across all roots and leaves.
type TreeScore struct {
	TreeIdx     int         `json:"treeIdx"`
	TreeID      string      `json:"treeId"`
	RootID      string      `json:"rootId"`
	RootContent string      `json:"rootContent"`
	RootCosine  float64     `json:"rootCosine"`
	RootBoosted float64     `json:"rootBoosted"`
	BoostFactor float64     `json:"boostFactor"`
	LeafScores  []LeafScore `json:"leafScores,omitempty"`
}

// DryRunResult contains the full classification trace for a prompt. All scoring
// is computed exactly as ProcessPrompt would — same tokenization, same TF-IDF
// vectors, same multiplicative Markov boost — but no state is mutated. This
// lets the user verify the classifier's behaviour before committing a prompt.
type DryRunResult struct {
	Prompt     string       `json:"prompt"`
	Tokens     []string     `json:"tokens"`
	Vector     []VectorTerm `json:"vector"`
	TreeScores []TreeScore  `json:"treeScores"`
	BestAction string       `json:"bestAction"`
	BestScore  float64      `json:"bestScore"`
	BestTree   int          `json:"bestTree"`
	BestLeaf   string       `json:"bestLeaf,omitempty"`
}

// DryRun classifies a prompt against the current forest state and returns
// detailed per-tree scoring without mutating any state. This mirrors the
// classify() logic exactly — same cosine similarity, same multiplicative
// Markov boost — so the result accurately predicts what ProcessPrompt would do.
//
// The caller should apply text.CleanPrompt before passing the prompt here,
// matching the pre-processing that handlePrompt performs in the hook path.
func (g *Gate) DryRun(prompt string) DryRunResult {
	tokens := text.Tokenize(prompt)
	vec := g.Engine.VectorizeTokens(tokens)

	// Convert the TF-IDF vector to a display-friendly format.
	var vecTerms []VectorTerm
	for _, t := range vec {
		vecTerms = append(vecTerms, VectorTerm{Term: t.Word, Weight: t.Weight})
	}

	result := DryRunResult{
		Prompt: prompt,
		Tokens: tokens,
		Vector: vecTerms,
	}

	// Empty forest or empty vector → automatic ActionNew.
	if len(g.Forest.Trees) == 0 || vec == nil {
		result.BestAction = ActionNew.String()
		return result
	}

	best := Classification{Action: ActionNew, Score: 0}
	alpha := g.Config.TransitionBoost

	for i, tree := range g.Forest.Trees {
		root := tree.Root()
		if root == nil {
			continue
		}

		// Markov boost factor: neutral (1.0) when no transition data exists,
		// scaled up to (1 + α) for high-probability transitions.
		boostFactor := 1.0
		if alpha > 0 && g.Chain.LastTopic != "" {
			boostFactor = 1.0 + alpha*g.Chain.Probability(g.Chain.LastTopic, tree.ID)
		}

		rootVec := g.nodeVec(root.ID, root.Content)
		rootCosine := tfidf.CosineSimilarity(vec, rootVec)
		rootBoosted := rootCosine * boostFactor

		ts := TreeScore{
			TreeIdx:     i,
			TreeID:      tree.ID,
			RootID:      root.ID,
			RootContent: root.Content,
			RootCosine:  rootCosine,
			RootBoosted: rootBoosted,
			BoostFactor: boostFactor,
		}

		if rootBoosted > best.Score {
			best.Score = rootBoosted
			best.TreeIdx = i
			best.LeafID = ""
		}

		// Score each leaf — leaves hold the actual user prompt text.
		for _, leaf := range tree.GetLeaves() {
			leafVec := g.nodeVec(leaf.ID, leaf.Content)
			leafCosine := tfidf.CosineSimilarity(vec, leafVec)
			leafBoosted := leafCosine * boostFactor

			ts.LeafScores = append(ts.LeafScores, LeafScore{
				LeafID:  leaf.ID,
				Content: leaf.Content,
				Cosine:  leafCosine,
				Boosted: leafBoosted,
			})

			if leafBoosted > best.Score {
				best.Score = leafBoosted
				best.TreeIdx = i
				best.LeafID = leaf.ID
			}
		}

		result.TreeScores = append(result.TreeScores, ts)
	}

	// Apply the same threshold logic as classify().
	if best.Score >= g.Config.ExtendThreshold {
		best.Action = ActionExtend
	} else if best.Score >= g.Config.BranchThreshold {
		best.Action = ActionBranch
	} else {
		best.Action = ActionNew
	}

	result.BestAction = best.Action.String()
	result.BestScore = best.Score
	result.BestTree = best.TreeIdx
	result.BestLeaf = best.LeafID

	return result
}
