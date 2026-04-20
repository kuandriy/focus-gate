package gate

import (
	"github.com/kuandriy/focus-gate/internal/text"
)

// VectorTerm is a single term-weight pair for display in dry-run output.
// It mirrors tfidf.Term but with JSON-friendly field names.
type VectorTerm struct {
	Term   string  `json:"term"`
	Weight float64 `json:"weight"`
}

// LeafScore holds per-leaf cosine similarity details.
type LeafScore struct {
	LeafID  string  `json:"leafId"`
	Content string  `json:"content"`
	Cosine  float64 `json:"cosine"`
}

// TreeScore holds per-tree classification scoring details.
type TreeScore struct {
	TreeIdx     int         `json:"treeIdx"`
	TreeID      string      `json:"treeId"`
	RootID      string      `json:"rootId"`
	RootContent string      `json:"rootContent"`
	RootCosine  float64     `json:"rootCosine"`
	LeafScores  []LeafScore `json:"leafScores,omitempty"`
}

// DryRunResult contains the full classification trace for a prompt. All scoring
// is computed exactly as ProcessPrompt would — same tokenization, same TF-IDF
// vectors — but no state is mutated. This lets the user verify the classifier's
// behaviour before committing a prompt.
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
// detailed per-tree scoring without mutating any state. This uses the same
// classifyDetailed() logic as ProcessPrompt so the result accurately predicts
// what ProcessPrompt would do.
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

	cls, treeScores := g.classifyDetailed(vec, true)
	result.TreeScores = treeScores
	result.BestAction = cls.Action.String()
	result.BestScore = cls.Score
	result.BestTree = cls.TreeIdx
	result.BestLeaf = cls.LeafID

	return result
}
