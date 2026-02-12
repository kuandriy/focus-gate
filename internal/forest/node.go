package forest

import (
	"math"
	"math/rand"
	"strconv"
	"time"
)

// Node is the atomic unit of the forest. It represents a single prompt,
// sub-topic, or abstracted parent summary.
type Node struct {
	ID           string   `json:"id"`
	Content      string   `json:"content"`
	Depth        int      `json:"depth"`
	Weight       float64  `json:"weight"`
	Frequency    int      `json:"frequency"`
	Created      int64    `json:"created"`
	LastAccessed int64    `json:"lastAccessed"`
	Sources      []string `json:"sources"`
	ChildIDs     []string `json:"childIds"`
	ParentID     string   `json:"parentId,omitempty"`

	// Indexed indicates this node's content is registered in the TF-IDF engine.
	// Only nodes holding real user prompt text are indexed. Synthetic abstractions
	// produced by bubbleUp are not. Prune uses this flag to decide whether to call
	// RemoveDocument — calling it on non-indexed content would decrement document
	// frequencies for terms that were never added, corrupting IDF over time.
	Indexed bool `json:"indexed,omitempty"`
}

// NewNode creates a node with a unique ID and initial values.
func NewNode(content string, depth int, source string) *Node {
	now := time.Now().UnixMilli()
	var sources []string
	if source != "" {
		sources = []string{source}
	}
	return &Node{
		ID:           generateID(now),
		Content:      content,
		Depth:        depth,
		Weight:       1.0,
		Frequency:    1,
		Created:      now,
		LastAccessed: now,
		Sources:      sources,
	}
}

// Score computes the survival priority for pruning.
//
//	score = weight × recency × depthFactor
//
// where:
//
//	weight     = log2(frequency + 1)
//	recency    = e^(-decayRate × ageHours)
//	depthFactor = 1 / (1 + depth × 0.15)
func (n *Node) Score(now int64, decayRate float64) float64 {
	ageHours := float64(now-n.LastAccessed) / 3600000.0
	if ageHours < 0 {
		ageHours = 0
	}
	recency := math.Exp(-decayRate * ageHours)
	depthFactor := 1.0 / (1.0 + float64(n.Depth)*0.15)
	return n.Weight * recency * depthFactor
}

// Touch increments the frequency and updates weight and last accessed time.
func (n *Node) Touch(maxSources int, source string) {
	n.Frequency++
	n.Weight = math.Log2(float64(n.Frequency) + 1)
	n.LastAccessed = time.Now().UnixMilli()
	if source != "" && maxSources > 0 {
		n.Sources = append(n.Sources, source)
		if len(n.Sources) > maxSources {
			n.Sources = n.Sources[len(n.Sources)-maxSources:]
		}
	}
}

// IsLeaf returns true if the node has no children.
func (n *Node) IsLeaf() bool {
	return len(n.ChildIDs) == 0
}

// generateID creates a unique ID from timestamp base36 + random suffix.
func generateID(now int64) string {
	return strconv.FormatInt(now, 36) + strconv.FormatInt(rand.Int63n(1000), 36)
}
