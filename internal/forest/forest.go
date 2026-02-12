package forest

import (
	"container/heap"
	"time"
)

// Meta holds forest-level metadata.
type Meta struct {
	TotalPrompts int   `json:"totalPrompts"`
	Created      int64 `json:"created"`
	LastUpdate   int64 `json:"lastUpdate"`
}

// Forest is a collection of topic trees with scoring, pruning, and metadata.
type Forest struct {
	Trees []*Tree `json:"trees"`
	Meta  Meta    `json:"meta"`
}

// NewForest creates an empty forest.
func NewForest() *Forest {
	now := time.Now().UnixMilli()
	return &Forest{
		Meta: Meta{
			Created:    now,
			LastUpdate: now,
		},
	}
}

// NodeCount returns the total number of nodes across all trees.
func (f *Forest) NodeCount() int {
	count := 0
	for _, t := range f.Trees {
		count += t.NodeCount()
	}
	return count
}

// AllLeaves returns all leaf nodes across all trees with their tree index.
// decayRate controls the exponential time-decay applied to each node's score.
func (f *Forest) AllLeaves(decayRate float64) []LeafEntry {
	var entries []LeafEntry
	now := time.Now().UnixMilli()
	for i, t := range f.Trees {
		for _, n := range t.GetLeaves() {
			// Skip root nodes — they should not be pruned directly
			if n.ID == t.RootID {
				continue
			}
			entries = append(entries, LeafEntry{
				Node:    n,
				TreeIdx: i,
				Score:   n.Score(now, decayRate),
			})
		}
	}
	return entries
}

// Prune removes the lowest-scoring leaves until the forest fits within memorySize.
// Uses a min-heap for O(log n) extraction per step. Returns the content of pruned
// nodes that were indexed in the TF-IDF engine, so the caller can RemoveDocument
// them. Non-indexed nodes (synthetic bubble-up abstractions) are excluded from
// the returned list to prevent document-frequency drift.
func (f *Forest) Prune(memorySize int, decayRate float64) []string {
	var removedContents []string

	for f.NodeCount() > memorySize {
		now := time.Now().UnixMilli()

		// Build min-heap of all non-root leaves
		h := &LeafHeap{}
		for i, t := range f.Trees {
			for _, n := range t.GetLeaves() {
				if n.ID == t.RootID {
					continue
				}
				heap.Push(h, LeafEntry{
					Node:    n,
					TreeIdx: i,
					Score:   n.Score(now, decayRate),
				})
			}
		}

		if h.Len() == 0 {
			// No removable leaves — remove the lowest-scoring entire tree
			if len(f.Trees) == 0 {
				break
			}
			worstIdx := 0
			worstScore := f.Trees[0].Root().Score(now, decayRate)
			for i := 1; i < len(f.Trees); i++ {
				s := f.Trees[i].Root().Score(now, decayRate)
				if s < worstScore {
					worstScore = s
					worstIdx = i
				}
			}
			// Only return content from indexed nodes for TF-IDF cleanup.
			for _, n := range f.Trees[worstIdx].Nodes {
				if n.Indexed {
					removedContents = append(removedContents, n.Content)
				}
			}
			f.Trees = append(f.Trees[:worstIdx], f.Trees[worstIdx+1:]...)
			continue
		}

		// Pop the lowest-scoring leaf
		entry := heap.Pop(h).(LeafEntry)
		tree := f.Trees[entry.TreeIdx]
		if entry.Node.Indexed {
			removedContents = append(removedContents, entry.Node.Content)
		}
		tree.RemoveNode(entry.Node.ID)

		// If the tree has only the root left (or is empty), remove the tree
		if tree.NodeCount() <= 1 {
			for _, n := range tree.Nodes {
				if n.Indexed {
					removedContents = append(removedContents, n.Content)
				}
			}
			f.Trees = append(f.Trees[:entry.TreeIdx], f.Trees[entry.TreeIdx+1:]...)
		}
	}

	return removedContents
}

// AddTree appends a new tree to the forest.
func (f *Forest) AddTree(t *Tree) {
	f.Trees = append(f.Trees, t)
	f.Meta.LastUpdate = time.Now().UnixMilli()
}

// RemoveTree removes a tree by index.
func (f *Forest) RemoveTree(idx int) {
	if idx >= 0 && idx < len(f.Trees) {
		f.Trees = append(f.Trees[:idx], f.Trees[idx+1:]...)
		f.Meta.LastUpdate = time.Now().UnixMilli()
	}
}
