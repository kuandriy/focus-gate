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

// Prune removes the lowest-scoring leaves until the forest fits within memorySize.
// Builds the min-heap once, then pops entries in a loop with parent cascading.
// When a leaf is removed and its parent becomes a new leaf (and is not a root),
// the parent is pushed onto the heap so it becomes a pruning candidate.
// Returns the content of pruned nodes that were indexed in the TF-IDF engine,
// so the caller can RemoveDocument them. Non-indexed nodes (synthetic bubble-up
// abstractions) are excluded to prevent document-frequency drift.
func (f *Forest) Prune(memorySize int, decayRate float64) []string {
	if f.NodeCount() <= memorySize {
		return nil
	}

	var removedContents []string
	now := time.Now().UnixMilli()

	// Build a tree-ID → *Tree lookup for O(1) access after slice mutations.
	treeByID := make(map[string]*Tree, len(f.Trees))
	for _, t := range f.Trees {
		treeByID[t.ID] = t
	}

	// Build the min-heap once from all non-root leaves.
	h := &LeafHeap{}
	for _, t := range f.Trees {
		for _, n := range t.GetLeaves() {
			if n.ID == t.RootID {
				continue
			}
			heap.Push(h, LeafEntry{
				Node:   n,
				TreeID: t.ID,
				Score:  n.Score(now, decayRate),
			})
		}
	}

	// Set of tree IDs that have been fully removed.
	deletedTrees := make(map[string]bool)

	for f.NodeCount() > memorySize {
		if h.Len() == 0 {
			// No removable leaves — remove the lowest-scoring entire tree.
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
			for _, n := range f.Trees[worstIdx].Nodes {
				if n.Indexed {
					removedContents = append(removedContents, n.Content)
				}
			}
			deletedTrees[f.Trees[worstIdx].ID] = true
			delete(treeByID, f.Trees[worstIdx].ID)
			f.Trees = append(f.Trees[:worstIdx], f.Trees[worstIdx+1:]...)
			continue
		}

		entry := heap.Pop(h).(LeafEntry)

		// Validate: tree or node may have been removed by a prior tree deletion.
		if deletedTrees[entry.TreeID] {
			continue
		}
		tree := treeByID[entry.TreeID]
		if tree == nil {
			continue
		}
		if tree.Nodes[entry.Node.ID] == nil {
			continue // node already removed
		}

		// Record content for TF-IDF cleanup.
		if entry.Node.Indexed {
			removedContents = append(removedContents, entry.Node.Content)
		}

		// Remember the parent before removal.
		parentID := entry.Node.ParentID

		tree.RemoveNode(entry.Node.ID)

		// If the tree has only the root left (or is empty), remove it.
		if tree.NodeCount() <= 1 {
			for _, n := range tree.Nodes {
				if n.Indexed {
					removedContents = append(removedContents, n.Content)
				}
			}
			deletedTrees[tree.ID] = true
			delete(treeByID, tree.ID)
			// Find and remove from slice
			for i, ft := range f.Trees {
				if ft.ID == tree.ID {
					f.Trees = append(f.Trees[:i], f.Trees[i+1:]...)
					break
				}
			}
			continue
		}

		// Cascade: if the parent became a leaf and is not the root, push it.
		if parentID != "" && parentID != tree.RootID {
			parent := tree.Nodes[parentID]
			if parent != nil && parent.IsLeaf() {
				heap.Push(h, LeafEntry{
					Node:   parent,
					TreeID: tree.ID,
					Score:  parent.Score(now, decayRate),
				})
			}
		}
	}

	return removedContents
}

// AddTree appends a new tree to the forest.
func (f *Forest) AddTree(t *Tree) {
	f.Trees = append(f.Trees, t)
	f.Meta.LastUpdate = time.Now().UnixMilli()
}
