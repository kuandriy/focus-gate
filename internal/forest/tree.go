package forest

import "time"

// Tree is a rooted hierarchy of Nodes. The root holds an abstracted summary
// of its children (via bubble-up). Leaf nodes hold actual prompt text.
// Nodes are stored in a flat map for O(1) lookup.
type Tree struct {
	ID           string           `json:"id"`
	RootID       string           `json:"rootId"`
	Nodes        map[string]*Node `json:"nodes"`
	Created      int64            `json:"created"`
	LastAccessed int64            `json:"lastAccessed"`
}

// NewTree creates a tree with a single root node containing the given content.
func NewTree(content string, source string) *Tree {
	root := NewNode(content, 0, source)
	now := time.Now().UnixMilli()
	return &Tree{
		ID:           generateID(now),
		RootID:       root.ID,
		Nodes:        map[string]*Node{root.ID: root},
		Created:      now,
		LastAccessed: now,
	}
}

// Root returns the root node of the tree.
func (t *Tree) Root() *Node {
	return t.Nodes[t.RootID]
}

// AddChild creates a new child node under the given parent and returns it.
func (t *Tree) AddChild(parentID string, content string, source string) *Node {
	parent := t.Nodes[parentID]
	if parent == nil {
		return nil
	}
	child := NewNode(content, parent.Depth+1, source)
	child.ParentID = parentID
	parent.ChildIDs = append(parent.ChildIDs, child.ID)
	t.Nodes[child.ID] = child
	t.LastAccessed = child.Created
	return child
}

// RemoveNode removes a node and all its descendants using iterative DFS.
// It also cleans up the parent's childIds reference.
func (t *Tree) RemoveNode(id string) {
	node := t.Nodes[id]
	if node == nil {
		return
	}

	// Remove from parent's childIds
	if node.ParentID != "" {
		parent := t.Nodes[node.ParentID]
		if parent != nil {
			for i, cid := range parent.ChildIDs {
				if cid == id {
					parent.ChildIDs = append(parent.ChildIDs[:i], parent.ChildIDs[i+1:]...)
					break
				}
			}
		}
	}

	// Iterative DFS to remove node and all descendants
	stack := []string{id}
	for len(stack) > 0 {
		nid := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		n := t.Nodes[nid]
		if n != nil {
			stack = append(stack, n.ChildIDs...)
			delete(t.Nodes, nid)
		}
	}
}

// GetLeaves returns all leaf nodes (nodes with no children).
func (t *Tree) GetLeaves() []*Node {
	var leaves []*Node
	for _, n := range t.Nodes {
		if n.IsLeaf() {
			leaves = append(leaves, n)
		}
	}
	return leaves
}

// GetChildren returns the direct children of a node.
func (t *Tree) GetChildren(nodeID string) []*Node {
	node := t.Nodes[nodeID]
	if node == nil {
		return nil
	}
	children := make([]*Node, 0, len(node.ChildIDs))
	for _, cid := range node.ChildIDs {
		if child := t.Nodes[cid]; child != nil {
			children = append(children, child)
		}
	}
	return children
}

// NodeCount returns the total number of nodes in this tree.
func (t *Tree) NodeCount() int {
	return len(t.Nodes)
}
