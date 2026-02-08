package forest

import (
	"testing"
)

func TestNewNode(t *testing.T) {
	n := NewNode("test content", 0, "src1")
	if n.Content != "test content" {
		t.Errorf("Content = %q, want %q", n.Content, "test content")
	}
	if n.Depth != 0 {
		t.Errorf("Depth = %d, want 0", n.Depth)
	}
	if n.Weight != 1.0 {
		t.Errorf("Weight = %f, want 1.0", n.Weight)
	}
	if n.Frequency != 1 {
		t.Errorf("Frequency = %d, want 1", n.Frequency)
	}
	if len(n.Sources) != 1 || n.Sources[0] != "src1" {
		t.Errorf("Sources = %v, want [src1]", n.Sources)
	}
	if !n.IsLeaf() {
		t.Error("new node should be a leaf")
	}
}

func TestNodeTouch(t *testing.T) {
	n := NewNode("test", 0, "")
	origWeight := n.Weight

	n.Touch(20, "src2")
	if n.Frequency != 2 {
		t.Errorf("Frequency after touch = %d, want 2", n.Frequency)
	}
	if n.Weight <= origWeight {
		t.Error("Weight should increase after touch")
	}
	if len(n.Sources) != 1 || n.Sources[0] != "src2" {
		t.Errorf("Sources = %v, want [src2]", n.Sources)
	}
}

func TestNodeScore(t *testing.T) {
	n := NewNode("test", 0, "")
	now := n.Created

	// Score at creation time: weight=1.0, recency=1.0, depthFactor=1.0
	score := n.Score(now, 0.05)
	if score != 1.0 {
		t.Errorf("Score at creation = %f, want 1.0", score)
	}

	// Score should decay over time
	oneHourLater := now + 3600000
	scoreDecayed := n.Score(oneHourLater, 0.05)
	if scoreDecayed >= score {
		t.Errorf("Score should decay: at creation=%f, after 1h=%f", score, scoreDecayed)
	}

	// Deeper nodes score lower
	deep := NewNode("test", 3, "")
	deep.Created = n.Created
	deep.LastAccessed = n.LastAccessed
	deepScore := deep.Score(now, 0.05)
	if deepScore >= score {
		t.Errorf("Deeper node should score lower: depth0=%f, depth3=%f", score, deepScore)
	}
}

func TestTreeAddChild(t *testing.T) {
	tree := NewTree("root content", "src1")
	root := tree.Root()

	child := tree.AddChild(root.ID, "child content", "src2")
	if child == nil {
		t.Fatal("AddChild returned nil")
	}
	if child.Depth != 1 {
		t.Errorf("child Depth = %d, want 1", child.Depth)
	}
	if child.ParentID != root.ID {
		t.Errorf("child ParentID = %q, want %q", child.ParentID, root.ID)
	}
	if len(root.ChildIDs) != 1 || root.ChildIDs[0] != child.ID {
		t.Errorf("root ChildIDs = %v, want [%s]", root.ChildIDs, child.ID)
	}
	if tree.NodeCount() != 2 {
		t.Errorf("NodeCount = %d, want 2", tree.NodeCount())
	}
}

func TestTreeRemoveNode(t *testing.T) {
	tree := NewTree("root", "")
	root := tree.Root()
	child := tree.AddChild(root.ID, "child", "")
	tree.AddChild(child.ID, "grandchild", "")

	if tree.NodeCount() != 3 {
		t.Fatalf("before removal: NodeCount = %d, want 3", tree.NodeCount())
	}

	// Remove child — should also remove grandchild
	tree.RemoveNode(child.ID)

	if tree.NodeCount() != 1 {
		t.Errorf("after removal: NodeCount = %d, want 1 (root only)", tree.NodeCount())
	}
	if len(root.ChildIDs) != 0 {
		t.Errorf("root ChildIDs = %v, want []", root.ChildIDs)
	}
}

func TestTreeGetLeaves(t *testing.T) {
	tree := NewTree("root", "")
	root := tree.Root()

	// Tree with only root — root is the sole leaf
	leaves := tree.GetLeaves()
	if len(leaves) != 1 {
		t.Fatalf("single-node tree leaves: len = %d, want 1", len(leaves))
	}

	// Add children — root is no longer a leaf
	tree.AddChild(root.ID, "child1", "")
	tree.AddChild(root.ID, "child2", "")

	leaves = tree.GetLeaves()
	if len(leaves) != 2 {
		t.Errorf("with children: leaves = %d, want 2", len(leaves))
	}
}

func TestForestNodeCount(t *testing.T) {
	f := NewForest()
	if f.NodeCount() != 0 {
		t.Errorf("empty forest: NodeCount = %d, want 0", f.NodeCount())
	}

	t1 := NewTree("topic1", "")
	t2 := NewTree("topic2", "")
	f.AddTree(t1)
	f.AddTree(t2)

	if f.NodeCount() != 2 {
		t.Errorf("two trees: NodeCount = %d, want 2", f.NodeCount())
	}

	t1.AddChild(t1.RootID, "child", "")
	if f.NodeCount() != 3 {
		t.Errorf("after add child: NodeCount = %d, want 3", f.NodeCount())
	}
}

func TestForestPrune(t *testing.T) {
	f := NewForest()
	tree := NewTree("root", "")
	root := tree.Root()

	// Add 5 children, pushing total to 6 nodes
	for i := 0; i < 5; i++ {
		tree.AddChild(root.ID, "child", "")
	}
	f.AddTree(tree)

	if f.NodeCount() != 6 {
		t.Fatalf("before prune: NodeCount = %d, want 6", f.NodeCount())
	}

	// Prune to limit of 4
	removed := f.Prune(4, 0.05)

	if f.NodeCount() > 4 {
		t.Errorf("after prune: NodeCount = %d, want <= 4", f.NodeCount())
	}
	if len(removed) == 0 {
		t.Error("Prune should return removed contents")
	}
}

func TestForestPruneRemovesEmptyTrees(t *testing.T) {
	f := NewForest()

	// Tree with root + 1 child = 2 nodes
	tree := NewTree("root", "")
	tree.AddChild(tree.RootID, "child", "")
	f.AddTree(tree)

	// Prune to 0 — should remove everything
	f.Prune(0, 0.05)

	if len(f.Trees) != 0 {
		t.Errorf("after pruning to 0: %d trees remain, want 0", len(f.Trees))
	}
}

func TestTreeAddChildInvalidParent(t *testing.T) {
	tree := NewTree("root", "")
	child := tree.AddChild("nonexistent", "child", "")
	if child != nil {
		t.Error("AddChild with invalid parent should return nil")
	}
}

func TestTreeRemoveNodeNonexistent(t *testing.T) {
	tree := NewTree("root", "")
	// Should not panic
	tree.RemoveNode("nonexistent")
	if tree.NodeCount() != 1 {
		t.Error("removing nonexistent node should not change tree")
	}
}
