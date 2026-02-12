package forest

// LeafEntry is a scored leaf for the pruning heap.
type LeafEntry struct {
	Node    *Node
	TreeIdx int
	Score   float64
}

// LeafHeap implements container/heap.Interface as a min-heap ordered by Score.
// The lowest-scoring leaf is at the top, making it the first candidate for pruning.
type LeafHeap []LeafEntry

func (h LeafHeap) Len() int           { return len(h) }
func (h LeafHeap) Less(i, j int) bool { return h[i].Score < h[j].Score }
func (h LeafHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *LeafHeap) Push(x any) { *h = append(*h, x.(LeafEntry)) }

func (h *LeafHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}
