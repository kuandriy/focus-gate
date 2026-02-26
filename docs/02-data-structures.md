# Data Structures: Forest, Tree, Node, Heap

The `internal/forest` package is the heart of Focus Gate's memory model. It defines a
hierarchical collection of topic trees that represents the user's conversation intent
at any given moment in time.

---

## Node — The Atomic Unit

**File:** [internal/forest/node.go](../internal/forest/node.go)

Every piece of information in the forest is stored in a `Node`. A node can represent
three different things depending on its position:

1. **Root node** — a synthetic abstraction of its children, produced by `bubbleUp`.
2. **Branch node** — an intermediate aggregation (also synthetic after its first child).
3. **Leaf node** — an actual verbatim user prompt (or a close excerpt of one).

```go
// node.go:12-30
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
    Indexed      bool     `json:"indexed,omitempty"`
}
```

### Field-by-Field Breakdown

| Field | Type | Purpose |
|-------|------|---------|
| `ID` | `string` | Base-36 timestamp + random suffix, globally unique. See [node.go:89-91](../internal/forest/node.go#L89). |
| `Content` | `string` | Raw text. Leaves hold verbatim prompt text; non-leaves hold a `\|`-separated term abstraction from bubbleUp. |
| `Depth` | `int` | Distance from root. Root=0, root's direct children=1, and so on. |
| `Weight` | `float64` | `log₂(Frequency + 1)`. Starts at 1.0. Increases logarithmically with revisits. |
| `Frequency` | `int` | How many times this node has been "touched" (visited/reinforced). |
| `Created` | `int64` | Unix milliseconds at creation. |
| `LastAccessed` | `int64` | Unix milliseconds at last `Touch`. The key input to recency decay. |
| `Sources` | `[]string` | Rolling window of prompt-source labels (e.g. `"p12"`, `"guide-reinforce"`). Capped at `maxSourcesPerNode`. |
| `ChildIDs` | `[]string` | Ordered list of child node IDs. This is the tree's edge structure. |
| `ParentID` | `string` | ID of this node's parent. Empty for root. Used during pruning cascade. |
| `Indexed` | `bool` | Whether this node's content was registered with the TF-IDF engine. Crucial for pruning correctness. |

### The Indexed Flag in Depth

The `Indexed` flag ([node.go:29](../internal/forest/node.go#L29)) is one of the most
important engineering decisions in the codebase. Here is why it exists:

- When a new user prompt becomes a leaf node, `apply()` in [gate.go:240](../internal/gate/gate.go#L240)
  sets `node.Indexed = true`.
- When `bubbleUp` replaces a node's content with a synthetic abstraction, it sets
  `node.Indexed = false` ([gate.go:318](../internal/gate/gate.go#L318)).
- When `Prune` removes a node, it checks `node.Indexed` before calling
  `RemoveDocument` ([forest.go:121-123](../internal/forest/forest.go#L121)).

Without this flag, pruning a synthetic root would try to remove terms from the TF-IDF
engine that were never added to it, decrementing their document-frequency counts below
zero and corrupting all future IDF values.

### Node Scoring Formula

```go
// node.go:60-68
func (n *Node) Score(now int64, decayRate float64) float64 {
    ageHours := float64(now-n.LastAccessed) / 3600000.0
    recency := math.Exp(-decayRate * ageHours)
    depthFactor := 1.0 / (1.0 + float64(n.Depth)*0.15)
    return n.Weight * recency * depthFactor
}
```

Full formula:

```
score = weight × recency × depthFactor

where:
  weight      = log₂(frequency + 1)
  recency     = e^(-decayRate × ageHours)
  depthFactor = 1 / (1 + depth × 0.15)
```

**Weight** grows logarithmically: a node visited once scores `log₂(2) = 1.0`, visited
ten times scores `log₂(11) ≈ 3.46`. Logarithmic growth prevents frequency from
completely dominating the score.

**Recency** is an exponential decay. With `decayRate = 0.05`:
- At 0 hours: `e^0 = 1.00` (full weight)
- At 14 hours: `e^(-0.7) ≈ 0.50` (half weight)
- At 24 hours: `e^(-1.2) ≈ 0.30` (30% weight)
- At 48 hours: `e^(-2.4) ≈ 0.09` (9% weight)

**DepthFactor** penalizes deep nodes. A node at depth 5 has a factor of
`1/(1+5×0.15) = 1/1.75 ≈ 0.57`. This makes deep leaves easier to prune than shallow
ones with similar recency, preventing over-deep trees.

### Touch

```go
// node.go:71-81
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
```

`Touch` increments frequency, recomputes weight, and resets `LastAccessed` to now. The
source window is maintained by slicing from the back, keeping only the most recent
`maxSources` labels.

---

## Tree — Rooted Hierarchy

**File:** [internal/forest/tree.go](../internal/forest/tree.go)

A `Tree` is a single topic thread. It owns a flat map of all its nodes for O(1) lookup
and maintains the logical tree structure through the `ChildIDs` / `ParentID` references
on each `Node`.

```go
// tree.go:8-14
type Tree struct {
    ID           string           `json:"id"`
    RootID       string           `json:"rootId"`
    Nodes        map[string]*Node `json:"nodes"`
    Created      int64            `json:"created"`
    LastAccessed int64            `json:"lastAccessed"`
}
```

The flat-map representation (`Nodes map[string]*Node`) is a deliberate choice:
- `tree.Nodes[id]` is O(1) — no tree traversal needed for lookups.
- The tree structure is encoded implicitly through node parent/child ID references.
- JSON serialization is straightforward; no recursive structs.

### AddChild

```go
// tree.go:35-46
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
```

Adding a child automatically increments `Depth` and registers it in the flat node map.

### RemoveNode — Iterative DFS Subtree Removal

```go
// tree.go:50-80
func (t *Tree) RemoveNode(id string) {
    // ... clean up parent's childIds ...

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
```

Rather than recursive deletion (which risks stack overflow on very deep trees), this uses
an iterative depth-first stack. It first severs the parent's reference to the node, then
walks the subtree removing every descendant.

### GetLeaves

```go
// tree.go:83-91
func (t *Tree) GetLeaves() []*Node {
    var leaves []*Node
    for _, n := range t.Nodes {
        if n.IsLeaf() {
            leaves = append(leaves, n)
        }
    }
    return leaves
}
```

A simple O(n) linear scan of all nodes. Since `IsLeaf` is O(1) (`len(ChildIDs) == 0`),
this is as efficient as possible. The result is unsorted — callers that need ordering
sort the result themselves.

---

## Forest — The Collection

**File:** [internal/forest/forest.go](../internal/forest/forest.go)

```go
// forest.go:16-19
type Forest struct {
    Trees []*Tree `json:"trees"`
    Meta  Meta    `json:"meta"`
}
```

The `Forest` itself is a simple slice of trees plus metadata. The real complexity
lives in `Prune`.

### Meta

```go
// forest.go:8-13
type Meta struct {
    TotalPrompts int   `json:"totalPrompts"`
    Created      int64 `json:"created"`
    LastUpdate   int64 `json:"lastUpdate"`
}
```

`TotalPrompts` is incremented after every `ProcessPrompt` call and used as a prompt
counter for source labels (`"p12"`).

### Prune — Memory-Bounded Leaf Removal with Parent Cascading

`Prune` is the most complex algorithm in the `forest` package.
Full implementation: [forest.go:48-163](../internal/forest/forest.go#L48).

**Goal:** Remove the lowest-scoring leaves until `NodeCount() <= memorySize`.

**Algorithm:**

```
1. Build tree-ID → *Tree lookup map for stable O(1) access after slice mutations.
2. Build initial min-heap of all non-root leaves.
3. Loop while NodeCount > memorySize:
   a. If heap empty → remove worst whole tree (no removable leaves edge case).
   b. Pop lowest-scoring leaf entry from heap.
   c. Validate: skip if tree or node was already removed.
   d. If entry.Node.Indexed → append content to removedContents list.
   e. Remember parentID before removal.
   f. tree.RemoveNode(entry.Node.ID) — iterative DFS subtree deletion.
   g. If tree now has ≤ 1 node → remove entire tree.
   h. Cascade: if parent is now a leaf and is not root → push parent onto heap.
4. Return removedContents for TF-IDF cleanup.
```

**The parent cascade** (step h) is essential for correctness. Consider:

```
root (bubbleUp abstraction)
  └── child-A (leaf)
        └── grandchild-B (leaf)  ← gets pruned first
```

After grandchild-B is pruned, child-A becomes a leaf. Without cascading, child-A would
never enter the heap and would survive even if its score is lower than other existing
leaves. The cascade pushes child-A onto the heap so it competes fairly on the next
iteration.

**Why a tree-ID map?** After a tree is removed from `f.Trees`, any heap entries
referencing it become stale. The `deletedTrees map[string]bool` and `treeByID` map
allow O(1) validation of each popped heap entry without re-scanning the trees slice.
See [forest.go:57-59](../internal/forest/forest.go#L57).

---

## LeafHeap — Min-Heap for Efficient Pruning

**File:** [internal/forest/heap.go](../internal/forest/heap.go)

```go
// heap.go:4-8
type LeafEntry struct {
    Node   *Node
    TreeID string  // stable identifier, survives tree-slice reordering
    Score  float64
}

type LeafHeap []LeafEntry
```

The `LeafHeap` type implements the `container/heap.Interface` from the Go standard
library. This provides:

- `heap.Push`: O(log n) insertion
- `heap.Pop`: O(log n) removal of minimum-score entry
- Initial `heap.Init` is O(n)

**Why min-heap?** Pruning always removes the lowest-scoring element. A min-heap makes
this O(log n) instead of O(n) for each removal. For `memorySize = 100` this is a modest
optimization, but for larger configurations it matters significantly.

```go
// heap.go:15-16
func (h LeafHeap) Len() int           { return len(h) }
func (h LeafHeap) Less(i, j int) bool { return h[i].Score < h[j].Score }
```

`Less(i, j) = h[i].Score < h[j].Score` makes it a min-heap (lowest score at top).
The standard library's heap operations guarantee that `h[0]` is always the minimum.

---

## Structural Invariants

The following invariants hold at all times (enforced by the construction and mutation
code):

1. Every tree has exactly one root node, identified by `Tree.RootID`.
2. Every non-root node's `ParentID` references a node that exists in `tree.Nodes`.
3. If node A's `ChildIDs` contains ID X, then `tree.Nodes[X].ParentID == A.ID`.
4. The root node has `Depth == 0`. Each `AddChild` call increments depth by 1.
5. A node with `Indexed == true` was added via `engine.AddDocument` and must be
   removed via `engine.RemoveDocument` when pruned.
6. A node with `Indexed == false` (synthetic `bubbleUp` content) must never be
   passed to `engine.RemoveDocument`.

---

## Memory Model: Numbers

With `memorySize = 100` (default):

- Maximum 100 nodes across all trees.
- A 10-tree forest with 10 nodes each is at the limit.
- A single deep tree could have up to 100 nodes before pruning.
- `NodeCount()` is O(trees × nodes_per_tree) — called frequently; kept simple.
- `Prune` is called at most once per prompt (only if over limit).

The forest's total size in memory is small: a Node is approximately 200-400 bytes
(content string dominates). 100 nodes ≈ 20-40 KB in memory.
