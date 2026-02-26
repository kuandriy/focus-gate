# The Classification Pipeline

**Primary file:** [internal/gate/gate.go](../internal/gate/gate.go)

The Gate is the engine room of Focus Gate. It receives a raw prompt and, in a single
call to `ProcessPrompt`, classifies it against the existing forest, mutates the forest,
records the transition in the Markov chain, maintains the TF-IDF corpus, potentially
prunes the forest, and returns the formatted context string.

---

## Entry Point: ProcessPrompt

```go
// gate.go:109-173
func (g *Gate) ProcessPrompt(prompt string, source string) string {
    tokens := text.Tokenize(prompt)
    if len(tokens) == 0 { return "" }

    vec := g.Engine.VectorizeTokens(tokens)

    cls := g.classify(vec)
    g.apply(cls, prompt, source, tokens)

    // ... Markov recording, TF-IDF update, vector cache reset, pruning ...

    return g.GenerateContext()
}
```

The function is intentionally split into:
1. `classify(vec)` — read-only scoring, returns a `Classification` decision
2. `apply(cls, ...)` — write-only mutation based on that decision

This separation makes each step independently testable and the logic easier to follow.

---

## Step 1: Tokenize and Vectorize

```go
tokens := text.Tokenize(prompt)
vec := g.Engine.VectorizeTokens(tokens)
```

`VectorizeTokens` is used here (not `Vectorize`) because `tokens` are already computed
and reused later in `apply` (for `AddDocument`). This avoids double tokenization.

If `tokens` is empty (the prompt is purely stop words or whitespace), `ProcessPrompt`
returns immediately with no context — no forest mutation, no state save.

---

## Step 2: classify — Score All Trees

```go
// gate.go:182-233
func (g *Gate) classify(vec tfidf.Vector) Classification {
    if len(g.Forest.Trees) == 0 || vec == nil {
        return Classification{Action: ActionNew, Score: 0}
    }

    best := Classification{Action: ActionNew, Score: 0}
    alpha := g.Config.TransitionBoost

    for i, tree := range g.Forest.Trees {
        root := tree.Root()

        boostFactor := 1.0
        if alpha > 0 && g.Chain.LastTopic != "" {
            boostFactor = 1.0 + alpha*g.Chain.Probability(g.Chain.LastTopic, tree.ID)
        }

        // Compare against root
        rootVec := g.nodeVec(root.ID, root.Content)
        rootSim := tfidf.CosineSimilarity(vec, rootVec) * boostFactor
        if rootSim > best.Score { /* update best */ }

        // Compare against each leaf
        for _, leaf := range tree.GetLeaves() {
            leafVec := g.nodeVec(leaf.ID, leaf.Content)
            leafSim := tfidf.CosineSimilarity(vec, leafVec) * boostFactor
            if leafSim > best.Score { /* update best */ }
        }
    }

    // Apply thresholds
    if best.Score >= g.Config.ExtendThreshold { best.Action = ActionExtend }
    else if best.Score >= g.Config.BranchThreshold { best.Action = ActionBranch }
    else { best.Action = ActionNew }

    return best
}
```

### What Gets Scored

Every call scores:
- The **root** of every tree (which holds a `bubbleUp` synthetic abstraction)
- Every **leaf** of every tree (which holds real user prompt text)

Branch nodes (intermediate, non-leaf, non-root) are **not scored directly**. They are
also synthetic `bubbleUp` abstractions and their content is already redundantly
represented in the root. Scoring them would double-count.

### Why Score Roots at All?

Roots hold term abstractions extracted from children, representing the broad topic. If a
new prompt is vaguely related to a topic (e.g., "tell me about OAuth" when the root
summarizes "auth | token | oauth | user"), the root may score higher than any single leaf,
triggering a `ActionBranch` rather than `ActionExtend`.

### Vector Cache

```go
// gate.go:99-106
func (g *Gate) nodeVec(nodeID string, content string) tfidf.Vector {
    if v, ok := g.vecCache[nodeID]; ok {
        return v
    }
    v := g.Engine.Vectorize(content)
    g.vecCache[nodeID] = v
    return v
}
```

The `vecCache map[string]tfidf.Vector` caches computed vectors per node ID. Without it,
every `classify()` call would re-tokenize and re-vectorize every node in the forest —
O(nodes × tokenize). With it, after the first prompt, subsequent classifies are O(nodes ×
dot_product), which is much cheaper.

The cache is **invalidated in two places**:

1. `bubbleUp` invalidates entries when a node's content changes ([gate.go:376](../internal/gate/gate.go#L376)):
   ```go
   delete(g.vecCache, nodeID)
   ```

2. `AddDocument` shifts IDF globally, making all cached vectors stale ([gate.go:142-143](../internal/gate/gate.go#L142)):
   ```go
   g.vecCache = make(map[string]tfidf.Vector)
   ```

The cache is **transient** — never persisted. This is correct because IDF weights
change between sessions as the corpus grows, so cached vectors from the previous session
would be stale even if stored.

### The Three Thresholds

```
score >= 0.55  →  ActionExtend   (closely related to an existing leaf/root)
score >= 0.25  →  ActionBranch   (broadly related to an existing topic)
score  < 0.25  →  ActionNew      (unrelated — start a new topic tree)
```

These defaults ([gate.go:28-38](../internal/gate/gate.go#L28)) were designed empirically.
Users can tune them via `config.json`. The `/focus score "prompt"` command ([slash.go:341](../cmd/focus/slash.go#L341))
lets you preview which action would fire without committing.

---

## Step 3: apply — Mutate the Forest

```go
// gate.go:236-276
func (g *Gate) apply(cls Classification, content string, source string, tokens []string) {
    switch cls.Action {
    case ActionNew:
        tree := forest.NewTree(content, source)
        tree.Root().Indexed = true
        g.Forest.AddTree(tree)

    case ActionBranch:
        tree := g.Forest.Trees[cls.TreeIdx]
        g.preserveRoot(tree)
        child := tree.AddChild(tree.RootID, content, source)
        if child != nil { child.Indexed = true }
        g.bubbleUp(tree, tree.RootID)

    case ActionExtend:
        tree := g.Forest.Trees[cls.TreeIdx]
        leaf := tree.Nodes[cls.LeafID]
        if leaf == nil {
            // Fallback to branch
            g.preserveRoot(tree)
            child := tree.AddChild(tree.RootID, content, source)
            if child != nil { child.Indexed = true }
        } else {
            parentID := leaf.ParentID
            if parentID == "" {
                // Leaf is root — preserve and add as sibling
                g.preserveRoot(tree)
                parentID = tree.RootID
            }
            child := tree.AddChild(parentID, content, source)
            if child != nil { child.Indexed = true }
        }
        g.bubbleUp(tree, tree.RootID)
    }
}
```

### ActionNew — Create a New Topic Tree

A new single-node tree is created with the prompt as its root content. The root is
immediately marked `Indexed = true` since it holds real prompt text.

No `bubbleUp` is needed — a single-node tree has no children to abstract over.

### ActionBranch — Add Under the Existing Root

The prompt is broadly related to the topic (`score ∈ [0.25, 0.55)`). It becomes a
direct child of the root.

**`preserveRoot` must be called first.** When the tree was a single-node tree, the root
held real prompt text. If we add a child now and then `bubbleUp` the root, we would
overwrite that real content with a synthetic abstraction — losing the original prompt
content. `preserveRoot` copies it to safety first.

After adding the child, `bubbleUp(tree, tree.RootID)` regenerates the root's content
from all children.

### ActionExtend — Add Near a Matching Leaf

The prompt is closely related to a specific existing leaf (`score >= 0.55`).
The new node becomes a sibling of that leaf (added under the leaf's parent).

**Why a sibling, not a child?** The matching leaf holds a specific past prompt. The new
prompt is semantically close, meaning they are parallel sub-topics of the same parent.
Making the new node a child of the leaf would imply the new prompt is a refinement of
the old specific prompt — which is almost never the intent. Siblings share a parent and
represent related but independent contributions to that topic.

**Edge case: leaf is root** — If the best-matching leaf is the root itself (a
single-node tree where root == leaf), `preserveRoot` is called and the new node becomes
a child of the root. See [gate.go:263-270](../internal/gate/gate.go#L263).

**Edge case: leaf was removed** — A heap pop during a concurrent prune (impossible in
single-threaded mode but defensive) or a race condition could result in `cls.LeafID` no
longer existing in the tree. The nil check at [gate.go:255](../internal/gate/gate.go#L255)
falls back to ActionBranch behavior.

---

## Step 4: preserveRoot — Edge Case Handling

```go
// gate.go:281-297
func (g *Gate) preserveRoot(tree *forest.Tree) {
    root := tree.Root()
    if root == nil || !root.IsLeaf() {
        return  // Nothing to do — root already has children
    }
    // Root is a leaf (single-node tree). Preserve its content as a child.
    child := tree.AddChild(root.ID, root.Content, "")
    if child != nil {
        child.Sources = append(child.Sources, root.Sources...)
        child.Frequency = root.Frequency
        child.Weight = root.Weight
        child.Created = root.Created
        child.LastAccessed = root.LastAccessed
        child.Indexed = root.Indexed  // Child now owns the original prompt
    }
}
```

This handles the transition from a single-node tree to a multi-node tree. The preserved
child:
- Gets all the original metadata (sources, frequency, weight, timestamps)
- Inherits the `Indexed` flag — it now owns the indexed content

After this, `bubbleUp` will treat the root as a non-leaf and replace its content with
an abstraction, while the preserved child retains the original prompt text as a leaf.

The root's `Indexed` flag is **not** explicitly cleared here — `bubbleUp` does that
when it generates the new synthetic content ([gate.go:318](../internal/gate/gate.go#L318)):
```go
node.Indexed = false
```

---

## Step 5: Post-Classification Bookkeeping

After `apply`, `ProcessPrompt` performs several housekeeping steps:

```go
// gate.go:121-170
// Determine current tree ID
currentTreeID := ...

// Record Markov transition
g.Chain.Record(g.Chain.LastTopic, currentTreeID)
g.Chain.LastTopic = currentTreeID

// Increment total prompts
g.Forest.Meta.TotalPrompts++

// Add prompt to TF-IDF corpus
g.Engine.AddDocument(tokens)

// Reset vector cache (IDF shifted globally)
g.vecCache = make(map[string]tfidf.Vector)

// Prune if over memory limit
if g.Forest.NodeCount() > g.Config.MemorySize {
    removed := g.Forest.Prune(g.Config.MemorySize, g.Config.DecayRate)
    for _, content := range removed {
        g.Engine.RemoveDocument(text.Tokenize(content))
    }
    // Sync Markov chain: remove topics for pruned trees
}
```

**Order matters here:**
1. `apply` adds the node to the forest — forest may now exceed `memorySize`.
2. `AddDocument` adds to TF-IDF corpus — IDF shifts.
3. `vecCache` reset — all previously cached vectors are stale post-`AddDocument`.
4. `Prune` removes nodes — provides `removedContents` for TF-IDF cleanup.
5. `RemoveDocument` removes pruned content from TF-IDF.
6. Markov sync — prunes topics for trees that are now gone.

Notice that `AddDocument` is called **after** `apply`. This means that during the
`bubbleUp` call inside `apply`, the new prompt's tokens are **not yet in the corpus**.
The `bubbleUp` code handles this explicitly:

```go
// gate.go:349-353
idf := g.Engine.IDF(t)
if idf == 0 {
    // Term not in corpus yet — use presence count alone as fallback.
    idf = 1.0
}
```

---

## The Classification struct

```go
// gate.go:63-68
type Classification struct {
    Action  Action
    TreeIdx int
    LeafID  string  // For extend: the matching leaf
    Score   float64
}
```

`TreeIdx` is an index into `g.Forest.Trees`. It is used immediately in `apply` (before
any slice mutation), so index stability is guaranteed. `LeafID` is the string ID of the
best-matching leaf — used in ActionExtend to find the node's parent.

---

## DryRun — Non-Mutating Scoring Trace

**File:** [internal/gate/dryrun.go](../internal/gate/dryrun.go)

`DryRun` mirrors `classify` exactly but collects per-tree, per-leaf scores into a
structured `DryRunResult` for display:

```go
// dryrun.go:62-158
func (g *Gate) DryRun(prompt string) DryRunResult {
    tokens := text.Tokenize(prompt)
    vec := g.Engine.VectorizeTokens(tokens)
    // ... same loop as classify, but records every TreeScore and LeafScore
}
```

`DryRunResult` includes:
- `Tokens []string` — the stemmed token list
- `Vector []VectorTerm` — the TF-IDF vector for the prompt
- `TreeScores []TreeScore` — per-tree scoring with root/leaf breakdowns
- `BestAction / BestScore / BestTree / BestLeaf` — the final decision

This is exposed via `/focus score "prompt"` in chat ([slash.go:341](../cmd/focus/slash.go#L341))
and `--dry-run "prompt"` on the CLI ([main.go:184](../cmd/focus/main.go#L184)).

The key guarantee of DryRun: **no state is mutated**. No forest changes, no
TF-IDF updates, no Markov recording, no saves. The result accurately predicts what
`ProcessPrompt` would do without committing to it.
