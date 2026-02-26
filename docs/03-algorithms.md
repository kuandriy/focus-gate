# Core Algorithms

This document explains every non-trivial algorithm in Focus Gate with mathematical
precision and references to the exact lines where each is implemented.

---

## 1. TF-IDF Vectorization

**Files:** [internal/tfidf/engine.go](../internal/tfidf/engine.go),
[internal/tfidf/vector.go](../internal/tfidf/vector.go)

TF-IDF (Term Frequency – Inverse Document Frequency) converts raw text into a vector
where each dimension corresponds to a term and its value represents that term's
discriminative importance in the document.

### Term Frequency (TF)

Implemented in [text/tokenizer.go:79-92](../internal/text/tokenizer.go#L79):

```go
func TermFrequency(tokens []string) map[string]float64 {
    tf := make(map[string]float64, len(tokens))
    for _, t := range tokens {
        tf[t]++
    }
    n := float64(len(tokens))
    for k := range tf {
        tf[k] /= n
    }
    return tf
}
```

This is **normalized TF**: for each term `t` in a document with `n` total tokens,
`TF(t) = count(t) / n`.

Normalization prevents longer prompts from dominating similarity scores. A prompt of
100 tokens where "authentication" appears twice scores the same TF as a prompt of 50
tokens where it appears once.

### Inverse Document Frequency (IDF)

Implemented in [tfidf/engine.go:67-77](../internal/tfidf/engine.go#L67):

```go
func (e *Engine) IDF(term string) float64 {
    df := e.DocFreq[term]
    if df == 0 {
        return 0
    }
    effectiveDocs := e.TotalDocs
    if effectiveDocs < minVirtualDocs {
        effectiveDocs = minVirtualDocs
    }
    return math.Log2(1 + float64(effectiveDocs)/float64(df))
}
```

Formula: `IDF(t) = log₂(1 + effectiveDocs / DF(t))`

Where:
- `DF(t)` = number of documents (prompts) containing term `t`
- `effectiveDocs = max(TotalDocs, 5)`  (see the virtual floor below)

**Why smoothed IDF?** Classical IDF is `log(N/DF)`. The `+1` inside the logarithm
prevents the formula from returning 0 when `DF = N` (a term in every document). In Focus
Gate's usage, terms like "file" or "function" may appear very frequently, and we still
want non-zero (though very low) weights for them.

**The Virtual Document Floor**

`minVirtualDocs = 5` is the key cold-start mechanism, declared at
[tfidf/engine.go:14](../internal/tfidf/engine.go#L14):

```go
const minVirtualDocs = 5
```

With only 1 real document in the corpus, every term has `DF = 1` and `TotalDocs = 1`,
so `IDF = log₂(1 + 1/1) = 1.0` for all terms — they are all equally discriminative,
which is meaningless.

With the floor at 5, the effective corpus size is 5. If a term appears in 1 of 5 docs:
`IDF = log₂(1 + 5/1) ≈ 2.58`. If it appears in all 5: `IDF = log₂(1 + 5/5) = 1.0`.
This discrimination exists from the very first prompt.

Once `TotalDocs >= 5`, the floor has no effect.

### TF-IDF Weight

For term `t` in document `d`:
```
tfidf(t, d) = TF(t, d) × IDF(t)
```

### Incremental Updates

The engine maintains **document frequency counts** persistently — not the full corpus.
This is the critical difference from a naive rebuild-on-every-prompt approach:

```go
// engine.go:33-42
func (e *Engine) AddDocument(tokens []string) {
    seen := make(map[string]bool, len(tokens))
    for _, t := range tokens {
        if !seen[t] {
            e.DocFreq[t]++
            seen[t] = true
        }
    }
    e.TotalDocs++
}
```

Each unique token's DF is incremented exactly once, regardless of how many times it
appears in the document. This correctly models "in how many documents does this term
appear".

Removal ([engine.go:46-61](../internal/tfidf/engine.go#L46)) is symmetric:

```go
func (e *Engine) RemoveDocument(tokens []string) {
    seen := make(map[string]bool, len(tokens))
    for _, t := range tokens {
        if !seen[t] {
            e.DocFreq[t]--
            if e.DocFreq[t] <= 0 {
                delete(e.DocFreq, t)
            }
            seen[t] = true
        }
    }
    e.TotalDocs--
    if e.TotalDocs < 0 {
        e.TotalDocs = 0
    }
}
```

Terms that drop to zero DF are deleted from the map to prevent unbounded growth.

---

## 2. Cosine Similarity

**File:** [internal/tfidf/vector.go:37-74](../internal/tfidf/vector.go#L37)

Cosine similarity measures the angle between two vectors in high-dimensional space.
A score of 1.0 means the vectors are identical (same terms, same proportions); 0.0 means
they share no terms.

```
cosine(A, B) = (A · B) / (||A|| × ||B||)
```

Where `A · B` is the dot product and `||A||` is the Euclidean norm.

### Zero-Allocation Merge-Join Implementation

```go
func CosineSimilarity(a, b Vector) float64 {
    if len(a) == 0 || len(b) == 0 {
        return 0
    }

    var dot, normA, normB float64
    i, j := 0, 0

    for i < len(a) && j < len(b) {
        if a[i].Word == b[j].Word {
            dot += a[i].Weight * b[j].Weight
            normA += a[i].Weight * a[i].Weight
            normB += b[j].Weight * b[j].Weight
            i++; j++
        } else if a[i].Word < b[j].Word {
            normA += a[i].Weight * a[i].Weight
            i++
        } else {
            normB += b[j].Weight * b[j].Weight
            j++
        }
    }

    // Drain remaining
    for ; i < len(a); i++ { normA += a[i].Weight * a[i].Weight }
    for ; j < len(b); j++ { normB += b[j].Weight * b[j].Weight }

    denom := math.Sqrt(normA) * math.Sqrt(normB)
    if denom == 0 { return 0 }
    return dot / denom
}
```

Both vectors are **sorted by word** ([vector.go:26-29](../internal/tfidf/vector.go#L26)).
This enables a merge-join: two pointers advance through both sorted lists simultaneously.

- If `a[i].Word == b[j].Word`: both terms exist in both vectors — contribute to dot product and both norms.
- If `a[i].Word < b[j].Word`: term only in A — contributes only to `normA`, advance i.
- If `a[i].Word > b[j].Word`: term only in B — contributes only to `normB`, advance j.

**Complexity:** O(n + m) where n = |a|, m = |b|. No allocations during the computation
(no intermediate maps or slices created). This is called on every node in the forest for
every prompt, so these constant-factor optimizations matter in aggregate.

### Why Cosine and Not Euclidean Distance?

Cosine similarity is length-invariant. A short prompt "fix authentication bug" and a
long prompt "please fix the authentication bug in the login service where tokens expire"
would have a high cosine similarity if they share the same key terms with similar
proportions, despite very different vector lengths. Euclidean distance would punish the
length difference.

---

## 3. Node Decay Scoring

**File:** [internal/forest/node.go:60-68](../internal/forest/node.go#L60)

```
score(node, now) = weight × recency × depthFactor

weight      = log₂(frequency + 1)
recency     = e^(-decayRate × ageHours)
depthFactor = 1 / (1 + depth × 0.15)
```

### Rationale for Each Factor

**Weight = log₂(frequency + 1)**

A topic revisited 10 times is more important than one visited once, but the 10th visit
adds less value than the 2nd. Logarithmic growth captures this diminishing returns
property. With `decayRate = 0.05` and equal freshness:

| Frequency | Weight |
|-----------|--------|
| 1         | 1.00   |
| 2         | 1.58   |
| 5         | 2.58   |
| 10        | 3.46   |
| 50        | 5.67   |

**Recency = e^(-k × hours)**

Exponential decay is the standard model for "memory fading over time". With `k = 0.05`:

| Age | Recency |
|-----|---------|
| 0h  | 1.000   |
| 4h  | 0.819   |
| 14h | 0.497   |
| 24h | 0.301   |
| 48h | 0.091   |
| 72h | 0.027   |

Topics from yesterday have ~30% of their original score. Topics from 2 days ago have ~9%.

**depthFactor = 1 / (1 + depth × 0.15)**

Shallow nodes (close to the root) represent broader topics that likely span many
conversations. Deep nodes are highly specific sub-topics of a current thread. When memory
pressure hits, we prefer to prune the most specific (deepest), most stale leaves first.

| Depth | Factor |
|-------|--------|
| 0     | 1.000  |
| 1     | 0.870  |
| 2     | 0.769  |
| 3     | 0.690  |
| 5     | 0.571  |
| 10    | 0.400  |

---

## 4. Markov Transition Model

**File:** [internal/markov/chain.go](../internal/markov/chain.go)

The Markov chain models how the user navigates between topics. After classifying each
prompt into a tree, the system records `Record(lastTopic, currentTopic)`.

### Sparse Matrix Representation

```go
// chain.go:12-17
type Chain struct {
    Counts    map[string]map[string]int  // Counts[from][to] = transition count
    Totals    map[string]int             // Totals[from] = row sum
    LastTopic string                     // ID of the last tree visited
}
```

Only non-zero transitions are stored. With N topics, the worst case is O(N²) but typical
usage is O(N) because users tend to have focused sessions with a few dominant transitions.

### Probability Computation

```go
// chain.go:44-53
func (c *Chain) Probability(from, to string) float64 {
    total := c.Totals[from]
    if total == 0 { return 0 }
    return float64(c.Counts[from][to]) / float64(total)
}
```

`P(to | from) = Counts[from][to] / Totals[from]`

`Totals[from]` is a pre-computed row sum, making probability computation O(1) — no
need to iterate over all destinations to sum them.

### Self-Transition Skipping

```go
// chain.go:31-34
func (c *Chain) Record(from, to string) {
    if from == "" || to == "" || from == to {
        return
    }
    // ...
}
```

Self-transitions (staying in the same topic on consecutive prompts) are deliberately
discarded. The comment at [chain.go:29-30](../internal/markov/chain.go#L29) explains:
> Self-transitions add no information about topic switching patterns and would inflate
> P(A|A), creating redundant stickiness on top of the existing recency/decay mechanism.

If self-transitions were counted, a topic visited 20 times in a row would develop a very
high self-transition probability, boosting it even further — a feedback loop on top of
the already-present recency boost.

### Multiplicative Boost in Classification

```go
// gate.go:199-201
boostFactor := 1.0
if alpha > 0 && g.Chain.LastTopic != "" {
    boostFactor = 1.0 + alpha*g.Chain.Probability(g.Chain.LastTopic, tree.ID)
}
```

The final score for a tree is:
```
boostedScore = cosineSimilarity × (1 + α × P(tree | lastTopic))
```

With `α = 0.2` and a 100% transition probability:
```
boostFactor = 1 + 0.2 × 1.0 = 1.2
```

So the maximum possible boost is 20%. This is intentionally modest — the Markov chain
is a tiebreaker and context hint, not a dominant signal. Crucially, if `cosineSimilarity = 0`,
then `boostedScore = 0` regardless of boost: the Markov chain cannot manufacture a match.

### PruneTopic

```go
// chain.go:101-126
func (c *Chain) PruneTopic(topicID string) {
    // Remove outgoing transitions from this topic
    delete(c.Counts, topicID)
    delete(c.Totals, topicID)

    // Remove incoming transitions from all other rows
    for from, row := range c.Counts {
        if count, ok := row[topicID]; ok {
            delete(row, topicID)
            c.Totals[from] -= count
            // Clean up empty rows
        }
    }
    if c.LastTopic == topicID { c.LastTopic = "" }
}
```

When a tree is pruned from the forest, `PruneTopic` removes all references to it in the
Markov chain — both as a source (outgoing) and as a destination (incoming). This keeps
the chain synchronized with the forest.

---

## 5. BubbleUp — Synthetic Parent Abstraction

**File:** [internal/gate/gate.go:299-377](../internal/gate/gate.go#L299)

When a prompt is added under a node, that node's content must be updated to reflect all
its children. `bubbleUp` is called post-order — children first, then parents — so the
root gets a summary of the entire subtree.

### Algorithm

```
for each non-leaf node (post-order):
  1. Set node.Indexed = false (content becomes synthetic)
  2. For each child:
       - Tokenize child.Content
       - Count per-child presence of each unique term
  3. For each unique term:
       - Score = presence × IDF(term)
       - Fallback IDF = 1.0 if term not in corpus yet
  4. Sort terms by score descending, break ties alphabetically
  5. Take top BubbleUpTerms (default: 6) terms
  6. node.Content = strings.Join(terms, " | ")
  7. Invalidate vecCache[nodeID] — content changed
```

### Scoring Logic

The `presence × IDF` scoring at [gate.go:348-356](../internal/gate/gate.go#L348)
is carefully chosen:

```go
for t, count := range presence {
    idf := g.Engine.IDF(t)
    if idf == 0 {
        idf = 1.0
    }
    sorted = append(sorted, termScore{t, float64(count) * idf})
}
```

**Presence** (how many children contain the term) favors terms with **breadth** across
the subtree. A term appearing in 3 of 4 children is more representative of the parent
than a term appearing only in 1 child.

**× IDF** penalizes terms that appear in most documents (low IDF = common across the
entire corpus). If "function" appears in every prompt, its IDF approaches 1.0. The term
"authentication" might have IDF 3.0 if it only appears in authentication-related prompts.

Combined: `presence × IDF` selects terms that are both breadth-representative within
the subtree AND semantically specific relative to the entire corpus.

### Example

Given a tree with root R and two leaves:
- Leaf 1: "fix authentication token expiry bug"
- Leaf 2: "add authentication token refresh endpoint"

After tokenize + stem:
- Leaf 1 tokens: `[fix, authenticat, token, expiry, bug]`
- Leaf 2 tokens: `[add, authenticat, token, refresh, endpoint]`

Presence counts: `{authenticat:2, token:2, fix:1, expiry:1, bug:1, add:1, refresh:1, endpoint:1}`

If `IDF(authenticat) = 2.5` and `IDF(token) = 1.5`:
- `authenticat`: `2 × 2.5 = 5.0`
- `token`: `2 × 1.5 = 3.0`
- Others: `1 × IDF`

Root content becomes: `"authenticat | token | ..."` — the terms that best represent the subtopic.

---

## 6. Pruning Algorithm (Heap + Parent Cascade)

Already detailed in [docs/02-data-structures.md](./02-data-structures.md#prune--memory-bounded-leaf-removal-with-parent-cascading).

Key algorithmic properties:
- **Time:** O(k log k) where k = initial leaf count (heap build + k pops)
- **Space:** O(k) for the heap
- **Correctness:** Parent cascading ensures that exposing a parent as a leaf makes it
  compete fairly in subsequent rounds
- **Stability:** Tree IDs survive slice reordering via the `treeByID` map

---

## 7. GenerateContext — Sorted Output with Markov Reranking

**File:** [internal/gate/gate.go:380-486](../internal/gate/gate.go#L380)

The context block output sorts trees by their root node score, but with a Markov boost
applied for display ordering too:

```go
decayScore := t.Root().Score(now, g.Config.DecayRate)
if alpha > 0 && g.Chain.LastTopic != "" {
    tp := g.Chain.Probability(g.Chain.LastTopic, t.ID)
    decayScore *= (1 + alpha*tp)
}
```

This means the tree you most likely transition to next appears higher in the context
block — useful for the LLM which reads the block top-to-bottom.

The context is capped at `ContextLimit` bytes (default 600) by a hard truncation,
cleanly breaking at the last newline boundary:

```go
// gate.go:477-483
if g.Config.ContextLimit > 0 && len(result) > g.Config.ContextLimit {
    result = result[:g.Config.ContextLimit]
    if idx := strings.LastIndex(result, "\n"); idx > 0 {
        result = result[:idx+1]
    }
}
```

The `[/Focus]` tag is appended after the limit check so it is always present — the LLM
can always find the boundary even if the content was truncated.
