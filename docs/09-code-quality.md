# Code Quality Analysis

This document examines the engineering quality of the Focus Gate codebase: design
patterns, testing strategy, identified trade-offs, and areas worth noting for future
development.

---

## Architecture Strengths

### 1. Separation of Concerns — No Package Knows Too Much

Each `internal/` package has a single, well-defined responsibility:

| Package | Responsibility | Knows About |
|---------|---------------|-------------|
| `forest` | Data structure | Nothing external |
| `tfidf` | Vectorization | `text` (for `Vectorize`) |
| `text` | Normalization | Nothing external |
| `gate` | Classification + context | `forest`, `tfidf`, `text`, `guide`, `markov` |
| `guide` | AI response buffer | `forest` (for dead-link filtering) |
| `markov` | Transition model | Nothing external |
| `persist` | Atomic I/O | Nothing external |

`gate` is the only package with multiple dependencies — it is the orchestration layer.
All other packages are independently usable and independently testable.

### 2. Indexed Flag — Careful State Tracking

The `Indexed bool` field on `Node` ([forest/node.go:29](../internal/forest/node.go#L29))
is an example of encoding invariants in the type system rather than hoping callers
remember the rules. Without this flag, any code that adds or removes nodes from the
TF-IDF corpus would have to track externally which nodes were real vs. synthetic — a
classic source of bugs.

By storing the flag on the node itself, `Prune` can make the correct decision locally:

```go
// forest.go:121-123
if entry.Node.Indexed {
    removedContents = append(removedContents, entry.Node.Content)
}
```

### 3. Zero External Dependencies

The entire project compiles with `go build ./...` and the standard library only.
No `go get`, no `vendor/`, no module proxy issues. This means:
- Reproducible builds forever
- No supply-chain risk
- Easy to audit (the entire codebase is the codebase)
- Deployable to any environment with `GOARCH/GOOS` set

### 4. Multiplicative Markov Boost Is Correctly Designed

A common mistake with "boost" signals is additive combination:
```
score = cosine + α × P  ← WRONG
```
This would allow a high `P` to create a positive score even when `cosine = 0`, meaning
the Markov chain could force a match with a semantically unrelated tree just because
the user visited it previously.

The multiplicative form ([gate.go:205](../internal/gate/gate.go#L205)) is correct:
```
score = cosine × (1 + α × P)  ← CORRECT
```
When `cosine = 0`, `score = 0` regardless of P. The boost only amplifies existing
similarity.

### 5. Vector Cache Invalidation Is Complete

The `vecCache` is invalidated in exactly the right places:

- **`bubbleUp` changes a node's content** → delete that node's cache entry ([gate.go:376](../internal/gate/gate.go#L376))
- **`AddDocument` shifts all IDF values** → reset entire cache ([gate.go:143](../internal/gate/gate.go#L143))

No cases are missed. The cache is also correctly declared as transient (not persisted)
because IDF values from the last session are stale when the session starts with a
different corpus size.

### 6. Windows-Safe Persistence

The `runtime.GOOS == "windows"` check in [persist.go:34](../internal/persist/persist.go#L34)
is the only platform-specific code in the entire codebase. Rather than using platform
build tags (which would require duplicate file versions), it's a runtime check on the
atomic rename path, which is clean and correct.

---

## Testing Strategy

### Coverage Overview

| Package | Test File | Test Count | Notable |
|---------|-----------|------------|---------|
| `forest` | `forest_test.go` | 10 | Pruning, cascade, score |
| `tfidf` | `engine_test.go` + `vector_test.go` | 13 | IDF floor, cosine |
| `text` | `tokenizer_test.go` + `stemmer_test.go` | 6 | Overrides, edge cases |
| `gate` | `gate_test.go` | 17 | All 3 actions, bubbleUp, Markov |
| `markov` | `chain_test.go` | 11 | Self-skip, prune sync |
| `guide` | `guide_test.go` | 5 | Overflow, dead links |
| `persist` | `persist_test.go` | 6 | Atomic, recovery |
| `cmd/focus` | `slash_test.go` | 3 | Parse, memory bar |

**Total: ~71 test cases**

### What Is Well-Tested

**The pruning algorithm** has comprehensive tests covering:
- Basic removal under memory limit
- Parent cascade (removing a leaf exposes its parent)
- Full tree removal when tree drops to 1 node
- Correct TF-IDF content reporting for indexed nodes
- Non-indexed (synthetic) nodes correctly excluded from return value

**TF-IDF vectorization** tests verify:
- Virtual document floor behavior (cold start discrimination)
- Incremental add/remove correctness
- IDF formula with specific values
- Cosine similarity with known vectors (verifiable by hand)

**Gate classification** tests cover all three decision paths (New, Branch, Extend),
`preserveRoot` edge cases, `bubbleUp` term selection, and Markov tiebreaking.

### Test Quality Notes

Tests use Go's built-in `testing` package with no test framework dependencies.
Most tests follow the pattern:
```go
func TestXxx(t *testing.T) {
    // Arrange
    e := NewEngine()
    e.AddDocument([]string{"auth", "token"})

    // Act
    idf := e.IDF("auth")

    // Assert
    if math.Abs(idf - expected) > 1e-9 {
        t.Errorf("IDF = %v, want %v", idf, expected)
    }
}
```

No test helpers or fixtures are shared across packages — each package's tests are
self-contained.

### Gaps in Test Coverage

1. **No integration test for the full hook path** — The pipeline from `handlePrompt`
   through all packages is not tested end-to-end. An integration test that writes
   a transcript file, invokes `handlePrompt`, and verifies the output would catch
   interface-level bugs.

2. **No concurrency tests** — Focus Gate is designed to be single-threaded (one
   invocation per prompt, no goroutines). Concurrent access is not a current concern,
   but this assumption could break if the hook model changes.

3. **No fuzz testing** — `text.Tokenize` and `text.Stem` are called with arbitrary
   user input. Fuzz testing these would surface edge cases in the stemmer and tokenizer.

4. **No benchmark tests** — The performance-critical paths (cosine similarity, heap
   operations, vectorization) have no benchmarks. Adding `BenchmarkXxx` tests would
   verify that optimizations don't regress.

---

## Notable Design Decisions

### Decision 1: Why Not Embed a Word Embedding Model?

Using pre-trained embeddings (word2vec, GloVe, sentence-transformers) would give
dramatically better semantic similarity. `"fix bug"` and `"repair defect"` would score
high similarity; TF-IDF would score 0 (no shared tokens).

The tradeoffs:
- **Model size**: A useful embedding model is 50-300MB. A 100KB binary becomes a 300MB
  package.
- **Network**: Cloud-based embeddings require API calls — latency, cost, offline failure.
- **Complexity**: Loading and running inference adds significant code complexity.
- **Zero-dependency constraint**: Any embedding library breaks the "no external deps" goal.

For the current use case (technical prompts in a single user's session), TF-IDF is
surprisingly effective because technical vocabulary is precise and distinctive.
"authentication", "kubernetes", "react-hook" have high IDF values and never co-occur
accidentally.

### Decision 2: Why Not Use Semantic Trees (AST/Dependency Parsing)?

Code prompts have structure: "add a function", "fix a bug", "refactor X to use Y".
Syntactic parsing could extract the intent structure more precisely.

Rejected because:
- Would require a full NLP parser (another large dependency)
- Works poorly on the mix of natural language and technical jargon in typical prompts
- The current bag-of-words approach captures the "what" (tokens) effectively
- The tree structure (forest) already models the hierarchical topic structure

### Decision 3: Parent Content = bubbleUp, Not User-Visible Summary

Parent content is an internal representation (`"authenticat | token | oauth"`) optimized
for TF-IDF comparison, not for human reading. The context block shows this to the LLM,
which can parse pipe-separated term lists effectively.

An alternative would be to generate natural language summaries ("Authentication and
OAuth token management") using an LLM. This was not implemented because:
- It would require an API call on every `bubbleUp`
- It would create a dependency on the LLM during the hook (chicken-and-egg problem)
- The pipe-separated format works well enough for the LLM to understand

### Decision 4: The 80-Character Leaf Truncation

In `GenerateContext` ([gate.go:438](../internal/gate/gate.go#L438)):
```go
if len(content) > 80 {
    content = content[:80] + "..."
}
```

Leaf content in the context block is truncated at 80 characters. This is a simple
length-based truncation that may split in the middle of a word. The `contextLimit`
setting (600 bytes total) is the primary constraint; the 80-character leaf truncation
is a secondary defense against very long prompts monopolizing the context block.

A more sophisticated approach would truncate at word boundaries and extract the most
informative portion of the prompt. For a future improvement.

---

## Potential Failure Modes

### 1. Very Short Prompts

Prompts like "why?" or "thanks" produce 0-1 tokens after stop-word filtering. These
create nodes with essentially empty TF-IDF vectors, scoring 0 against everything, and
always landing in `ActionNew`. Over time, this could create many tiny one-word trees
that fill memory with noise.

Mitigation: The decay model means these promptly scored-zero trees will be the first
to be pruned.

### 2. Very Long Prompts (Pasted Code)

If the user pastes a large code block and CleanPrompt doesn't strip it (e.g., the
code is not wrapped in IDE tags), it generates a very large token set. The TF-IDF
vector may be dominated by code-specific terms.

Mitigation: `CleanPrompt` removes `<file_content>` and similar IDE-injected blocks.
But directly-pasted code without tags would not be stripped.

### 3. Topic Drift from bubbleUp

As a tree grows, `bubbleUp` replaces the root content with an abstraction of the current
children. If the conversation shifts direction within a tree (e.g., starts with
"authentication" but evolves to "deployment pipelines"), the root abstraction drifts to
reflect the mix. Future prompts about authentication may score lower against this tree
than expected.

This is somewhat intentional — the root should reflect the current subtopic distribution.
But it can cause surprising behavior if the tree evolves significantly.

### 4. IDF Drift After --reset

After `--reset`, the TF-IDF engine starts fresh. Terms that had high IDF in the previous
session (rare, specific terms) may temporarily have low IDF (because they're the only
terms in the new corpus). The virtual document floor mitigates this for the first few
prompts.

---

## Code Style and Conventions

- **Receiver naming**: Single lowercase letter (`f *Forest`, `g *Gate`, `e *Engine`)
- **Error handling**: Always `if err != nil { return err }` — no `_` discarding unless
  intentional and commented
- **Comments**: Public functions have doc comments; complex private functions have
  inline explanations; algorithms have their mathematical formulas in comments
- **Package layout**: `*_test.go` files in the same package (`package forest` not
  `package forest_test`) — allows access to unexported fields where needed for testing
- **JSON tags**: All serialized fields have explicit JSON tags with `omitempty` where
  appropriate

---

## Summary Scorecard

| Dimension | Score | Notes |
|-----------|-------|-------|
| Correctness | ★★★★★ | Core algorithms are mathematically sound; edge cases handled |
| Test Coverage | ★★★★☆ | 71 tests, good unit coverage; no integration/fuzz tests |
| Performance | ★★★★★ | Zero-allocation cosine; incremental TF-IDF; vector caching |
| Readability | ★★★★★ | Clear naming, good comments, logical organization |
| Maintainability | ★★★★☆ | Clean packages; gate.go is the most complex (~550 lines) |
| Robustness | ★★★★★ | Panic recovery; graceful missing files; Windows-safe saves |
| Observability | ★★★★★ | Excellent — health, inspect, dry-run, slash commands |
| Simplicity | ★★★★★ | No external deps; no build complexity; single binary |
