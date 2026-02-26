# The Bidirectional Feedback Loop

**Files:** [internal/guide/guide.go](../internal/guide/guide.go),
[internal/gate/gate.go:488-547](../internal/gate/gate.go#L488),
[cmd/focus/main.go:334-411](../cmd/focus/main.go#L334)

One of Focus Gate's most distinctive features is that it doesn't just track user prompts
— it also tracks **AI responses**. The system closes a feedback loop: user prompts define
topics, AI responses reinforce topics. A tree that the AI has responded to extensively
is harder to prune than one the AI has ignored, even if both had equal prompt frequency.

---

## The Guide: Ring Buffer of AI Summaries

```go
// guide.go:34-37
type Guide struct {
    Entries []Entry `json:"entries"`
    MaxSize int     `json:"maxSize"`
}
```

The `Guide` is a fixed-capacity ring buffer of `Entry` values, each representing the
first 200 characters of an AI response:

```go
// guide.go:11-21
type Entry struct {
    Summary    string   `json:"summary"`
    IntentID   string   `json:"intentId"`
    Refs       []string `json:"refs,omitempty"`
    Timestamp  int64    `json:"timestamp"`
    Reinforced bool     `json:"reinforced,omitempty"`
}
```

- `Summary` — truncated AI response text (up to 200 chars)
- `IntentID` — the ID of the node that was the "active intent" when this response was
  generated (the last leaf in the last tree at time of capture)
- `Reinforced` — set to `true` after `ReinforceFromGuide` has processed this entry

### Ring Buffer Semantics

```go
// guide.go:47-60
func (g *Guide) Add(summary string, intentID string, refs []string) {
    if summary == "" { return }
    g.Entries = append(g.Entries, Entry{...})
    if len(g.Entries) > g.MaxSize {
        g.Entries = g.Entries[len(g.Entries)-g.MaxSize:]
    }
}
```

When the buffer is full, the oldest entry is evicted by slicing from the back. With
`maxSize = 15` (default), Focus Gate keeps the 15 most recent AI response summaries.

---

## Harvesting AI Responses

At [cmd/focus/main.go:334-411](../cmd/focus/main.go#L334), `updateGuide` reads the
Claude Code transcript file to extract the last assistant message:

```go
func updateGuide(g *guide.Guide, transcriptPath string, f *forest.Forest) {
    data, err := os.ReadFile(transcriptPath)
    // ...

    type transcriptEntry struct {
        Role    string `json:"role"`
        Message struct {
            Content json.RawMessage `json:"content"`
        } `json:"message"`
    }

    var transcript []transcriptEntry
    json.Unmarshal(data, &transcript)

    // Walk backwards to find the last assistant message
    snippet := ""
    for i := len(transcript) - 1; i >= 0; i-- {
        if transcript[i].Role != "assistant" { continue }
        // Try as plain string, then as array of content blocks
        // ...
        break
    }

    // Truncate to 200 chars
    if len(snippet) > 200 {
        snippet = snippet[:200] + "..."
    }

    // Link to the most recent leaf
    intentID := ""
    if len(f.Trees) > 0 {
        lastTree := f.Trees[len(f.Trees)-1]
        leaves := lastTree.GetLeaves()
        if len(leaves) > 0 {
            intentID = leaves[len(leaves)-1].ID
        }
    }

    g.Add(snippet, intentID, nil)
}
```

### Content Format Flexibility

Claude Code's transcript uses a flexible content format. The content field of a message
can be:
- A plain JSON string: `"content": "Here is the implementation..."`
- An array of content blocks: `"content": [{"type": "text", "text": "Here is..."}]`

The code tries both, handling both formats transparently. This makes it robust across
different Claude Code versions.

### Timing

`updateGuide` is called **before** `ProcessPrompt`:

```go
// main.go:287-302
if input.TranscriptPath != "" {
    updateGuide(g, input.TranscriptPath, f)
}
gt := gate.NewWithChain(f, e, c, gateCfg)
gt.ReinforceFromGuide(g)
ctx := gt.ProcessPrompt(prompt, ...)
```

This means the AI's response to the **previous prompt** is captured and used to
reinforce the forest **before** the current prompt is classified. The reinforcement
is applied in the correct order: the response to a topic strengthens that topic,
making it more likely to survive pruning when the next prompt arrives.

---

## ReinforceFromGuide — AI Responses Touch Trees

```go
// gate.go:499-547
func (g *Gate) ReinforceFromGuide(gd *guide.Guide) int {
    unreinforced := gd.UnreinforcedEntries()
    if len(unreinforced) == 0 { return 0 }

    reinforced := 0

    for _, entry := range unreinforced {
        tokens := text.Tokenize(entry.Summary)
        if len(tokens) == 0 {
            entry.Reinforced = true
            continue
        }

        responseVec := g.Engine.Vectorize(strings.Join(tokens, " "))

        // Find best-matching tree root by cosine similarity (no Markov boost)
        bestScore := 0.0
        bestTreeIdx := -1
        for i, tree := range g.Forest.Trees {
            root := tree.Root()
            score := tfidf.CosineSimilarity(responseVec, g.nodeVec(root.ID, root.Content))
            if score > bestScore { bestScore = score; bestTreeIdx = i }
        }

        // Only reinforce above branch threshold
        if bestTreeIdx >= 0 && bestScore >= g.Config.BranchThreshold {
            root := g.Forest.Trees[bestTreeIdx].Root()
            root.Touch(g.Config.MaxSourcesPerNode, "guide-reinforce")
            reinforced++
        }

        entry.Reinforced = true
    }

    return reinforced
}
```

### What "Reinforce" Means

Calling `root.Touch(...)` on a tree's root:
1. Increments `root.Frequency`
2. Updates `root.Weight = log₂(Frequency + 1)` (higher weight → higher survival score)
3. Resets `root.LastAccessed = now` (resets the recency decay clock)

The effect: a tree whose topic the AI has been actively discussing is treated as
"recently accessed" even if the user hasn't typed a new prompt about it recently.

### Why Only Roots?

The comment at [gate.go:493-496](../internal/gate/gate.go#L493) explains:
> Only Touch is applied — no new nodes or content changes. AI responses confirm
> existing topics rather than defining new ones.

An AI response is evidence that a topic is active, not that a new sub-topic exists.
Touching only the root:
- Makes the entire tree harder to prune (root score affects all decay calculations)
- Does not introduce new content or structure
- Does not corrupt the `bubbleUp` abstraction

### Why No Markov Boost?

The regular classifier uses `boostFactor = 1 + α × P(tree | lastTopic)`. Reinforcement
deliberately does not:

> Markov boost is excluded because the transition model captures user navigation
> patterns, not AI response flow. — [gate.go:497-498](../internal/gate/gate.go#L497)

The Markov chain models "what topic does the user navigate to next?" — a property of
user intent. AI responses matching a topic is independent of navigation. Using the
Markov boost here could incorrectly reinforce the "likely next" topic even if the AI's
response was about the "current" topic.

### The Reinforced Flag

```go
// guide.go:18-20
Reinforced bool `json:"reinforced,omitempty"`
```

Each entry is reinforced **exactly once**. The `Reinforced` flag is persisted so that
across restarts, entries already processed by `ReinforceFromGuide` are not re-processed.

`UnreinforcedEntries` ([guide.go:65-73](../internal/guide/guide.go#L65)) returns only
unprocessed entries:

```go
func (g *Guide) UnreinforcedEntries() []*Entry {
    var entries []*Entry
    for i := range g.Entries {
        if !g.Entries[i].Reinforced {
            entries = append(entries, &g.Entries[i])
        }
    }
    return entries
}
```

Note it returns **pointers** (`*Entry`) so that `entry.Reinforced = true` in
`ReinforceFromGuide` actually modifies the entry in the slice.

---

## Guide Rendering — Context Injection

```go
// guide.go:77-106
func (g *Guide) Render(f *forest.Forest) string {
    // Build set of valid node IDs (still in forest)
    valid := make(map[string]bool)
    for _, tree := range f.Trees {
        for id := range tree.Nodes { valid[id] = true }
    }

    var b strings.Builder
    for _, e := range g.Entries {
        if e.IntentID != "" && !valid[e.IntentID] {
            continue  // Dead link — node was pruned
        }
        fmt.Fprintf(&b, "  - %s\n", e.Summary)
    }
    return b.String()
}
```

Guide entries are rendered into the context block after the forest summary. They appear
as a `Guide:` section:

```
[Focus | 45 prompts | 12/100 mem | 3 trees]
  [0.82] auth | token | oauth
    - fix the oauth token refresh bug
  [0.41] database | query | index
    - optimize the slow user lookup query
  -> next: database | query (67%)
Guide:
  - Here's the OAuth token refresh endpoint implementation I've created...
  - I've optimized the query by adding a composite index on (user_id, created_at)...
[/Focus]
```

### Dead Link Filtering

Entries whose `IntentID` no longer exists in the forest are silently omitted from the
rendered output. This happens naturally as nodes are pruned. There is no need to
explicitly remove guide entries when a node is pruned — the render function filters
them on the fly.

An entry with an empty `IntentID` is always included (backwards compatibility with
older guide entries that didn't track intent IDs).

---

## The Complete Feedback Loop

```
User types prompt
        │
        ▼
  updateGuide()             ← Harvests AI response to PREVIOUS prompt
        │                      (links it to last-active node)
        ▼
  ReinforceFromGuide()      ← Matches AI summary against trees by cosine
        │                      Touches best matching root
        ▼
  ProcessPrompt()           ← Classifies and applies current prompt
        │
        ▼
  GenerateContext()         ← Emits [Focus...] block
        │
        ├── Guide.Render()  ← Injects guide summaries for live links
        │
        ▼
  AI receives context + prompt
        │
        ▼
  AI generates response
        │
        ▼
  Transcript file updated
        │
        └──── (on next prompt) → updateGuide() harvests this response
```

This creates a genuine bidirectional loop:
- **Forward:** User prompts → forest structure → context → AI behavior
- **Backward:** AI responses → guide entries → reinforcement → forest scores

Topics that are actively discussed by both the user **and** the AI survive longer,
emerge more prominently in context summaries, and are more likely to be extended rather
than branched or replaced.
