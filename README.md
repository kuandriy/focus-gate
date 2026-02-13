# Focus Gate

**Prompt intent tracking for AI coding assistants.**

Focus Gate intercepts every prompt you send to your AI coding assistant, mathematically compares it against your accumulated conversation intent, and injects a compact context summary back into the conversation. The AI always knows what you've been working on, even across long sessions where earlier context would otherwise be lost.

Zero external dependencies. Single binary. Built entirely on Go's standard library.

---

## Table of Contents

- [The Idea](#the-idea)
- [How It Works](#how-it-works)
- [Install](#install)
- [Usage](#usage)
- [Algorithms](#algorithms)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [License](#license)

---

## The Idea

AI coding assistants have a context window — a fixed amount of text they can "see" at once. In long conversations, your earlier requests scroll out of that window and are forgotten.

Focus Gate solves this by maintaining a **living summary** of everything you've discussed. Each time you type a prompt, Focus Gate:

1. Reads your new prompt
2. Compares it to everything you've asked before
3. Decides if this is a continuation of an existing topic or something new
4. Updates its internal model of your intent
5. Sends a compact summary back to the AI alongside your prompt

The AI never sees this machinery — it just receives your prompt enriched with a small block of context that says "here's what the user has been focused on."

---

## How It Works

### The Hook

Focus Gate runs as a [Claude Code hook](https://docs.anthropic.com/en/docs/claude-code/hooks) on the `UserPromptSubmit` event. Every time you press Enter, before Claude processes your message, Focus Gate executes and prints a context block to stdout.

```
You type a prompt
       |
       v
+------------------+
| UserPromptSubmit  | -- Claude Code fires the hook
|      Hook         |
+--------+---------+
         |
         v
+------------------+
|   Focus Gate     | -- Reads prompt from stdin (JSON)
|   (focus-gate)   | -- Compares against intent forest
|                  | -- Updates trees, prunes if needed
|                  | -- Writes context to stdout
+--------+---------+
         |
         v
+------------------+
|   Claude Code    | -- Receives your prompt + Focus context
|                  | -- Processes with enriched awareness
+------------------+
```

### The Forest

Your conversation intent is stored as a **forest** — a collection of trees. Each tree represents a distinct topic. Within a tree, nodes represent individual prompts or sub-topics, organized hierarchically.

```
Forest
+-- Tree: "authentication | session | token"
|   +-- "add JWT authentication to the API"
|   +-- "fix the session expiry bug"
|   +-- "add refresh token rotation"
|
+-- Tree: "database | migration | schema"
    +-- "create users table migration"
    +-- "add index on email column"
```

Each new prompt is classified by **TF-IDF cosine similarity**:

| Similarity Score | Action | Meaning |
|:---:|:---:|:---|
| **>= 0.55** | **Extend** | Very related to an existing leaf — add as sibling |
| **0.25 - 0.55** | **Branch** | Related to a tree's theme — add under root |
| **< 0.25** | **New Tree** | Unrelated to anything — start a new topic |

### Markov Chain

A **Markov chain** tracks topic-to-topic transitions. When you repeatedly switch between topics in a pattern (e.g. auth -> database -> frontend), the chain learns this and boosts the likely next topic during classification:

```
score = cosine_similarity * (1 + alpha * P(tree | last_topic))
```

The multiplicative form ensures that a zero-similarity prompt cannot match a tree through transition history alone — Markov only amplifies existing content similarity, acting as a tiebreaker between genuinely related trees.

`alpha` defaults to 0.2. A prediction line appears in the context output when the top transition probability exceeds 30%:

```
  -> next: database migration (78%)
```

### Self-Cleaning

The forest has a configurable memory limit (default: 100 nodes). When it fills up, the system **prunes** by removing the lowest-scoring leaves first. Scores combine three factors:

- **Weight**: How many times this topic has been revisited (logarithmic growth)
- **Recency**: Exponential decay based on time since last access
- **Depth**: Deeper nodes are slightly less valuable than shallow ones

Topics you keep revisiting stay. Topics you mentioned once hours ago fade away.

Nodes carry an **indexed** flag that tracks whether their content was registered with the TF-IDF engine. Only real user-prompt nodes are indexed; synthetic bubble-up abstractions are not. During pruning, only indexed content triggers `RemoveDocument`, preventing document-frequency counters from drifting over long sessions.

---

## Install

Download the binary for your platform from [Releases](https://github.com/kuandriy/focus-gate/releases), or build from source:

```bash
go build -o focus-gate ./cmd/focus
```

---

## Usage

### As a Claude Code Hook

Add to `.claude/settings.local.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/focus-gate"
          }
        ]
      }
    ]
  }
}
```

### CLI

```bash
# Show current forest state
./focus-gate --status

# Reset all tracking data
./focus-gate --reset

# Inspect full internal state (forest, TF-IDF, guide, Markov)
./focus-gate --inspect

# Inspect with JSON output for programmatic analysis
./focus-gate --inspect --json

# Dry-run: classify a prompt without modifying any state
./focus-gate --dry-run "your prompt text here"

# Dry-run with JSON output
./focus-gate --dry-run "your prompt text" --json

# Process a prompt (hook mode, reads JSON from stdin)
echo '{"prompt":"your prompt text"}' | ./focus-gate
```

#### Observability

**`--inspect`** dumps the complete internal state in a single view: all forest trees with their full node hierarchy (IDs, depth, weight, frequency, indexed flag, decay score), TF-IDF corpus statistics (total documents, top terms by document frequency), guide entries with reinforcement state, and the Markov transition matrix with probabilities. Add `--json` for machine-readable output.

**`--dry-run "prompt"`** runs the full classification pipeline — tokenization, TF-IDF vectorization, cosine similarity against every root and leaf, multiplicative Markov boost — and shows exactly what would happen, without mutating any state. The output includes per-tree scoring breakdown and the predicted action (new / branch / extend). Useful for verifying threshold tuning and understanding classification decisions.

### Context Output

The injected context looks like this:

```
[Focus | 12 prompts | 8/100 mem | 3 trees]
  [0.95] token | authentica | session | jwt
    - add refresh token rotation
    - fix the session expiry bug
  [0.82] database | migration | schema
    - add index on email column
  [0.45] readme | documentation | project
  -> next: database migration (78%)
Guide:
  - Implemented JWT auth with RS256 signing
  - Created users migration with email index
[/Focus]
```

Trees are sorted by score (highest first), limited to 5. Each tree shows up to 3 recent leaves. The entire output is capped at `contextLimit` characters (default 600).

### Bidirectional Guide Reinforcement

The Guide doesn't just display past AI responses — it feeds them back into the forest. Before each prompt is classified, unreinforced guide entries are tokenized, vectorized, and matched against tree roots by cosine similarity. The best-matching root is **touched** (weight and recency increase), making actively-discussed trees stickier and harder to prune.

This means both user prompts and AI responses shape the intent forest. When you ask about "authentication" and the AI responds about "JWT token rotation," that response reinforces the authentication tree. Each entry is marked as reinforced after processing, so it is never double-counted.

---

## Algorithms

### TF-IDF Vectorization

[TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) converts text into numerical vectors where each dimension represents a term's importance.

- **Term Frequency (TF)**: `count(term in doc) / length(doc)`
- **Inverse Document Frequency (IDF)**: `log2(1 + totalDocs / df(term))` — rare terms score higher
- **TF-IDF**: `TF * IDF`

### Cosine Similarity

Two TF-IDF vectors are compared using the cosine of the angle between them. Implemented as a merge-join over sorted sparse vectors — O(n+m) time, zero allocations.

- **1.0** = identical topic
- **0.0** = completely unrelated

This metric is magnitude-independent — a short prompt and a long one will score high similarity if they share key terms.

### Classification

Uses a two-level comparison:

1. Compare prompt vector against each tree's **root** (catches broad thematic matches)
2. Compare against each tree's **leaves** (catches precise matches)
3. Multiply by Markov transition boost per tree
4. Best score determines action (extend / branch / new)

Node vectors are **cached** after first computation and invalidated when content changes (bubble-up) or when a new document shifts IDF weights. This avoids re-tokenizing and re-vectorizing every node on every prompt.

### Stemmer

A lightweight two-pass suffix stemmer:

- **Pass 1**: Strip plurals (`-ies` -> `-y`, `-es` -> strip, `-s` -> strip)
- **Pass 2**: Strip one derivational suffix (longest match: `-ization`, `-tion`, `-ment`, `-ing`, `-ed`, etc.)

`"er"` is intentionally excluded — too many root words end in "er" (container, server, docker) causing false conflation.

### Bubble-Up Abstraction

After any tree modification, parent node content is regenerated bottom-up. Leaf nodes hold actual prompt text; parents hold the top N most frequent terms across their children, pipe-separated:

```
Children:                          Parent becomes:
  "add JWT authentication"         "token | jwt | authentica | session"
  "fix session expiry bug"
  "add refresh token rotation"
```

### Decay Scoring

```
score = weight * recency * depthFactor

weight      = log2(frequency + 1)
recency     = e^(-decayRate * ageHours)
depthFactor = 1 / (1 + depth * 0.15)
```

At default decay rate (0.05), a node untouched for 24 hours retains 30% recency. After 48 hours: 9%.

---

## Configuration

Create a `config.json` alongside the binary:

```json
{
  "memorySize": 100,
  "decayRate": 0.05,
  "similarity": { "extend": 0.55, "branch": 0.25 },
  "contextLimit": 600,
  "bubbleUpTerms": 6,
  "maxSourcesPerNode": 20,
  "guideSize": 15,
  "transitionBoost": 0.2
}
```

Only fields present in the file override defaults. A field explicitly set to `0` (e.g. `"transitionBoost": 0` to disable Markov boost) is respected — it will not be replaced with the default.

| Parameter | Default | Description |
|:---|:---:|:---|
| `memorySize` | 100 | Maximum total nodes across all trees |
| `decayRate` | 0.05 | Exponential decay rate per hour. Higher = faster forgetting |
| `similarity.extend` | 0.55 | Threshold to extend an existing leaf |
| `similarity.branch` | 0.25 | Threshold to branch into an existing tree |
| `contextLimit` | 600 | Maximum characters in the context block |
| `bubbleUpTerms` | 6 | Top terms in bubble-up abstractions |
| `maxSourcesPerNode` | 20 | Maximum source IDs stored per node |
| `guideSize` | 15 | Maximum AI response entries tracked |
| `transitionBoost` | 0.2 | Markov chain boost factor (0 to disable) |

### Tuning

- **Too many unrelated trees?** Raise `similarity.branch` (e.g. 0.35)
- **Related prompts keep splitting?** Lower `similarity.branch` (e.g. 0.20)
- **Old topics persist too long?** Raise `decayRate` (e.g. 0.10)
- **Memory fills too quickly?** Raise `memorySize` (e.g. 200)

---

## Architecture

```
cmd/focus/          Entry point (CLI, stdin/stdout, inspect/dry-run)
internal/
  text/             Tokenizer, stemmer, stop words
  tfidf/            TF-IDF engine, sparse vectors, cosine similarity
  forest/           Node, Tree, Forest, heap-based pruning
  gate/             Focus Gate classifier (classify, apply, bubble-up, dry-run)
  markov/           Topic transition chain (prediction, boost)
  guide/            AI response tracking (ring buffer + forest reinforcement)
  persist/          Atomic JSON persistence (Windows-safe, .tmp recovery)
```

Data is persisted as JSON in a `data/` directory alongside the binary. Writes use **atomic save** (write to `.tmp`, then rename). On Windows, where `os.Rename` is not atomic, the target is removed before rename; a **recovery pass** on startup promotes any orphaned `.tmp` files left by interrupted saves.

All `persist.Load` errors are logged to stderr rather than silently discarded — a corrupt file does not block the user's prompt; the system continues with empty state and the user can `--reset` if needed.

| File | Purpose |
|:---|:---|
| `data/intent.json` | Intent forest — what the user is asking about |
| `data/engine.json` | TF-IDF document frequency counts |
| `data/guide.json` | AI response summaries with intent links and reinforcement state |
| `data/markov.json` | Topic transition probability matrix |

---

## License

MIT
