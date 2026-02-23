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
  - [In-Chat Commands](#in-chat-commands)
  - [CLI Flags](#cli-flags)
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

**Self-transitions are skipped** — when consecutive prompts hit the same tree, no transition is recorded. This prevents `P(A|A)` from inflating and creating redundant stickiness on top of the existing recency/decay mechanism. The prediction line shows genuinely predicted topic *switches*, not the current topic.

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

Pruning builds a min-heap **once**, then pops entries in a loop with **parent cascading** — when a leaf is removed and its parent becomes a new leaf (and is not a root), the parent is pushed onto the heap as a pruning candidate. Trees are tracked by stable ID rather than slice index, so removals mid-loop don't corrupt references.

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

### In-Chat Commands

Type any `/focus` command directly in your Claude Code conversation. The command is intercepted before classification — **no state is modified**, the output appears inline as context.

| Command | Description |
|:---|:---|
| `/focus status` | Compact context summary (same output the AI normally sees) |
| `/focus inspect` | Full state dump — forest hierarchy, TF-IDF, guide, Markov |
| `/focus tree` | List all trees with scores |
| `/focus tree 0` | Deep-dive into tree #0 — full node hierarchy, vector terms, pruning candidates |
| `/focus tree abc123` | Deep-dive by partial tree ID |
| `/focus terms` | TF-IDF vocabulary — top 30 terms with DF and IDF values |
| `/focus terms 50` | Show top 50 terms |
| `/focus markov` | Transition matrix with probabilities |
| `/focus score "prompt"` | Dry-run classification — see how a prompt would be scored without sending it |
| `/focus health` | System diagnostics — memory pressure, tree balance, staleness, pruning forecast |
| `/focus help` | List all available commands |

#### Example: `/focus health`

```
=== Focus Health ===

  Memory:  42/100 nodes (42%) [█████░░░░░░░]
  Trees:   3 (nodes per tree: min=8 avg=14.0 max=22, max depth=3)
  Prompts: 42

  TF-IDF:  38 docs, 127 unique terms
           31 terms with df=1 (noise: 24%)

  Tree activity:
    #0 [HOT]   score=0.952  age=5m   "authentication and JWT tokens"
    #1 [WARM]  score=0.614  age=2.3h "database migration schema"
    #2 [COLD]  score=0.089  age=1.2d "readme documentation project"

  Pruning forecast (lowest-scoring leaves):
    [PRUNE?] tree#2 a1b2c3d4  score=0.0312  "update project description"
    [PRUNE?] tree#2 e5f6g7h8  score=0.0487  "add license section"
    [PRUNE?] tree#1 i9j0k1l2  score=0.1205  "add index on email column"

    58 slots remaining before pruning triggers.

  Markov:  2 topics tracked, last=a1b2c3d4 (authentication...)
```

#### Example: `/focus tree 0`

```
=== Tree: a1b2c3d4e5f6 ===
  Nodes: 4, Leaves: 3
  Created:  2026-02-22 10:03:15
  Accessed: 2026-02-22 14:22:08
  Root score: 0.952

  [root] a1b2c3d4  d=0 w=1.58 f=2 idx=- s=0.952
  "token | authentica | session | jwt"
  ├── b2c3d4e5  d=1 w=1.00 f=1 idx=Y s=0.871
  │   "add JWT authentication to the API"
  ├── c3d4e5f6  d=1 w=1.00 f=1 idx=Y s=0.843
  │   "fix the session expiry bug"
  └── d4e5f6g7  d=1 w=1.58 f=2 idx=Y s=0.921
      "add refresh token rotation"

  Root vector terms:
    token                0.4821
    authentica           0.3912
    session              0.3654
    jwt                  0.3201

  Pruning candidates (lowest score first):
    [PRUNE?] c3d4e5f6  score=0.843  "fix the session expiry bug"
    [PRUNE?] b2c3d4e5  score=0.871  "add JWT authentication to the API"
```

### CLI Flags

```bash
# Show current context (same as /focus status)
./focus-gate --status

# Reset all tracking data
./focus-gate --reset

# Full state dump (same as /focus inspect)
./focus-gate --inspect

# Full state dump as JSON
./focus-gate --inspect --json

# Dry-run classification (same as /focus score)
./focus-gate --dry-run "your prompt text here"

# Dry-run as JSON
./focus-gate --dry-run "your prompt text" --json

# Process a prompt (hook mode, reads JSON from stdin)
echo '{"prompt":"your prompt text"}' | ./focus-gate
```

The CLI flags are useful for scripting and programmatic analysis. For day-to-day debugging, the in-chat `/focus` commands are more convenient — no terminal switching required.

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
- **Inverse Document Frequency (IDF)**: `log2(1 + effectiveDocs / df(term))` — rare terms score higher. `effectiveDocs` is `max(totalDocs, 5)` — a virtual floor that ensures IDF can discriminate between terms even during the first few prompts of a session, when the corpus is too small for meaningful frequency statistics
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

A lightweight two-pass suffix stemmer with an override map for known false conflations:

- **Override map**: Checked first — prevents mechanical suffix stripping from producing unrelated roots (e.g. "authorization" → "author", "organization" → "organ"). Overridden words stem to a consistent form ("authoriz", "organiz") that groups related variants correctly.
- **Pass 1**: Strip plurals (`-ies` -> `-y`, `-es` -> strip, `-s` -> strip)
- **Pass 2**: Strip one derivational suffix (longest match: `-ization`, `-tion`, `-ment`, `-ing`, `-ed`, etc.)

`"er"` is intentionally excluded — too many root words end in "er" (container, server, docker) causing false conflation.

### Bubble-Up Abstraction

After any tree modification, parent node content is regenerated bottom-up. Leaf nodes hold actual prompt text; parents hold the top N terms across their children, pipe-separated, scored by **presence × IDF**:

- **Presence**: How many children contain the term (not raw frequency — a term in 3 of 4 children scores higher than a term repeated 5 times in 1 child)
- **IDF**: Inverse document frequency from the TF-IDF engine — suppresses corpus-common terms like "add" or "fix" that survive stop-word filtering, promoting distinctive topic terms

```
Children:                          Parent becomes:
  "add JWT authentication"         "jwt | token | authentica | session"
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
