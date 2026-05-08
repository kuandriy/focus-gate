# The Sliding-Window Intent Forest

**Stage 1 of Focus Gate's organic refinement pipeline. Where every prompt and every reply is classified, scored, and silently sharpened — and where most of what you say is allowed to fade.**

---

## Table of Contents

- [The story](#the-story)
- [How a prompt flows through it](#how-a-prompt-flows-through-it)
- [The hook](#the-hook)
- [The forest](#the-forest)
- [Classification](#classification)
- [Continuation for short prompts](#continuation-for-short-prompts)
- [Session boundaries](#session-boundaries)
- [Cluster merging](#cluster-merging)
- [Self-cleaning (pruning)](#self-cleaning-pruning)
- [Bidirectional guide reinforcement](#bidirectional-guide-reinforcement)
- [Algorithms](#algorithms)
- [File reference extraction](#file-reference-extraction)
- [Typo tolerance](#typo-tolerance)
- [Configuration](#configuration)
- [In-chat inspection](#in-chat-inspection)
- [Architecture and persistence](#architecture-and-persistence)

---

## The story

You start a coding session. You ask about authentication. Two prompts later you're knee-deep in JWT key rotation. Five prompts later you remember the migration script you never finished and pivot to that. Twenty prompts later you're back on auth, a refresh-token bug this time. By prompt forty, your context window has compressed half the conversation into vague summaries, and the assistant has forgotten which auth file you were even editing.

The Sliding-Window Intent Forest is the answer to that. It is not a transcript. It is not a database. It is a **small, opinionated model of your attention right now** — a forest of topic trees, each tree a hill you've been climbing, each leaf a prompt or AI reply that pulled you up that hill. It refreshes from both ends: your prompts shape it, and the assistant's replies feed back to reinforce the trees you're actively working on.

It has one mathematical job — to classify your next prompt against your accumulated intent and decide whether it continues an existing climb or starts a new one — and one practical purpose: to write a compact block of context to the assistant on every turn so it never has to guess what you've been focused on.

It is also the first refinery in Focus Gate's pipeline. Topics you keep returning to gain weight. Topics you mention once fade with time. When the forest fills up, the lowest-scoring leaves are quietly pruned. The survivors — the topics that actually mattered enough to be revisited — are exactly the candidates that the [Memory Layer](memory-focus.md) considers crystallizing into long-term stories.

Nothing here is prescribed. The forest is a model of *what you have done*, not a list of *what you should do*. The user's attention is the only signal that matters; the math just makes it durable across a session.

---

## How a prompt flows through it

1. Read the prompt from the hook's stdin.
2. Tokenize, stem, canonicalize typos, vectorize against the live TF-IDF corpus.
3. Compare against everything you've asked before — root-level then leaf-level.
4. Decide: extend an existing leaf, branch under an existing tree, or start a new tree. (For terse prompts: continue the most recent tree.)
5. Update the forest — bump weights, recompute parent abstractions, run a single cluster-merge pass if two trees converged.
6. Match the prompt against the [Memory Layer's](memory-focus.md) index — surface pointers to any prior stories whose time markers, interests, topics, or assets line up.
7. Emit a compact context block to the assistant: current focus + relevant memory pointers.

The assistant never sees the machinery. It receives your prompt enriched with a small block that says *here's what the user has been focused on, and here are stories that may apply.*

---

## The hook

Focus Gate runs as a [Claude Code hook](https://docs.anthropic.com/en/docs/claude-code/hooks) on the `UserPromptSubmit` event. Every time you press Enter, before Claude processes your message, Focus Gate executes and prints a context block to stdout.

```
You type a prompt
       |
       v
+-------------------+
| UserPromptSubmit  | -- Claude Code fires the hook
|       Hook        |
+--------+----------+
         |
         v
+-------------------+
|   Focus Gate      | -- Reads prompt from stdin (JSON)
|   (focus-gate)    | -- Compares against intent forest
|                   | -- Updates trees, prunes if needed
|                   | -- Writes context to stdout
+--------+----------+
         |
         v
+-------------------+
|   Claude Code     | -- Receives your prompt + Focus context
|                   | -- Processes with enriched awareness
+-------------------+
```

---

## The forest

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

Roots hold abstract topic vectors (computed bottom-up from their leaves). Leaves hold the actual prompts. The shape is the model.

---

## Classification

Each new prompt is classified by **TF-IDF cosine similarity** against every existing root and leaf, in two passes:

1. Compare the prompt vector against every tree's **root** (catches broad thematic matches).
2. Compare against every tree's **leaves** (catches precise matches).
3. The best score across both passes determines the action.

| Similarity score | Action | Meaning |
|:---:|:---:|:---|
| **≥ 0.55** | **Extend** | Very related to an existing leaf — add as a sibling under the same parent |
| **0.25 – 0.55** | **Branch** | Related to a tree's theme but not to a specific leaf — add directly under the root |
| **< 0.25** | **New tree** | Unrelated to anything — start a new topic |

Node vectors are **cached** after first computation and invalidated when content changes (bubble-up) or when a new document shifts IDF weights. This avoids re-tokenizing and re-vectorizing every node on every prompt.

---

## Continuation for short prompts

When you send a terse prompt that tokenizes to nothing meaningful — e.g. `"fix"`, `"yes"`, `"continue"`, `"run it"` — cosine similarity is zero against every tree. Rather than spawning a noise tree, Focus Gate attaches such prompts to the **most recently active tree** as a continuation leaf. This preserves context for follow-ups and keeps the forest clean.

If no tree exists yet, a terse prompt is skipped (no new tree is created).

---

## Session boundaries

After `sessionTimeout` hours of inactivity (default 4h), a session boundary fires on the next prompt. Every node's frequency is halved (minimum 1) so an old session doesn't dominate scoring for the new one. Set `sessionTimeout` to `0` to disable.

This is a soft reset. State is not erased — old trees still exist, still have their content, still match against new prompts. They just lose half their accumulated weight, so a yesterday-tree doesn't outrank a today-tree on inertia alone.

---

## Cluster merging

After each prompt, if two tree roots are semantically close (cosine ≥ `mergeSimilarity`, default 0.6), the smaller tree is merged into the larger one and bubble-up is re-run. This prevents slow fragmentation when related prompts keep spawning sibling trees. One merge per prompt to bound the cost; repeated similar prompts will converge over a few rounds.

---

## Self-cleaning (pruning)

The forest has a configurable memory limit (default: 100 nodes). When it fills up, the system **prunes** by removing the lowest-scoring leaves first. Scores combine three factors:

- **Weight**: How many times this topic has been revisited (logarithmic growth).
- **Recency**: Exponential decay based on time since last access.
- **Depth**: Deeper nodes are slightly less valuable than shallow ones.

Topics you keep revisiting stay. Topics you mentioned once hours ago fade away.

Pruning builds a min-heap **once**, then pops entries in a loop with **parent cascading** — when a leaf is removed and its parent becomes a new leaf (and is not a root), the parent is pushed onto the heap as a pruning candidate. Trees are tracked by stable ID rather than slice index, so removals mid-loop don't corrupt references.

Nodes carry an **indexed** flag that tracks whether their content was registered with the TF-IDF engine. Only real user-prompt nodes are indexed; synthetic bubble-up abstractions are not. During pruning, only indexed content triggers `RemoveDocument`, preventing document-frequency counters from drifting over long sessions.

Before a tree is fully pruned, the [Memory Layer](memory-focus.md) inspects it as a **promotion candidate**. Trees with enough leaves, enough touches, and a high enough peak score are queued for crystallization into a long-term story instead of being silently discarded. Pruning, in other words, is also Stage-1 → Stage-2 hand-off.

> **The forest scores attention, not importance.** Frequency × recency × depth tells you what the user has been working on, not whether the work was right, generalizable, or worth remembering. That's deliberate. Importance judgment is deferred to the promote-time LLM (which sees the candidate plus the full aggregate index of existing memories and decides append/create/discard) and to the recall-time LLM (which sees pointer rows with a `t:N` cornerstone count and decides what's worth opening). The forest's job is to make sure no high-attention topic is silently lost; deciding what's important about that topic happens later, with full context. See [memory-focus.md §"Why we don't score importance"](memory-focus.md#why-we-dont-score-importance) for the long-form rationale.

---

## Bidirectional guide reinforcement

The Guide doesn't just display past AI responses — it feeds them back into the forest. Before each prompt is classified, unreinforced guide entries are tokenized, vectorized, and matched against tree roots by cosine similarity. The best-matching root is **touched** (weight and recency increase), making actively-discussed trees stickier and harder to prune.

This means both user prompts and AI responses shape the intent forest. When you ask about "authentication" and the AI responds about "JWT token rotation," that response reinforces the authentication tree. Each entry is marked as reinforced after processing, so it is never double-counted across restarts.

The same Guide entries are also carried forward into a tree's promotion candidate when the tree is about to be pruned, so the Memory Layer's LLM review sees the full story-shape of the topic — your prompts AND the assistant's replies — not just one half of the dialogue.

---

## Algorithms

### TF-IDF vectorization

[TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) converts text into numerical vectors where each dimension represents a term's importance.

- **Term Frequency (TF)**: `count(term in doc) / length(doc)`
- **Inverse Document Frequency (IDF)**: `log2(1 + effectiveDocs / df(term))` — rare terms score higher. `effectiveDocs` is `max(totalDocs, 5)` — a virtual floor that ensures IDF can discriminate between terms even during the first few prompts of a session, when the corpus is too small for meaningful frequency statistics.
- **TF-IDF**: `TF * IDF`

### Cosine similarity

Two TF-IDF vectors are compared using the cosine of the angle between them. Implemented as a merge-join over sorted sparse vectors — O(n+m) time, zero allocations.

- **1.0** = identical topic
- **0.0** = completely unrelated

This metric is magnitude-independent — a short prompt and a long one will score high similarity if they share key terms.

### Stemmer

A lightweight two-pass suffix stemmer with an override map for known false conflations:

- **Override map**: Checked first — prevents mechanical suffix stripping from producing unrelated roots (e.g. "authorization" → "author", "organization" → "organ"). Overridden words stem to a consistent form ("authoriz", "organiz") that groups related variants correctly.
- **Pass 1**: Strip plurals (`-ies` → `-y`, `-es` → strip, `-s` → strip).
- **Pass 2**: Strip one derivational suffix (longest match: `-ization`, `-tion`, `-ment`, `-ing`, `-ed`, etc.).

`"er"` is intentionally excluded — too many root words end in "er" (container, server, docker) causing false conflation.

### Bubble-up abstraction

After any tree modification, parent node content is regenerated bottom-up. Leaf nodes hold actual prompt text; parents hold the top N terms across their children, pipe-separated, scored by **presence × IDF**:

- **Presence**: How many children contain the term (not raw frequency — a term in 3 of 4 children scores higher than a term repeated 5 times in 1 child).
- **IDF**: Inverse document frequency from the TF-IDF engine — suppresses corpus-common terms like "add" or "fix" that survive stop-word filtering, promoting distinctive topic terms.

```
Children:                          Parent becomes:
  "add JWT authentication"         "jwt | token | authentica | session"
  "fix session expiry bug"
  "add refresh token rotation"
```

### Decay scoring

```
score = weight * recency * depthFactor

weight      = log2(frequency + 1)
recency     = e^(-decayRate * ageHours)
depthFactor = 1 / (1 + depth * 0.15)
```

At default decay rate (0.05), a node untouched for 24 hours retains 30% recency. After 48 hours: 9%. Frequency only ever climbs (when the node is touched again); recency only ever falls (until the next touch resets it). The product of the two is what the heap-based pruner orders by.

---

## File reference extraction

Every prompt is scanned for file paths — either explicit (`src/auth/middleware.go`) or backtick-quoted (`` `handler.ts` ``). URLs and Go-style package imports (domain-looking paths) are filtered out. Paths are validated against the project's working directory; non-existent paths are dropped, so references reflect actual code, not rhetorical mentions. The resulting refs are attached to the node and surface in the injected context so the AI knows which files are in scope.

---

## Typo tolerance

When you type `envaeronment` after having already said `environment` a few times, Focus Gate canonicalizes the new token to the established one using Levenshtein edit distance. The canonicalization happens at tokenize time — the typo never enters the TF-IDF corpus, so `engine.json` stays clean and repeated typos of the same word keep remapping to the same canonical stem. No spell-check dictionary, no external dependencies.

Guards:

- Only tokens **shorter than `minWordLen`** characters (default 5) are considered — short words like `auth`, `each`, `run` are too prone to false merges.
- Only rewrites toward terms **already seen `minEstablishedDF` times** (default 3) — a single-occurrence term cannot become a canonical attractor.
- **Maximum edit distance** (default 2) is tight enough to catch realistic typos (`dificult` → `difficult`, `envaeron` → `environ`) without merging genuinely different words.

Set `"typoTolerance": { "enabled": false }` to disable the feature completely and restore the original tokenizer behaviour.

---

## Configuration

Configuration resolution order (first match wins):

1. `.focus-gate.json` in the current project directory
2. `$FOCUS_GATE_CONFIG` — explicit path via environment variable
3. `config.json` alongside the binary (global fallback)

Forest-relevant keys (memory-layer keys are documented in [docs/memory-focus.md](memory-focus.md)):

| Parameter | Default | Description |
|:---|:---:|:---|
| `memorySize` | 100 | Maximum total nodes across all trees |
| `decayRate` | 0.05 | Exponential decay rate per hour. Higher = faster forgetting |
| `similarity.extend` | 0.55 | Threshold to extend an existing leaf |
| `similarity.branch` | 0.25 | Threshold to branch into an existing tree |
| `contextLimit` | 600 | Maximum characters in the context block |
| `bubbleUpTerms` | 6 | Top terms in bubble-up abstractions |
| `maxRefsPerNode` | 5 | Maximum file references stored per node |
| `guideSize` | 15 | Maximum AI response entries tracked |
| `sessionTimeout` | 4.0 | Hours of inactivity before session boundary halves frequencies. `0` disables. |
| `mergeSimilarity` | 0.6 | Cosine threshold for cluster merging of similar trees. `0` disables. |
| `typoTolerance.enabled` | `true` | Rewrite novel tokens to the nearest established corpus term. `false` = no canonicalization. |
| `typoTolerance.maxDistance` | 2 | Max Levenshtein edits between a novel token and the term it may be rewritten to. |
| `typoTolerance.minWordLen` | 5 | Tokens shorter than this (after stemming) are never rewritten. |
| `typoTolerance.minEstablishedDF` | 3 | A corpus term must appear in at least this many prior prompts before it can be a rewrite target. |

Only fields present in the file override defaults. A field explicitly set to `0` (e.g. `"sessionTimeout": 0` to disable session boundaries) is respected — it will not be replaced with the default.

### Tuning

- **Too many unrelated trees?** Raise `similarity.branch` (e.g. 0.35).
- **Related prompts keep splitting?** Lower `similarity.branch` (e.g. 0.20) or lower `mergeSimilarity` (e.g. 0.5).
- **Old topics persist too long?** Raise `decayRate` (e.g. 0.10) or lower `sessionTimeout`.
- **Memory fills too quickly?** Raise `memorySize` (e.g. 200).
- **Typos creating duplicate trees?** Raise `typoTolerance.maxDistance` to 3 if you spell aggressively wrong; lower `minEstablishedDF` to 2 if your sessions are short.
- **Typos silently merging unrelated words?** Raise `minWordLen` to 6 or `minEstablishedDF` to 5, or lower `maxDistance` to 1.
- **Unbounded growth concern?** If `sessionTimeout=0` *and* `memorySize` is very high, `engine.json` can grow indefinitely because pruning is what limits the TF-IDF corpus. Keep at least one of the two bounded.

---

## In-chat inspection

Type any `/focus` command (CLI) or its `fg:` alias (works in every environment, including the VSCode extension where the `/` slash picker would otherwise intercept the command) directly in chat. The command is routed to the inspector before classification — **no state is modified**, the output appears inline as a fenced block.

| CLI form | Alias | Description |
|:---|:---|:---|
| `/focus status` | `fg: status` | Compact context summary (same output the AI normally sees) |
| `/focus inspect` | `fg: inspect` | Full state dump — forest hierarchy, TF-IDF engine, guide |
| `/focus tree` | `fg: tree` | List all trees with scores |
| `/focus tree 0` | `fg: tree 0` | Deep-dive into tree #0 — full node hierarchy, vector terms, pruning candidates |
| `/focus tree abc123` | `fg: tree abc123` | Deep-dive by partial tree ID |
| `/focus terms` | `fg: terms` | TF-IDF vocabulary — top 30 terms with DF and IDF values |
| `/focus terms 50` | `fg: terms 50` | Show top 50 terms |
| `/focus last` | `fg: last` | Last classifications (action + similarity score) |
| `/focus score "prompt"` | `fg: score "prompt"` | Dry-run classification — see how a prompt would be scored without sending it |
| `/focus health` | `fg: health` | System diagnostics — memory pressure, tree balance, staleness, pruning forecast |
| `/focus help` | `fg: help` | List all available commands |

> **Two invocation paths.** `/focus <sub>` is a registered Claude Code slash command defined in [.claude/commands/focus.md](../.claude/commands/focus.md); it runs the binary in `--cmd` mode and returns output inline. `fg: <sub>` is intercepted by the `UserPromptSubmit` hook itself and works in any environment where the hook runs, with zero custom-command setup. Both route to the same handler.

### Example: `fg: status`

Compact summary of the current forest state — the same block the AI sees as injected context on every prompt.

```
[Focus | 23 prompts | 18/100 mem | 3 trees]
  [0.95] token | authentica | session | jwt
    - add refresh token rotation
    - fix the session expiry bug
  [0.82] migrat | schema | user | email
    - add index on email column
  [0.45] readme | documentation | project
Guide:
  - Implemented JWT auth with RS256 signing
  - Created users migration with email index
[/Focus]
```

Each tree's `[0.95]`-style prefix is its decay-weighted score. Leaves are shown most-recent-first, truncated to 3 per tree, and the whole block is capped at `contextLimit` characters (default 600). The `Guide:` section lists AI response summaries linked to still-alive intent nodes.

### Example: `fg: health`

System diagnostics — memory pressure, per-tree temperature, TF-IDF noise ratio, and a forecast of which leaves are closest to being pruned.

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
```

HOT/WARM/COLD thresholds: HOT ≥ 0.5 (actively worked on), WARM 0.1–0.5 (recent but cooling), COLD < 0.1 (likely to be pruned soon). Noise ratio rising above ~50% is a signal that your session has accumulated many one-off typos or cryptic prompts.

### Example: `fg: last`

Ring buffer of the most recent classifications — action taken, top similarity score, and the prompt snippet that triggered each. Useful when a prompt lands in an unexpected tree.

```
=== Recent Classifications ===
  #5  extend    0.618  "add refresh token rotation"      -> tree#0
  #4  branch    0.312  "write migration for users table" -> tree#1
  #3  new       0.000  "update the readme section"       -> tree#2
  #2  continue  0.000  "yes"                             -> tree#0 (recent)
  #1  extend    0.554  "fix the session expiry bug"      -> tree#0
```

The `continue` entry shows the short-prompt behaviour in action: a terse prompt ("yes") was attached to the most-recently-active tree rather than spawning its own noise tree.

### Example: `fg: tree 0`

Deep-dive into a specific tree. Without an argument, `fg: tree` lists all trees with scores; with a numeric index or partial hex ID, the full node hierarchy appears.

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

Node columns: `d` = depth, `w` = log₂ weight, `f` = frequency (times touched), `idx` = registered in the TF-IDF corpus (Y = real prompt, − = synthetic bubble-up abstraction), `s` = decay-weighted score.

### Example: `fg: score "prompt"`

Dry-run classification. Shows how the prompt would be scored against every tree without mutating any state — and flags **near-misses** (scores in `[branch × 0.5, branch)`) that often indicate a typo or rare-term mismatch.

```
=== Score ===
  Prompt: "fx the auth middleware"
  Tokens: [fx auth middlewar]

  TF-IDF Vector (2 terms):
    authentica           0.4821
    middlewar            0.3912

  Thresholds: extend >= 0.550, branch >= 0.250

  Tree #0 "token | authentica | session | jwt"
    Root a1b2c3d4  cosine=0.2103
    Leaf b2c3d4e5  cosine=0.1918  "add JWT authentication to the API"  <- BEST

  Result: new (score=0.2103)
    Would create a new topic tree with this prompt.

Near-miss: top score 0.2103 fell short of branch threshold 0.250.
  A typo, missing stem, or rare-term mismatch may have prevented a match
  you were expecting. Check /focus tree N to see the tree's vector terms.
```

The near-miss hint is strictly diagnostic — behaviour is unchanged. Re-running the prompt with the typo fixed ("fix the auth middleware") would likely cross the branch threshold and extend the existing tree.

---

## Architecture and persistence

```
cmd/focus/          Entry point (CLI, stdin/stdout, inspect/dry-run, /focus commands)
internal/
  text/             Tokenizer, stemmer, stop words, file-ref extraction, typo canonicalizer
  tfidf/            TF-IDF engine, sparse vectors, cosine similarity
  forest/           Node, Tree, Forest, heap-based pruning, peak-score tracking
  gate/             Focus Gate classifier (classify, apply, bubble-up, merge, dry-run)
  guide/            AI response tracking (ring buffer + forest reinforcement)
  memory/           Long-term memory (see docs/memory-focus.md)
  persist/          Atomic JSON + raw-bytes persistence, schema version, flock
```

Data is persisted as JSON in a per-project data directory. Each project (keyed by `sha256(cwd)[:12]`) gets its own namespace under `~/.focus-gate/<slug>/`, so state never leaks between projects. Override with `--data-dir` or `$FOCUS_GATE_DATA_DIR`.

Writes use **atomic save** (write to `.tmp`, then rename). On Windows, where `os.Rename` is not atomic, the target is removed before rename; a **recovery pass** on startup promotes any orphaned `.tmp` files left by interrupted saves.

Concurrent invocations of the hook acquire a file lock on the data directory to prevent races during simultaneous state reads/writes.

Every persisted file carries a `schemaVersion` field. Loaders reject mismatched versions with a warning to stderr and fall back to empty state rather than corrupting data with partial unmarshaling.

All `persist.Load` errors are logged to stderr rather than silently discarded — a corrupt file does not block the user's prompt; the system continues with empty state and the user can `--reset` if needed.

| File | Purpose |
|:---|:---|
| `intent.json` | Intent forest — what the user is asking about |
| `engine.json` | TF-IDF document frequency counts |
| `guide.json` | AI response summaries with intent links and reinforcement state |
| `pending_memories.json` | Append-only queue of candidates awaiting LLM promotion (Stage 1 → Stage 2 hand-off) |
| `memories/` | Long-term memory Markdown files + `index.json` manifest (Stage 2 home — see [docs/memory-focus.md](memory-focus.md)) |
