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
- [Long-Term Memory](#long-term-memory)
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

### Continuation for Short Prompts

When you send a terse prompt that tokenizes to nothing meaningful — e.g. `"fix"`, `"yes"`, `"continue"`, `"run it"` — cosine similarity is zero against every tree. Rather than spawning a noise tree, Focus Gate attaches such prompts to the **most recently active tree** as a continuation leaf. This preserves context for follow-ups and keeps the forest clean.

If no tree exists yet, a terse prompt is skipped (no new tree is created).

### Session Boundaries

After `sessionTimeout` hours of inactivity (default 4h), a session boundary fires on the next prompt. Every node's frequency is halved (minimum 1) so an old session doesn't dominate scoring for the new one. Set `sessionTimeout` to `0` to disable.

### Cluster Merging

After each prompt, if two tree roots are semantically close (cosine ≥ `mergeSimilarity`, default 0.6), the smaller tree is merged into the larger one and bubble-up is re-run. This prevents slow fragmentation when related prompts keep spawning sibling trees. One merge per prompt to bound the cost; repeated similar prompts will converge over a few rounds.

### Self-Cleaning

The forest has a configurable memory limit (default: 100 nodes). When it fills up, the system **prunes** by removing the lowest-scoring leaves first. Scores combine three factors:

- **Weight**: How many times this topic has been revisited (logarithmic growth)
- **Recency**: Exponential decay based on time since last access
- **Depth**: Deeper nodes are slightly less valuable than shallow ones

Topics you keep revisiting stay. Topics you mentioned once hours ago fade away.

Pruning builds a min-heap **once**, then pops entries in a loop with **parent cascading** — when a leaf is removed and its parent becomes a new leaf (and is not a root), the parent is pushed onto the heap as a pruning candidate. Trees are tracked by stable ID rather than slice index, so removals mid-loop don't corrupt references.

Nodes carry an **indexed** flag that tracks whether their content was registered with the TF-IDF engine. Only real user-prompt nodes are indexed; synthetic bubble-up abstractions are not. During pruning, only indexed content triggers `RemoveDocument`, preventing document-frequency counters from drifting over long sessions.

### Bidirectional Guide Reinforcement

The Guide doesn't just display past AI responses — it feeds them back into the forest. Before each prompt is classified, unreinforced guide entries are tokenized, vectorized, and matched against tree roots by cosine similarity. The best-matching root is **touched** (weight and recency increase), making actively-discussed trees stickier and harder to prune.

This means both user prompts and AI responses shape the intent forest. When you ask about "authentication" and the AI responds about "JWT token rotation," that response reinforces the authentication tree. Each entry is marked as reinforced after processing, so it is never double-counted across restarts.

---

## Install

Download the binary for your platform from [Releases](https://github.com/kuandriy/focus-gate/releases), or build from source:

```bash
go build -o focus-gate ./cmd/focus
```

> The binary requires no runtime dependencies. Drop it anywhere on your `$PATH` (e.g. `~/.local/bin/`) or reference it by absolute path.

---

## Usage

### As a Claude Code Hook

Add to `.claude/settings.local.json` in the target project:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/absolute/path/to/focus-gate"
          }
        ]
      }
    ]
  }
}
```

> Focus Gate is a CLI-level hook. It runs in the terminal-based Claude Code session. The VSCode extension does not fire `UserPromptSubmit` hooks, so `/focus` in-chat commands do not reach the binary there — run Focus Gate via the CLI instead.

### In-Chat Commands

Type any `/focus` command (CLI) or the equivalent `fg:` alias (works in every environment, including the VSCode extension where the `/` slash picker would otherwise intercept the command) directly in chat. The command is routed to the inspector before classification — **no state is modified**, the output appears inline as a fenced block.

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
| `/focus memory list` | `fg: memory list` | List long-term memory files with touch counts (see [Long-Term Memory](#long-term-memory)) |
| `/focus memory show <id>` | `fg: memory show <id>` | Pretty-print one memory's frontmatter + body |
| `/focus memory pending` | `fg: memory pending` | Queue of candidates awaiting LLM promotion |
| `/focus memory discard <id\|all>` | `fg: memory discard <id\|all>` | Clear pending candidates without promoting |
| `/focus memory health` | `fg: memory health` | Manifest counts, stale refs, soft warnings |
| `/focus help` | `fg: help` | List all available commands |

> **Two invocation paths.** `/focus <sub>` is a registered Claude Code slash command defined in [.claude/commands/focus.md](.claude/commands/focus.md); it runs the binary in `--cmd` mode and returns output inline. `fg: <sub>` is intercepted by the `UserPromptSubmit` hook itself and works in any environment where the hook runs, with zero custom-command setup. Both route to the same handler. The examples below use `fg:` (it's the bedrock); every output is identical under `/focus`.

#### Example: `fg: status`

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

#### Example: `fg: health`

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

#### Example: `fg: last`

Ring buffer of the most recent classifications — action taken, top similarity score, and the prompt snippet that triggered each. Useful when a prompt lands in an unexpected tree.

```
=== Recent Classifications ===
  #5  extend    0.618  "add refresh token rotation"      -> tree#0
  #4  branch    0.312  "write migration for users table" -> tree#1
  #3  new       0.000  "update the readme section"       -> tree#2
  #2  continue  0.000  "yes"                             -> tree#0 (recent)
  #1  extend    0.554  "fix the session expiry bug"      -> tree#0
```

The `continue` entry shows the M2 behaviour in action: a terse prompt ("yes") was attached to the most-recently-active tree rather than spawning its own noise tree.

#### Example: `fg: tree 0`

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

#### Example: `fg: score "prompt"`

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

# List data directories for all known projects
./focus-gate --list-projects

# Process a prompt (hook mode, reads JSON from stdin)
echo '{"prompt":"your prompt text"}' | ./focus-gate
```

The CLI flags are useful for scripting and programmatic analysis. For day-to-day debugging, the in-chat `/focus` commands are more convenient — no terminal switching required.

### Context Output

The injected context looks like this:

```
[Focus | 12 prompts | 8/100 mem | 3 trees | extend 0.61]
  [0.95] token | authentica | session | jwt
    - add refresh token rotation
    - fix the session expiry bug
  [0.82] database | migration | schema
    - add index on email column
  [0.45] readme | documentation | project
Guide:
  - Implemented JWT auth with RS256 signing
  - Created users migration with email index
[/Focus]
```

The header includes the classification action taken for the current prompt (`extend`, `branch`, `new`, or `continue`) and the similarity score. Trees are sorted by score (highest first), limited to 5. Each tree shows up to 3 recent leaves. The entire output is capped at `contextLimit` characters (default 600).

---

## Algorithms

### TF-IDF Vectorization

[TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) converts text into numerical vectors where each dimension represents a term's importance.

- **Term Frequency (TF)**: `count(term in doc) / length(doc)`
- **Inverse Document Frequency (IDF)**: `log2(1 + effectiveDocs / df(term))` — rare terms score higher. `effectiveDocs` is `max(totalDocs, 5)` — a virtual floor that ensures IDF can discriminate between terms even during the first few prompts of a session, when the corpus is too small for meaningful frequency statistics.
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
3. Best score determines action (extend / branch / new)

If the prompt tokenizes to nothing meaningful (stop words only, or very short), the classifier returns `continue` and attaches the prompt to the most recently active tree.

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

### File Reference Extraction

Every prompt is scanned for file paths — either explicit (`src/auth/middleware.go`) or backtick-quoted (`` `handler.ts` ``). URLs and Go-style package imports (domain-looking paths) are filtered out. Paths are validated against the project's working directory; non-existent paths are dropped, so references reflect actual code, not rhetorical mentions. The resulting refs are attached to the node and surface in the injected context so the AI knows which files are in scope.

---

## Long-Term Memory

Focus Gate's in-session forest is working memory: pruning is correct for one-off prompts but destructive for topics the user will return to — project conventions, recurring debugging patterns, "we did X because Y" decisions. **Long-term memory catches the valuable subset before pruning erases it, distills it into a small set of Markdown files, and replays pointers to those files into future prompts where they are relevant.**

The distinction from agentic *skills* is deliberate: memories **describe** (this is how it was done, why, what bit us), they do **not prescribe** (do it this way). The AI decides whether to act on a surfaced memory in the context of the current prompt.

### Design invariants

1. **Focus Gate never calls an LLM directly.** Single binary, zero network, zero external dependencies, fully deterministic. The LLM work happens in the host (Claude Code or the CLI) using its own tools — Read, Write, Edit, Bash. Focus Gate *produces* structured prompts and *validates* structured replies.
2. **Memories are non-restrictive.** Surfaced as titled pointers (path + title + cosine similarity), never inlined as bodies on every prompt, never phrased as imperatives.

### Three-stage pipeline

```
A. SELECT  (Go, every prune cycle)
     Evaluate trees about to lose content. Bundle valuable ones as
     candidates. Append to pending_memories.json. Continue prune.

B. WRITE   (LLM, via slash command)
     User runs fg: memory promote. Focus Gate emits pending + existing
     manifest + instruction block. AI replies with fg: memory commit
     <json>. Focus Gate validates, stamps metadata, writes memory files.

C. SURFACE (Go, every prompt)
     Cosine-match prompt vector against manifest fingerprints. Inject
     top-K titles + paths into the Focus block. Touch counter bumps.
```

Stages A and C are live today. Stage B (promote/commit) is planned — see [docs/LONG_TERM_MEMORY_PLAN.md](docs/LONG_TERM_MEMORY_PLAN.md) §7 for the slash protocol and §14 Session C for rollout.

### Memory file format

One Markdown file per memory at `<projectDataDir>/memories/<id>.md`:

```markdown
---
schemaVersion: "1"
id: "mem_20260422_a8c3f1"
title: "JWT authentication with refresh-token rotation"
sources: ["tree_b21f"]
refs: ["cmd/api/auth.go", "internal/session/store.go"]
created: "2026-04-22T14:01:00Z"
updated: "2026-04-22T14:01:00Z"
topTerms: ["jwt", "session", "refresh", "token", "middleware"]
fingerprint: "jwt:0.4821 session:0.3912 refresh:0.3654 token:0.3201 ..."
vocabHash: "ed96811f47add21e"
touchedBy: 7
---

## What we did
Used JWT with RS256, 15-minute access tokens rotated on /auth/refresh.

## Why
RS256 over HS256 because multi-service rotation needs asymmetric keypairs.

## Pitfalls
Clock skew between services had to be tolerated up to 30s.

## Skills (historical, not prescriptive)
- Signing keys at `internal/auth/keys/`.
- Middleware short-circuits on `Authorization: Bearer`; missing header = 401.
```

**Two sections required:** `## What we did` and `## Why`. Everything else is free-form. Binary-managed fields (`created`, `updated`, `topTerms`, `fingerprint`, `vocabHash`, `touchedBy`) are recomputed on every write — whatever the LLM returns for them is ignored.

### Lifecycle — what you'll see

Below is what happens end-to-end once you've been working in a project long enough for the forest to prune. None of this requires manual intervention beyond running the promotion slash command.

**1. A substantive topic gets identified for preservation.** When a tree about to be pruned (or absorbed by cluster-merge) meets the score + floor thresholds, a candidate record is written to `<projectDataDir>/pending_memories.json`. You'll see a one-line nudge appear in the Focus block:

```
[Memory ↪ relevant prior context]
  mem_20260420_a1b2c3 [sim 0.68] Auth & session model
    → memories/mem_20260420_a1b2c3.md
  (1 topic(s) queued for memory promotion — run `fg: memory promote`)
```

**2. User promotes.** Running `fg: memory promote` (Session C — not yet shipped) emits the pending candidates + existing memory manifest + an instruction block telling the AI to reply with `fg: memory commit <fenced-json>`. The AI drafts Markdown bodies, decides merge-vs-create per candidate, and Focus Gate validates and persists.

**3. A related future prompt surfaces the memory.** When a later prompt's cosine against the manifest fingerprint clears `memory.surfaceThreshold` (default 0.35), the Focus block gets a titled pointer. The AI reads the file on demand using its own Read tool — Focus Gate never inlines memory bodies.

### Example: `fg: memory pending`

```
=== Pending Memory Candidates ===
  Queue: 2 candidate(s), last updated 5m ago

  TEMPID                              REASON  ACTION    ABSTRACTION
  cand_20260423_145626_moblu0jw       prune   merge→mem_20260420_a1b2...  token | authentica | session | jwt
  cand_20260423_150901_mobmuf05       merge   create    migrat | schema | user | email

Run `fg: memory promote` to generate an LLM-ready bundle (Session C).
Run `fg: memory discard <tempId|all>` to clear entries manually.
```

### Example: `fg: memory list`

```
=== Memories ===
  Directory: ~/.focus-gate/<slug>/memories
  Total: 3, manifest rebuilt 12m ago

  ID                          TOUCH   UPDATED       TITLE
  mem_20260420_a1b2c3         14      3.2h ago      Auth & session model
  mem_20260418_b71e02         8       1.8d ago      Test harness conventions
  mem_20260416_c93d14         2       5.1d ago      HTTP error-shape decision tree
```

### Slash commands

| Command | Purpose |
|:---|:---|
| `fg: memory list` | Table of all memories (id, touches, updated, title) |
| `fg: memory show <id-or-prefix>` | Pretty-print one memory's frontmatter + body |
| `fg: memory pending` | Queued candidates awaiting promotion |
| `fg: memory discard <tempId\|all>` | Clear pending without promoting |
| `fg: memory health` | Counts, stale-ref warnings, manifest state |
| `fg: memory promote` *(Session C)* | Render pending + manifest + LLM instructions |
| `fg: memory commit <json>` *(Session C)* | Persist the LLM's commit payload |
| `fg: memory forget <id>` *(Session C)* | Delete a memory file + manifest entry |

All `/focus memory ...` equivalents route to the same handlers.

### Memory configuration

| Parameter | Default | Description |
|:---|:---:|:---|
| `memory.enabled` | `true` | Master switch. `false` → no detection, no surface, slash subcommands respond "disabled." |
| `memory.dir` | `"memories"` | Sub-directory of the project data dir where files live. |
| `memory.surfaceThreshold` | 0.35 | Minimum cosine for a memory to surface on a prompt. |
| `memory.topK` | 2 | Max memories surfaced per prompt. |
| `memory.maxBlockChars` | 250 | Soft cap on the rendered surface-block length. |
| `memory.minLeaves` | 4 | Floor: tree must have this many real leaves to qualify as a candidate. |
| `memory.minPrompts` | 3 | Floor: tree must have this many indexed prompt contributions. |
| `memory.promotionThreshold` | 1.5 | Minimum `candidateScore` to queue a candidate. |
| `memory.rescueThreshold` | 1.2 | Trees below the score threshold but with historical `PeakScore` above this are rescued on prune (not on merge). |
| `memory.promotionCooldown` | `"4h"` | Per-tree cooldown preventing re-promotion storms. |
| `memory.pendingMaxAge` | `"168h"` | Candidates older than this are dropped from the queue on load. |
| `memory.mergeSuggestCosine` | 0.6 | Threshold at which a candidate's fingerprint is suggested to merge into an existing memory. |
| `memory.autoNudge` | `true` | Append the "`N topic(s) queued…`" line to the Focus block when pending is non-empty. |

### Tuning memory

- **Memories never surface?** Lower `memory.surfaceThreshold` to 0.20 and test with `fg: memory list` to see which memories are eligible. Or raise TF-IDF signal by writing memories with richer `## What we did` sections.
- **Too many candidates queuing?** Raise `memory.minLeaves` to 6 and `memory.promotionThreshold` to 2.0. The defaults are lenient to make Session A/B feel live; production sessions may want tighter filters.
- **Noisy surface blocks?** Lower `memory.topK` to 1 or raise `memory.maxBlockChars` to make the single entry more descriptive.
- **Feature blocks feel flaky?** Set `"memory": { "enabled": false }` to revert to pre-memory behaviour exactly.

### What Focus Gate does NOT do (explicit anti-goals)

- **No LLM HTTP client in Go.** Stage B is always slash-mediated. If a future version wants direct LLM calls, that's a separate feature behind a separate flag — the core invariant holds.
- **No body inlining at surface time.** Always pointers + titles + similarity. `contextLimit` stays the same; memory never balloons the Focus block.
- **No automatic memory deletion.** Stale → flagged in `fg: memory health`. Removal is always a deliberate user action via `fg: memory forget`.
- **No cross-project memories** in v1. Per-project directories only.

See [docs/LONG_TERM_MEMORY_PLAN.md](docs/LONG_TERM_MEMORY_PLAN.md) for the full design — invariants, detailed stage contracts, edge cases, test matrix, and rollout plan.

---

## Configuration

Configuration resolution order (first match wins):

1. `.focus-gate.json` in the current project directory
2. `$FOCUS_GATE_CONFIG` — explicit path via environment variable
3. `config.json` alongside the binary (global fallback)

Example `config.json`:

```json
{
  "memorySize": 100,
  "decayRate": 0.05,
  "similarity": { "extend": 0.55, "branch": 0.25 },
  "contextLimit": 600,
  "bubbleUpTerms": 6,
  "maxRefsPerNode": 5,
  "guideSize": 15,
  "sessionTimeout": 4.0,
  "mergeSimilarity": 0.6,
  "typoTolerance": {
    "enabled": true,
    "maxDistance": 2,
    "minWordLen": 5,
    "minEstablishedDF": 3
  },
  "memory": {
    "enabled": true,
    "dir": "memories",
    "surfaceThreshold": 0.35,
    "topK": 2,
    "maxBlockChars": 250,
    "minLeaves": 4,
    "minPrompts": 3,
    "promotionThreshold": 1.5,
    "rescueThreshold": 1.2,
    "promotionCooldown": "4h",
    "pendingMaxAge": "168h",
    "mergeSuggestCosine": 0.6,
    "autoNudge": true
  }
}
```

Only fields present in the file override defaults. A field explicitly set to `0` (e.g. `"sessionTimeout": 0` to disable session boundaries) is respected — it will not be replaced with the default.

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
| `typoTolerance.enabled` | `true` | Rewrite novel tokens to the nearest established corpus term (see below). `false` = no canonicalization. |
| `typoTolerance.maxDistance` | 2 | Max Levenshtein edits between a novel token and the term it may be rewritten to. |
| `typoTolerance.minWordLen` | 5 | Tokens shorter than this (after stemming) are never rewritten — prevents collisions like `auth` ↔ `each`. |
| `typoTolerance.minEstablishedDF` | 3 | A corpus term must appear in at least this many prior prompts before it can be a rewrite target. Blocks early-session cementing of wrong canonical forms. |
| `memory.*` | see above | Long-term memory parameters. Full table in the [Long-Term Memory](#long-term-memory) section. |

### Typo Tolerance

When the user types `envaeronment` after having already said `environment` a few times, Focus Gate canonicalizes the new token to the established one using Levenshtein edit distance. The canonicalization happens at tokenize time — the typo never enters the TF-IDF corpus, so `engine.json` stays clean and repeated typos of the same word keep remapping to the same canonical stem. No spell-check dictionary, no external dependencies.

Guards:

- Only tokens **shorter than `minWordLen`** characters (default 5) are considered — short words like `auth`, `each`, `run` are too prone to false merges.
- Only rewrites toward terms **already seen `minEstablishedDF` times** (default 3) — a single-occurrence term cannot become a canonical attractor.
- **Maximum edit distance** (default 2) is tight enough to catch realistic typos (`dificult` → `difficult`, `envaeron` → `environ`) without merging genuinely different words.

Set `"typoTolerance": { "enabled": false }` to disable the feature completely and restore the original tokenizer behaviour.

### Tuning

- **Too many unrelated trees?** Raise `similarity.branch` (e.g. 0.35)
- **Related prompts keep splitting?** Lower `similarity.branch` (e.g. 0.20) or lower `mergeSimilarity` (e.g. 0.5)
- **Old topics persist too long?** Raise `decayRate` (e.g. 0.10) or lower `sessionTimeout`
- **Memory fills too quickly?** Raise `memorySize` (e.g. 200)
- **Typos creating duplicate trees?** Raise `typoTolerance.maxDistance` to 3 if you spell aggressively wrong; lower `minEstablishedDF` to 2 if your sessions are short.
- **Typos silently merging unrelated words?** Raise `minWordLen` to 6 or `minEstablishedDF` to 5, or lower `maxDistance` to 1.
- **Unbounded growth concern?** If `sessionTimeout=0` *and* `memorySize` is very high, `engine.json` can grow indefinitely because pruning is what limits the TF-IDF corpus. Keep at least one of the two bounded.

---

## Architecture

```
cmd/focus/          Entry point (CLI, stdin/stdout, inspect/dry-run, /focus commands)
internal/
  text/             Tokenizer, stemmer, stop words, file-ref extraction, typo canonicalizer
  tfidf/            TF-IDF engine, sparse vectors, cosine similarity
  forest/           Node, Tree, Forest, heap-based pruning, peak-score tracking
  gate/             Focus Gate classifier (classify, apply, bubble-up, merge, dry-run)
  guide/            AI response tracking (ring buffer + forest reinforcement)
  memory/           Long-term memory (Memory, Manifest, Surface, candidate selection, pending queue)
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
| `pending_memories.json` | Append-only queue of candidates awaiting LLM promotion (see [Long-Term Memory](#long-term-memory)) |
| `memories/` | Long-term memory Markdown files + `index.json` manifest |

---

## License

MIT
