# Focus Gate — User Guide

Focus Gate keeps your AI coding assistant aware of what you've been working on.
It runs silently as a Claude Code hook, tracks your prompt history as a semantic
forest, and injects a compact context block before every message you send.

---

## Contents

- [What It Does](#what-it-does)
- [Install](#install)
- [Set Up the Hook](#set-up-the-hook)
- [How Context Looks](#how-context-looks)
- [Configuration](#configuration)
- [In-Chat Commands](#in-chat-commands)
  - [status](#focus-status)
  - [inspect](#focus-inspect)
  - [tree](#focus-tree)
  - [terms](#focus-terms)
  - [score](#focus-score)
  - [last](#focus-last)
  - [health](#focus-health)
  - [help](#focus-help)
- [CLI Flags](#cli-flags)
- [Understanding the Forest](#understanding-the-forest)
- [How Classification Works](#how-classification-works)
- [Guide: AI Response Tracking](#guide-ai-response-tracking)
- [Session Boundaries](#session-boundaries)
- [Cluster Merging](#cluster-merging)
- [File Ref Tracking](#file-ref-tracking)
- [Per-Project Isolation](#per-project-isolation)
- [Data Files](#data-files)
- [Tuning](#tuning)
- [Troubleshooting](#troubleshooting)

---

## What It Does

Every time you press Enter in Claude Code, Focus Gate:

1. Reads your prompt (JSON from stdin)
2. Compares it to your accumulated intent forest using TF-IDF cosine similarity
3. Classifies it as a continuation, a new branch, or a new topic
4. Updates the forest and generates a short context summary
5. Writes the context to stdout — Claude prepends it to your message

The AI never sees Focus Gate. It only sees your prompt, enriched with a block
that says "here is what this user has been focused on recently."

---

## Install

**From source** (requires Go 1.23+):

```bash
git clone https://github.com/kuandriy/focus-gate
cd focus-gate
go build -o focus-gate ./cmd/focus
```

**From Releases**: download the binary for your platform and make it executable.

```bash
chmod +x focus-gate
```

---

## Set Up the Hook

Add Focus Gate to `.claude/settings.local.json` in your project directory. This
file is not committed by default and applies only to your machine.

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

Use the absolute path to the binary. Once saved, the hook is active immediately
— no restart needed.

> **Multiple projects**: the same binary works everywhere. Each project gets its
> own isolated state directory (see [Per-Project Isolation](#per-project-isolation)).

---

## How Context Looks

Each prompt you send is preceded by a block like this:

```
[Focus | 23 prompts | 18/100 mem | 3 trees]
  [0.95] token | authentica | session | jwt
    - add refresh token rotation
    - fix the session expiry bug
  [0.82] migrat | schema | user | email
    - add index on email column
    @ internal/auth/middleware.go, internal/db/migrate.go
  [0.31] readme | document | project
[/Focus]
Guide:
  - Implemented JWT auth with RS256 signing [auth middleware]
  - Created users migration with email index [db schema]
[/Guide]
```

| Part | Meaning |
|:---|:---|
| `[Focus | 23 prompts | 18/100 mem | 3 trees]` | Stats header |
| `[0.95]` | Tree's decay-weighted score — higher = more recent/revisited |
| `token | authentica | session | jwt` | Root node: top terms bubbled up from leaves |
| `- add refresh token rotation` | Recent leaf node (actual prompt text) |
| `@ internal/auth/middleware.go, ...` | Top file paths referenced in this tree |
| `Guide:` | Recent AI response summaries tracked per-topic |

The entire block is capped at `contextLimit` characters (default 600). The
most important tree always fits; additional trees are added until the budget
runs out.

---

## Configuration

Focus Gate resolves configuration in this order (first match wins):

1. `.focus-gate.json` in the current project directory (per-project override)
2. `$FOCUS_GATE_CONFIG` — explicit path via environment variable
3. `config.json` alongside the binary (global fallback, shared across projects)

The per-project file lets you tune thresholds for a particular codebase
without affecting your other projects. Create `config.json`:

```json
{
  "memorySize": 100,
  "decayRate": 0.05,
  "similarity": {
    "extend": 0.55,
    "branch": 0.25
  },
  "contextLimit": 600,
  "bubbleUpTerms": 6,
  "maxRefsPerNode": 5,
  "guideSize": 15,
  "sessionTimeout": 4.0,
  "mergeSimilarity": 0.6
}
```

You can include only the fields you want to change — every absent field keeps
its default. A field explicitly set to `0` (e.g. `"sessionTimeout": 0` to
disable session boundaries) is respected as-is.

### Parameters

| Parameter | Default | Description |
|:---|:---:|:---|
| `memorySize` | 100 | Maximum total nodes across all trees |
| `decayRate` | 0.05 | Exponential decay per hour. Higher = faster forgetting |
| `similarity.extend` | 0.55 | Cosine threshold to add near an existing leaf |
| `similarity.branch` | 0.25 | Cosine threshold to branch off a tree root |
| `contextLimit` | 600 | Max characters in the injected context block |
| `bubbleUpTerms` | 6 | Terms per parent node in bubble-up abstractions |
| `maxRefsPerNode` | 5 | Max file paths stored per node |
| `guideSize` | 15 | Max AI response summaries in the guide |
| `sessionTimeout` | 4.0 | Hours of inactivity before session boundary (0 = off) |
| `mergeSimilarity` | 0.6 | Root cosine threshold for merging similar trees (0 = off) |

---

## In-Chat Commands

Type any `/focus <sub>` command directly in the chat. The command is
intercepted before classification — **no state is modified** — and the
output appears inline.

**VSCode users:** the Claude Code extension's slash-command picker swallows
anything that starts with `/`, so commands never reach the hook there. Use
the `fg:` alias instead (`fg: status`, `fg: health`, `fg: tree 0`, etc.).
Both prefixes route to the same handler; the CLI also accepts `fg:`.

All commands are case-insensitive: `/focus STATUS`, `/Focus Tree 0`,
`fg: Status`, and `FG: tree 0` are equivalent.

---

### `/focus status`

Shows the same context block the AI would currently see.

```
/focus status
```

Useful for checking what Focus Gate is injecting without scrolling up through
the conversation.

---

### `/focus inspect`

Full structured state dump: every tree with its node hierarchy, TF-IDF corpus
stats, and guide entries with reinforcement status.

```
/focus inspect
```

Example output (truncated):

```
=== Focus Gate Inspect ===

--- Config ---
  memorySize:        100
  decayRate:         0.050
  similarity.extend: 0.550
  similarity.branch: 0.250
  contextLimit:      600
  bubbleUpTerms:     6
  guideSize:         15

--- Forest: 2 trees, 8/100 nodes, 23 prompts ---
  created:    2026-04-18 09:12:04
  lastUpdate: 2026-04-20 14:33:17

  Tree #0 [id=a1b2c3d4e5f6] score=0.952
    4 nodes, 3 leaves, created 2026-04-18 09:12:04
    [root] a1b2c3d4  d=0 w=1.58 f=2 idx=- s=0.952
    "token | authentica | session | jwt"
    ├── b2c3d4e5  d=1 w=1.00 f=1 idx=Y s=0.871
    │   "add JWT authentication to the API"
    ├── c3d4e5f6  d=1 w=1.00 f=1 idx=Y s=0.843
    │   "fix the session expiry bug"
    └── d4e5f6g7  d=1 w=1.58 f=2 idx=Y s=0.921
        "add refresh token rotation"
...
```

Node columns: `d` = depth, `w` = weight (log₂ of frequency), `f` = frequency,
`idx` = indexed in TF-IDF (Y/−), `s` = decay score.

---

### `/focus tree`

Without an argument, lists all trees sorted by score:

```
/focus tree
```

With a numeric index or partial tree ID, shows a deep-dive into that tree:

```
/focus tree 0
/focus tree a1b2
```

The deep-dive includes:
- Full node hierarchy with scores
- Root vector terms and their TF-IDF weights
- Pruning candidates (lowest-score leaves shown first)

---

### `/focus terms`

Lists TF-IDF vocabulary sorted by document frequency. Default: top 30 terms.

```
/focus terms
/focus terms 50
```

Example:

```
=== TF-IDF Terms (top 30 by DF) ===
  total docs: 23, unique terms: 87

  Term                    DF    IDF
  authentica               8   1.474
  token                    6   1.738
  migrat                   5   1.906
  session                  4   2.130
  ...
```

`IDF` is high for rare terms (more discriminating) and low for common ones.

---

### `/focus score`

Classifies a prompt without modifying any state. Shows exactly how each tree
would be scored.

```
/focus score add rate limiting to the auth middleware
```

Example:

```
=== Score ===
  Prompt: "add rate limiting to the auth middleware"
  Tokens: [rate limit auth middlewar]

  TF-IDF Vector (4 terms):
    authentica           0.4821
    middlewar            0.3912
    rate                 0.2104
    limit                0.1876

  Thresholds: extend >= 0.550, branch >= 0.250

  Tree #0 "token | authentica | session | jwt"
    Root a1b2c3d4  cosine=0.6203
    Leaf b2c3d4e5  cosine=0.5918  "add JWT authentication to the API"  <- BEST
    Leaf c3d4e5f6  cosine=0.3102  "fix the session expiry bug"
    Leaf d4e5f6g7  cosine=0.4817  "add refresh token rotation"

  Tree #1 "migrat | schema | user | email"
    Root e5f6g7h8  cosine=0.0341
    Leaf f6g7h8i9  cosine=0.0212  "add index on email column"

  Result: extend (score=0.6203)
    Would add as sibling near leaf b2c3d4e5 in Tree #0.
```

Use this before sending an important prompt to verify it classifies where you
expect, or to understand why a previous prompt went to the wrong tree.

If the top score falls in a narrow "near-miss" band below `branch` (e.g. 0.15
to 0.25 when `branch = 0.25`), the dry-run surfaces a warning explaining that
a typo or rare-term mismatch may have prevented a match you expected. The
warning does not change behaviour — it is strictly diagnostic.

---

### `/focus last`

Shows the most recent classifications — action, top similarity, and snippet of
the prompt that triggered each. Useful when a prompt lands in the wrong tree
and you want to understand why without running a dry-run.

```
/focus last
```

Example:

```
=== Recent Classifications ===
  #5  extend    0.618  "add refresh token rotation"      -> tree#0
  #4  branch    0.312  "write migration for users table" -> tree#1
  #3  new       0.000  "update the readme section"       -> tree#2
  #2  continue  0.000  "yes"                             -> tree#0 (recent)
  #1  extend    0.554  "fix the session expiry bug"      -> tree#0
```

---

### `/focus health`

System diagnostics: memory pressure, tree temperature, TF-IDF noise ratio,
pruning forecast.

```
/focus health
```

Example:

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

**HOT/WARM/COLD** thresholds:
- HOT: score > 0.5 — actively being worked on
- WARM: 0.1 – 0.5 — recent but cooling
- COLD: < 0.1 — likely to be pruned soon

---

### `/focus help`

Lists all available commands.

```
/focus help
```

---

## CLI Flags

Run Focus Gate directly from a terminal for scripting, debugging, or a quick
check without opening Claude Code.

```bash
# Show current context
./focus-gate --status

# Full state dump (human-readable)
./focus-gate --inspect

# Full state dump (machine-readable JSON)
./focus-gate --inspect --json

# Classify a prompt without mutating state
./focus-gate --dry-run "add rate limiting to the auth middleware"

# Dry-run as JSON
./focus-gate --dry-run "add rate limiting to auth" --json

# Reset all tracking data for this project
./focus-gate --reset

# Override the data directory (useful for scripting)
./focus-gate --data-dir /path/to/dir --status

# List all per-project data directories (name, hash, size, last-modified)
./focus-gate --list-projects

# Suppress stderr logging
./focus-gate --quiet --status
```

**Hook mode** (how Claude Code calls it) reads a JSON object from stdin:

```bash
echo '{"prompt": "add JWT auth to the API", "transcript_path": ""}' | ./focus-gate
```

The `transcript_path` field is optional. When provided, Focus Gate reads the
last assistant message from the Claude Code transcript file and adds it to the
Guide.

---

## Understanding the Forest

The **forest** is Focus Gate's memory. It is a collection of **trees**, where
each tree represents a distinct topic cluster. Within a tree:

- The **root node** holds an abstract summary: the top 6 terms bubbled up from
  children, pipe-separated (e.g. `token | authentica | session | jwt`).
- **Leaf nodes** hold actual prompt text.
- A prompt that closely matches an existing leaf is added as a sibling near it.
- A prompt broadly related to a tree's theme gets added under the root.
- An unrelated prompt starts a new tree.

### Node Scoring

Every node has a **survival score** used during pruning:

```
score = weight × recency × depthFactor

weight      = log₂(frequency + 1)   — how often revisited
recency     = e^(−decayRate × ageHours)  — how recent
depthFactor = 1 / (1 + depth × 0.15)    — root > leaf
```

At the default decay rate (0.05), a node untouched for 24 h retains ~30%
of its recency score. After 48 h: ~9%.

Nodes you keep revisiting accumulate frequency, which counteracts decay.
A topic you return to every few hours stays alive indefinitely.

### Pruning

When the node count exceeds `memorySize`, the lowest-scoring leaves are removed
one at a time using a min-heap. When a leaf's removal makes its parent a new
leaf, the parent is pushed onto the heap as the next candidate. Only nodes that
were indexed in TF-IDF trigger a `RemoveDocument` call — synthetic root
abstractions are never in the corpus.

---

## How Classification Works

Each prompt goes through this pipeline:

1. **Clean**: strip IDE/system XML tags injected by the editor context system.
2. **Tokenize**: lowercase, split on boundaries, stem each word, remove stop
   words and single-character tokens.
3. **Vectorize**: compute a TF-IDF vector — each token weighted by how rare it
   is across your prompt history.
4. **Classify**: compare the vector against every tree's root and every leaf
   using cosine similarity. The best score determines the action:

| Score | Action | What happens |
|:---:|:---:|:---|
| >= 0.55 (`extend`) | **Extend** | Added as sibling near the matching leaf |
| 0.25 – 0.55 (`branch`) | **Branch** | Added as child under the tree root |
| < 0.25 | **New tree** | A new topic tree is created |
| empty vector | **Continue** | Attached to the most-recently-active tree |

The **continue** action handles terse prompts like `fix`, `yes`, `run it`, or
`continue` that tokenize to nothing after stop-word filtering. Instead of
spawning a noise tree for every one-word follow-up, Focus Gate treats them as
implicit continuations of whatever you were just working on. If no tree exists
yet, the prompt is skipped entirely.

5. **Apply**: insert the new node, then run **bubble-up** — regenerate every
   parent's content bottom-up as the top N terms across its children, weighted
   by cross-child presence × IDF.
6. **Prune**: if node count exceeds `memorySize`, remove lowest-scoring leaves.
7. **Merge**: if any two tree roots score >= `mergeSimilarity` against each
   other, merge the smaller tree into the larger.
8. **Render**: generate the context block and print it to stdout.

### Stemming

Words are stemmed before vectorization so variants group together:
`authentication`, `authenticated`, `authenticating` → all stem to `authentica`.

The stemmer runs two passes: first strips plurals (`-ies`, `-es`, `-s`), then
strips derivational suffixes (`-ization`, `-tion`, `-ment`, `-ing`, `-ed`, etc.).
An override map prevents false conflations — `authorization` does not stem to
`author`.

### Cosine Similarity

Two vectors are compared as the cosine of the angle between them. This is
magnitude-independent: a short 3-word prompt and a detailed 20-word prompt
score high similarity if they share the same key terms.

A score of 1.0 means identical topic; 0.0 means completely unrelated.

---

## Guide: AI Response Tracking

When `transcript_path` is provided by Claude Code, Focus Gate reads the last
assistant message from the conversation transcript and adds a 200-character
summary to the **Guide** — a ring buffer of up to `guideSize` entries.

The Guide serves two purposes:

**1. Context output** — recent summaries appear at the end of the context block:

```
Guide:
  - Implemented JWT auth with RS256 signing [auth middleware]
  - Created users migration with email index [db schema]
[/Guide]
```

Only entries whose linked intent node still exists in the forest are shown
(pruned topics disappear from the guide output automatically).

**2. Forest reinforcement** — before each new prompt is classified, unreinforced
guide entries are vectorized and matched against tree roots by cosine similarity.
The best-matching root above the `branch` threshold gets a `Touch()` — its
frequency increments and its recency resets to now. This makes actively-discussed
trees harder to prune.

Each entry is marked as reinforced after processing and is never double-counted,
even across binary restarts.

---

## Session Boundaries

If the gap between your last prompt and a new one exceeds `sessionTimeout`
hours (default: 4), all node frequencies are **halved**. This reduces weights
(via log₂) without destroying accumulated knowledge, making old trees easier
to prune and new prompts more likely to create fresh trees.

Set `"sessionTimeout": 0` in config to disable this behaviour.

---

## Cluster Merging

After each prompt, Focus Gate compares every pair of tree roots. If any two
roots score >= `mergeSimilarity` (default: 0.6) against each other, the smaller
tree's leaves are re-parented under the larger tree's root, and bubble-up is
re-run. Only one merge fires per prompt to avoid cascades.

This prevents topic fragmentation when closely related prompts happen to cross
the `branch` threshold and create separate trees.

Set `"mergeSimilarity": 0` to disable merging.

---

## File Ref Tracking

When you mention a file path in a prompt (e.g. `fix the bug in src/auth/middleware.go`
or reference `` `schema.sql` `` in backticks), Focus Gate extracts it and stores it
on the node. In the context output, the top 3 most-referenced paths per tree
appear as:

```
    @ src/auth/middleware.go, internal/db/schema.sql
```

Supported extensions cover common code, config, and infra files (Go, TypeScript,
Python, SQL, YAML, Terraform, etc.). URLs and Go module import paths are filtered
out. Extracted paths are validated against the project working directory with
`os.Stat`; paths that do not exist on disk are dropped before being stored, so
rhetorical mentions never accumulate as false refs.

Up to `maxRefsPerNode` (default: 5) paths are stored per node.

---

## Per-Project Isolation

Focus Gate stores state in `~/.focus-gate/<12-char-hash>/` where the hash is
the first 12 hex characters of SHA-256 of your current working directory. Each
project gets a separate namespace automatically — switching projects means
switching contexts.

You can override the data directory three ways (checked in priority order):

1. `--data-dir /path/to/dir` flag
2. `FOCUS_GATE_DATA_DIR` environment variable
3. Default per-project isolation (`~/.focus-gate/<hash>/`)

Configuration is resolved in this order: a `.focus-gate.json` in the project
directory overrides `$FOCUS_GATE_CONFIG`, which overrides the global
`config.json` next to the binary. Drop a per-project file when you want to
tune thresholds for that codebase without touching your global defaults.

---

## Data Files

All state is written as indented JSON using atomic saves (write to `.tmp`, then
rename). On startup, any orphaned `.tmp` files from interrupted saves are
automatically recovered.

Every persisted file carries a `schemaVersion` field at its top level. When
Focus Gate loads a file, it compares the version on disk against the version
the binary expects. A mismatch logs a warning to stderr and the file is treated
as empty state rather than partially unmarshaled — preventing silent data
corruption across upgrades.

Concurrent invocations of the hook (e.g. two rapid `UserPromptSubmit` events
from the editor) acquire a file lock on the data directory before reading or
writing state, so simultaneous writes cannot race and lose prompts.

| File | Contents |
|:---|:---|
| `~/.focus-gate/<hash>/intent.json` | Intent forest (trees, nodes, scores) |
| `~/.focus-gate/<hash>/engine.json` | TF-IDF document frequency counts |
| `~/.focus-gate/<hash>/guide.json` | AI response summaries with reinforcement state |
| `~/.focus-gate/<hash>/.lock` | File lock (zero-byte, advisory) |
| `.focus-gate.json` (project root) | Per-project config override |
| `config.json` (next to binary) | Global configuration fallback |

These files are human-readable JSON and can be inspected or deleted manually.
A corrupt file logs an error to stderr and the system continues with empty
state — your prompts are never blocked.

---

## Tuning

### Too many unrelated trees forming?

Lower the branch threshold so more prompts join existing trees:

```json
{ "similarity": { "branch": 0.20 } }
```

### Related prompts creating separate trees?

Lower merge similarity so closer trees get merged sooner:

```json
{ "mergeSimilarity": 0.5 }
```

Or lower the branch threshold:

```json
{ "similarity": { "branch": 0.20 } }
```

### Old topics lingering too long?

Increase decay rate to forget faster:

```json
{ "decayRate": 0.10 }
```

Or reduce the session timeout so old sessions get penalized sooner:

```json
{ "sessionTimeout": 2.0 }
```

### Memory fills too quickly?

Raise `memorySize`:

```json
{ "memorySize": 200 }
```

### Context block too long / too short?

```json
{ "contextLimit": 800 }
```

### Too many root terms obscuring meaning?

Reduce `bubbleUpTerms` for tighter abstractions:

```json
{ "bubbleUpTerms": 4 }
```

---

## Troubleshooting

**Focus Gate isn't running at all**

Check that the hook path in `.claude/settings.local.json` is an absolute path
and the binary is executable (`chmod +x focus-gate`).

Test it manually:

```bash
echo '{"prompt":"test prompt"}' | /path/to/focus-gate
```

You should see a context block on stdout.

**Context block appears empty or just the header**

This is normal for your first few prompts — the forest needs a few entries
before it has anything meaningful to show. Send a few more prompts and check
`/focus status`.

**A prompt went to the wrong tree**

Use `/focus score "your prompt"` to see the exact cosine scores. If the right
tree scores lower than expected:
- The tree's root terms may not overlap with your prompt's vocabulary yet.
  Send a few more related prompts to build up the tree's content.
- Try lowering `similarity.branch` so the prompt more easily joins existing trees.

**Too many singleton trees**

Lower `similarity.branch` (e.g. `0.20`) or raise `mergeSimilarity` (e.g. `0.65`)
to encourage consolidation.

**Stems look wrong** (e.g. important terms disappearing)

Check `/focus terms` to see what vocabulary has been indexed. The stemmer is
aggressive — `container` → `contain`, `server` → `server` (intentionally not
stripped because `-er` exclusion). This is by design. The TF-IDF layer
compensates: if a stemmed form is rare in your history, it still gets a high
IDF weight.

**Reset everything and start fresh**

```bash
./focus-gate --reset
```

This removes `intent.json`, `engine.json`, and `guide.json` for the current
project. Config is not touched.
